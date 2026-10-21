#[cfg(unix)]
mod imp {
    use crate::error::NviError;
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;
    use nix::unistd::{Uid, User};
    use procfs::process::Process;

    /// Executable name + username for a pid (GUI process list). argv[0]'s
    /// file name survives spaced install paths; falls back to comm, then `[pid]`.
    pub fn get_process_identity(pid: u32) -> Option<(String, String)> {
        let process = Process::new(pid as i32).ok()?;
        let uid = process.uid().ok()?;
        // An unresolvable uid (rootless container mapping) still shows the
        // row with "?" — only a dead pid drops it. Mirrors the Windows imp.
        let username = User::from_uid(Uid::from_raw(uid))
            .ok()
            .flatten()
            .map(|u| u.name)
            .unwrap_or_else(|| "?".to_string());
        let name = process
            .cmdline()
            .ok()
            .and_then(|cmd| cmd.into_iter().next())
            .map(|argv0| argv0.rsplit('/').next().unwrap_or(&argv0).to_string())
            .filter(|n| !n.is_empty())
            .or_else(|| process.stat().ok().map(|stat| stat.comm))
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| format!("[{pid}]"));
        Some((name, username))
    }

    pub fn kill_selected_process(pid: u32, command: &str) -> Result<(), NviError> {
        match kill(Pid::from_raw(pid as i32), Signal::SIGTERM) {
            Ok(_) => Ok(()),
            Err(nix::Error::EPERM) => Err(NviError::Process(format!(
                "Permission denied to terminate process {pid} ({command})"
            ))),
            Err(e) => Err(NviError::Process(format!(
                "Failed to terminate process {pid} ({command}): {e}"
            ))),
        }
    }
}

#[cfg(windows)]
mod imp {
    use crate::error::NviError;
    use std::cell::RefCell;
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind, Users};

    /// Shared sysinfo probe: `System` for process data, `Users` for username
    /// lookup. Held in a `thread_local!` so the public signature of
    /// `get_process_identity` stays unchanged (the whole app is
    /// single-threaded for these calls). Constructing a `System::new_all()`
    /// per lookup would re-enumerate every process per GPU process per
    /// refresh; the probe refreshes only what is needed.
    pub struct PlatformProbe {
        system: System,
        users: Users,
        // Whether `users` has been populated yet. `Users::new()` is empty by
        // design (see `ensure_users`); the LSA-backed enumeration runs lazily on
        // the first real production lookup, not at construction. This keeps the
        // pure-logic unit tests (which drive an empty `System`) off the live
        // Windows backend entirely.
        users_refreshed: bool,
    }

    impl PlatformProbe {
        fn new() -> Self {
            Self {
                system: System::new(),
                users: Users::new(),
                users_refreshed: false,
            }
        }

        /// Populate the user table once per probe lifetime. Logged-in users
        /// barely change during a session, and `Users::refresh()` is the one
        /// sysinfo call that touches the Windows LSA (`LsaEnumerateLogonSessions`)
        /// — deferring it (instead of running it at construction, as
        /// `Users::new_with_refreshed_list` would) keeps username lookup
        /// pay-per-use and the unit tests off the live system backend.
        fn ensure_users(&mut self) {
            if !self.users_refreshed {
                self.users.refresh();
                self.users_refreshed = true;
            }
        }
    }

    thread_local! {
        static PROBE: RefCell<PlatformProbe> = RefCell::new(PlatformProbe::new());
    }

    /// Run `f` against the thread-local probe.
    fn with_probe<R>(f: impl FnOnce(&mut PlatformProbe) -> R) -> R {
        PROBE.with_borrow_mut(f)
    }

    /// Refresh kind for the process probe. The default `refresh_processes`
    /// kind omits `cmd` and `user`, and on Windows `Process::user_id()` is
    /// only populated when the refresh kind asks for it — without this the
    /// name and username columns stay empty. `OnlyIfNotSet` matches
    /// `ProcessRefreshKind::everything()`'s convention: fetched once per
    /// process lifetime rather than re-read every tick. CPU, memory, disk
    /// and exe are not read here.
    fn process_refresh_kind() -> ProcessRefreshKind {
        ProcessRefreshKind::nothing()
            .with_cmd(UpdateKind::OnlyIfNotSet)
            .with_user(UpdateKind::OnlyIfNotSet)
    }

    /// Executable name + username for a pid, resolved in one probe pass.
    /// `Process::name()` is already the basename, so spaced install paths
    /// (`C:\Program Files\...`) need no argv parsing. GUI process list only.
    pub fn get_process_identity(pid: u32) -> Option<(String, String)> {
        with_probe(|probe| {
            let sid = Pid::from_u32(pid);
            probe.system.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[sid]),
                true,
                process_refresh_kind(),
            );
            probe.ensure_users();
            let process = probe.system.process(sid)?;
            let username = process
                .user_id()
                .and_then(|uid| probe.users.get_user_by_id(uid))
                .map(|u| u.name().to_string())
                .unwrap_or_else(|| "?".to_string());
            let name = process.name().to_string_lossy().into_owned();
            let name = if name.is_empty() {
                format!("[{pid}]")
            } else {
                name
            };
            Some((name, username))
        })
    }

    pub fn kill_selected_process(pid: u32, command: &str) -> Result<(), NviError> {
        // sysinfo's Process::kill() shells out to `taskkill /PID <pid> /F` as
        // well but returns only a bool, which cannot distinguish permission
        // denial from other failures. Calling taskkill directly keeps the
        // unix branch's message set via its exit codes: 0 success, 1 access
        // denied, 128 no such process. stderr text is deliberately not
        // matched — it is localized on non-English Windows, the exit codes
        // are not. Output is captured (not inherited) so taskkill's own
        // ERROR/SUCCESS lines cannot corrupt the GUI.
        let output = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .output()
            .map_err(|e| {
                NviError::Process(format!(
                    "Failed to terminate process {pid} ({command}): {e}"
                ))
            })?;
        match output.status.code() {
            Some(0) => Ok(()),
            Some(1) => Err(NviError::Process(format!(
                "Permission denied to terminate process {pid} ({command})"
            ))),
            Some(128) => Err(NviError::Process(format!(
                "Failed to terminate process {pid} ({command}): no such process"
            ))),
            _ => Err(NviError::Process(format!(
                "Failed to terminate process {pid} ({command})"
            ))),
        }
    }
}

pub use imp::*;
