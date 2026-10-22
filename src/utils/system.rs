#[cfg(unix)]
mod imp {
    use crate::error::NviError;
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;
    use procfs::process::Process;

    /// Executable name for a pid (GUI process list). argv[0]'s file name
    /// survives spaced install paths; falls back to comm, then `[pid]`.
    pub fn get_process_name(pid: u32) -> Option<String> {
        let process = Process::new(pid as i32).ok()?;
        let name = process
            .cmdline()
            .ok()
            .and_then(|cmd| cmd.into_iter().next())
            .map(|argv0| argv0.rsplit('/').next().unwrap_or(&argv0).to_string())
            .filter(|n| !n.is_empty())
            .or_else(|| process.stat().ok().map(|stat| stat.comm))
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| format!("[{pid}]"));
        Some(name)
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
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

    thread_local! {
        static PROBE: RefCell<System> = RefCell::new(System::new());
    }

    /// Refresh kind for the process probe. The default `refresh_processes`
    /// kind omits `cmd`, and on Windows an unrefreshed name reads empty.
    /// `OnlyIfNotSet` matches `ProcessRefreshKind::everything()`'s
    /// convention: fetched once per process lifetime rather than re-read
    /// every tick. CPU, memory, disk, exe and user are not read here.
    fn process_refresh_kind() -> ProcessRefreshKind {
        ProcessRefreshKind::nothing().with_cmd(UpdateKind::OnlyIfNotSet)
    }

    /// Executable name for a pid, resolved in one probe pass. `Process::name()`
    /// is already the basename, so spaced install paths (`C:\Program Files\...`)
    /// need no argv parsing. GUI process list only.
    pub fn get_process_name(pid: u32) -> Option<String> {
        PROBE.with_borrow_mut(|system| {
            let sid = Pid::from_u32(pid);
            system.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[sid]),
                true,
                process_refresh_kind(),
            );
            let process = system.process(sid)?;
            let name = process.name().to_string_lossy().into_owned();
            Some(if name.is_empty() {
                format!("[{pid}]")
            } else {
                name
            })
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
