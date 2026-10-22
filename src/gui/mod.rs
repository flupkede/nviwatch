//! gpui window app: NVIDIA GPU monitoring without the terminal.
//!
//! A background thread owns `Nvml` and pushes snapshots over a channel; the
//! window drains it on a timer loop. NVML stays off the UI executor so a slow
//! driver call can never stall a frame.

mod data;
mod graph;
mod render;
mod vram;

use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

use gpui::{
    App, Application, Bounds, Context, KeyBinding, TitlebarOptions, WindowBounds, WindowOptions,
    actions, prelude::*, px, size,
};
use nvml::Nvml;
use nvml::device::Device;
use nvml::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml::struct_wrappers::device::{ProcessInfo, ProcessUtilizationSample};

use crate::Result;
use crate::utils::system::get_process_name;
use data::{GpuProc, MemSample, ProcKind, SortKey, push_samples};

actions!(nviwatch, [Quit, SelectUp, SelectDown, KillSelected]);

/// One poll of one GPU.
#[derive(Clone, Debug)]
pub(crate) struct GpuSnapshot {
    pub index: usize,
    pub name: String,
    pub temperature: u32,
    pub utilization: u32,
    pub memory_used: u64,
    pub memory_total: u64,
    pub power_usage: u32,
    pub power_limit: u32,
    pub clock_freq: u32,
    pub processes: Vec<GpuProc>,
    /// Sum of compute-process VRAM (the LLM segment of the memory graph).
    pub compute_bytes: u64,
    /// Sum of graphics-process VRAM (the video segment).
    pub graphics_bytes: u64,
}

/// Merge the NVML process lists into typed rows: name resolved per
/// platform, kind decided by name, VRAM from D3DKMT with the NVML figure as
/// fallback (D3DKMT is Windows-only; NVML reports "NA" there on WDDM) and
/// SM utilization keyed by pid. Dead processes drop silently. `vram` and
/// `utils` are injected so the chains are testable without hardware.
fn enrich_with(
    vram: &std::collections::HashMap<u32, u64>,
    utils: &std::collections::HashMap<u32, u32>,
    compute: Vec<ProcessInfo>,
    graphics: Vec<ProcessInfo>,
) -> Vec<GpuProc> {
    let mut nvml_mem: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for info in compute.iter().chain(graphics.iter()) {
        let bytes = match info.used_gpu_memory {
            nvml::enums::device::UsedGpuMemory::Used(bytes) => bytes,
            nvml::enums::device::UsedGpuMemory::Unavailable => 0,
        };
        nvml_mem
            .entry(info.pid)
            .and_modify(|existing| *existing = (*existing).max(bytes))
            .or_insert(bytes);
    }
    union_pids(&compute, &graphics)
        .into_iter()
        .filter_map(|pid| {
            let name = get_process_name(pid)?;
            Some(GpuProc {
                pid,
                used_gpu_memory: vram
                    .get(&pid)
                    .copied()
                    .or_else(|| nvml_mem.get(&pid).copied())
                    .unwrap_or(0),
                kind: data::classify_by_name(&name),
                name,
                sm_util: utils.get(&pid).copied(),
            })
        })
        .collect()
}

/// Pids from both NVML lists, deduped, first-seen order.
fn union_pids(compute: &[ProcessInfo], graphics: &[ProcessInfo]) -> Vec<u32> {
    let mut pids: Vec<u32> = Vec::new();
    for info in compute.iter().chain(graphics.iter()) {
        if !pids.contains(&info.pid) {
            pids.push(info.pid);
        }
    }
    pids
}

/// The NVML reads one snapshot needs, collapsed to `Option` — the degrade
/// mapping only distinguishes success from failure. Behind a trait so the
/// per-field fallback is unit-testable without a GPU (mirrors `enrich_with`).
trait DeviceSource {
    fn name(&self) -> Option<String>;
    fn temperature(&self) -> Option<u32>;
    fn utilization(&self) -> Option<u32>;
    fn memory(&self) -> Option<(u64, u64)>;
    fn power_usage(&self) -> Option<u32>;
    fn power_limit(&self) -> Option<u32>;
    fn clock(&self) -> Option<u32>;
    fn running_compute_processes(&self) -> Option<Vec<ProcessInfo>>;
    fn running_graphics_processes(&self) -> Option<Vec<ProcessInfo>>;
    /// Per-pid SM utilization samples (empty on platforms/drivers that do
    /// not report them).
    fn process_utils(&self) -> Option<Vec<ProcessUtilizationSample>>;
}

impl DeviceSource for Device<'_> {
    fn name(&self) -> Option<String> {
        self.name().ok()
    }

    fn temperature(&self) -> Option<u32> {
        self.temperature(TemperatureSensor::Gpu).ok()
    }

    fn utilization(&self) -> Option<u32> {
        self.utilization_rates().ok().map(|u| u.gpu)
    }

    fn memory(&self) -> Option<(u64, u64)> {
        self.memory_info().ok().map(|m| (m.used, m.total))
    }

    fn power_usage(&self) -> Option<u32> {
        self.power_usage().ok()
    }

    fn power_limit(&self) -> Option<u32> {
        self.enforced_power_limit().ok()
    }

    fn clock(&self) -> Option<u32> {
        self.clock_info(Clock::Graphics).ok()
    }

    fn running_compute_processes(&self) -> Option<Vec<ProcessInfo>> {
        self.running_compute_processes().ok()
    }

    fn running_graphics_processes(&self) -> Option<Vec<ProcessInfo>> {
        self.running_graphics_processes().ok()
    }

    fn process_utils(&self) -> Option<Vec<ProcessUtilizationSample>> {
        self.process_utilization_stats(None).ok()
    }
}

/// Assemble one GPU's snapshot, degrading every failed read to its neutral
/// value — never panics (pinned by the fake-source tests below).
fn snapshot_for(
    src: &impl DeviceSource,
    index: usize,
    vram_fn: &dyn Fn(&[u32]) -> std::collections::HashMap<u32, u64>,
) -> GpuSnapshot {
    let name = src.name().unwrap_or_else(|| "GPU".to_string());
    let temperature = src.temperature().unwrap_or(0);
    let utilization = src.utilization().unwrap_or(0);
    let (memory_used, memory_total) = src.memory().unwrap_or((0, 0));
    let power_usage = src.power_usage().unwrap_or(0) / 1000;
    let power_limit = src.power_limit().unwrap_or(0) / 1000;
    let clock_freq = src.clock().unwrap_or(0);
    let compute = src.running_compute_processes().unwrap_or_default();
    let graphics = src.running_graphics_processes().unwrap_or_default();
    let pids = union_pids(&compute, &graphics);
    let utils: std::collections::HashMap<u32, u32> = src
        .process_utils()
        .unwrap_or_default()
        .into_iter()
        .map(|s| (s.pid, s.sm_util))
        .collect();
    let processes = enrich_with(&vram_fn(&pids), &utils, compute, graphics);
    let compute_bytes = processes
        .iter()
        .filter(|p| p.kind == ProcKind::Compute)
        .map(|p| p.used_gpu_memory)
        .sum();
    let graphics_bytes = processes
        .iter()
        .filter(|p| p.kind == ProcKind::Graphics)
        .map(|p| p.used_gpu_memory)
        .sum();
    GpuSnapshot {
        index,
        name,
        temperature,
        utilization,
        memory_used,
        memory_total,
        power_usage,
        power_limit,
        clock_freq,
        processes,
        compute_bytes,
        graphics_bytes,
    }
}

fn poll_loop(nvml: Nvml, watch_ms: u64, tx: std::sync::mpsc::Sender<Vec<GpuSnapshot>>) {
    loop {
        let mut snapshots = Vec::new();
        if let Ok(count) = nvml.device_count() {
            for index in 0..count {
                let Ok(device) = nvml.device_by_index(index) else {
                    continue;
                };
                snapshots.push(snapshot_for(
                    &device,
                    index as usize,
                    &vram::process_vram_bytes,
                ));
            }
        }
        // Receiver gone means the window closed; end the thread.
        if tx.send(snapshots).is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(watch_ms));
    }
}

pub(crate) struct GpuMonitor {
    rx: Receiver<Vec<GpuSnapshot>>,
    pub(crate) snapshots: Vec<GpuSnapshot>,
    /// Per-GPU-index ring of memory samples for the stage-3 graph.
    pub(crate) history: Vec<std::collections::VecDeque<MemSample>>,
    /// Selection keyed by (gpu index, pid) so it follows the process across
    /// re-sorts and polls, instead of pointing at a display row another
    /// process can occupy between polls.
    pub(crate) selected: Option<(usize, u32)>,
    /// Current column sort of the process list (clickable headers).
    pub(crate) sort_key: SortKey,
    pub(crate) sort_asc: bool,
    /// Last kill outcome for the footer: (text, ok).
    pub(crate) last_message: Option<(String, bool)>,
}

impl GpuMonitor {
    fn start_drain(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(100))
                    .await;
                // Update failing means the window is gone; stop the loop.
                let alive = this
                    .update(cx, |view, cx| {
                        let mut updated = false;
                        // Drain to the newest snapshot; skip stale ones.
                        while let Ok(snapshots) = view.rx.try_recv() {
                            push_samples(&mut view.history, &snapshots);
                            view.snapshots = snapshots;
                            updated = true;
                        }
                        if updated {
                            view.revalidate_selection();
                            cx.notify();
                        }
                    })
                    .is_ok();
                if !alive {
                    break;
                }
            }
        })
        .detach();
    }

    /// Flattened process rows across GPUs in the current column sort — the
    /// display order for the list, selection and (stage 4) kill targeting.
    pub(crate) fn process_rows(&self) -> Vec<(usize, &GpuProc)> {
        let mut rows: Vec<(usize, &GpuProc)> = self
            .snapshots
            .iter()
            .flat_map(|gpu| gpu.processes.iter().map(move |p| (gpu.index, p)))
            .collect();
        let (key, asc) = (self.sort_key, self.sort_asc);
        rows.sort_by(|a, b| {
            let ord = match key {
                SortKey::Pid => a.1.pid.cmp(&b.1.pid),
                SortKey::Memory => a.1.used_gpu_memory.cmp(&b.1.used_gpu_memory),
                SortKey::Kind => a.1.kind.cmp(&b.1.kind),
                SortKey::Name => a.1.name.to_lowercase().cmp(&b.1.name.to_lowercase()),
                SortKey::Util => a.1.sm_util.unwrap_or(0).cmp(&b.1.sm_util.unwrap_or(0)),
            };
            if asc { ord } else { ord.reverse() }
        });
        rows
    }

    /// Header click: same column flips direction, a new column starts in its
    /// natural direction (memory and utilization read best largest-first).
    pub(crate) fn toggle_sort(&mut self, key: SortKey) {
        if self.sort_key == key {
            self.sort_asc = !self.sort_asc;
        } else {
            self.sort_key = key;
            self.sort_asc = !matches!(key, SortKey::Memory | SortKey::Util);
        }
    }

    /// Display index of the selected process, if it is still listed.
    pub(crate) fn selected_row_index(&self) -> Option<usize> {
        let sel = self.selected?;
        self.process_rows()
            .iter()
            .position(|(gpu, p)| *gpu == sel.0 && p.pid == sel.1)
    }

    /// Drop a selection whose process is gone (died, or left the GPU).
    fn revalidate_selection(&mut self) {
        if self.selected.is_some() && self.selected_row_index().is_none() {
            self.selected = None;
        }
    }

    /// Move the selection by `step` rows, clamped to the list bounds. From no
    /// selection, down enters at the first row and up at the last.
    fn step_selection(&mut self, step: isize) {
        let rows = self.process_rows();
        if rows.is_empty() {
            self.selected = None;
            return;
        }
        let current = match self.selected_row_index() {
            Some(i) => i as isize,
            None if step < 0 => rows.len() as isize,
            None => -1,
        };
        let next = (current + step).clamp(0, rows.len() as isize - 1) as usize;
        self.selected = Some((rows[next].0, rows[next].1.pid));
    }

    pub(crate) fn select_up(&mut self, _: &SelectUp, _: &mut gpui::Window, cx: &mut Context<Self>) {
        self.step_selection(-1);
        cx.notify();
    }

    pub(crate) fn select_down(
        &mut self,
        _: &SelectDown,
        _: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        self.step_selection(1);
        cx.notify();
    }

    /// The process the kill action targets: the selected row in display order.
    fn kill_target(&self) -> Option<(u32, String)> {
        let index = self.selected_row_index()?;
        self.process_rows()
            .get(index)
            .map(|(_, p)| (p.pid, p.name.clone()))
    }

    pub(crate) fn kill_selected(
        &mut self,
        _: &KillSelected,
        _: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        let Some((pid, name)) = self.kill_target() else {
            return;
        };
        self.last_message = Some(kill_message(
            crate::utils::system::kill_selected_process(pid, &name),
            pid,
            &name,
        ));
        cx.notify();
    }
}

/// Footer text for a kill attempt: (message, ok).
fn kill_message<E: std::fmt::Display>(
    result: std::result::Result<(), E>,
    pid: u32,
    name: &str,
) -> (String, bool) {
    match result {
        Ok(()) => (format!("killed {pid} ({name})"), true),
        Err(e) => (format!("kill {pid} failed: {e}"), false),
    }
}

fn format_memory(bytes: u64) -> String {
    let mib = bytes / 1_048_576;
    if mib >= 1024 {
        format!("{:.1}GiB", mib as f64 / 1024.0)
    } else {
        format!("{mib}MiB")
    }
}

pub(crate) fn run(nvml: Nvml, watch_ms: u64) -> Result<()> {
    let (tx, rx) = channel::<Vec<GpuSnapshot>>();
    std::thread::Builder::new()
        .name("nvml-poller".into())
        .spawn(move || poll_loop(nvml, watch_ms, tx))?;

    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1100.), px(780.)), cx);
        cx.bind_keys([
            KeyBinding::new("q", Quit, Some("GpuMonitor")),
            KeyBinding::new("up", SelectUp, Some("GpuMonitor")),
            KeyBinding::new("down", SelectDown, Some("GpuMonitor")),
            KeyBinding::new("x", KillSelected, Some("GpuMonitor")),
        ]);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some("nviwatch".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|cx| {
                    let mut view = GpuMonitor {
                        rx,
                        snapshots: Vec::new(),
                        history: Vec::new(),
                        selected: None,
                        sort_key: SortKey::Memory,
                        sort_asc: false,
                        last_message: None,
                    };
                    view.start_drain(cx);
                    view
                })
            },
        )
        .unwrap();
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::ProcKind;

    #[test]
    fn test_format_memory_mib_below_gib() {
        assert_eq!(format_memory(0), "0MiB");
        assert_eq!(format_memory(512 * 1_048_576), "512MiB");
    }

    #[test]
    fn test_format_memory_gib_boundary() {
        // 1 GiB exactly: switches to GiB display.
        assert_eq!(format_memory(1_048_576 * 1024), "1.0GiB");
        assert_eq!(format_memory(1_048_576 * 1024 - 1), "1023MiB");
        assert_eq!(format_memory(24 * 1_048_576 * 1024), "24.0GiB");
    }

    #[test]
    fn test_process_rows_sorted_by_memory_desc() {
        use data::GpuProc;
        let mk = |pid: u32, mem: u64, kind: ProcKind| GpuProc {
            pid,
            used_gpu_memory: mem,
            kind,
            name: "x".into(),
            sm_util: None,
        };
        let mut view = GpuMonitor {
            rx: channel::<Vec<GpuSnapshot>>().1,
            snapshots: vec![
                GpuSnapshot {
                    index: 0,
                    name: "G".into(),
                    temperature: 0,
                    utilization: 0,
                    memory_used: 0,
                    memory_total: 0,
                    power_usage: 0,
                    power_limit: 0,
                    clock_freq: 0,
                    processes: vec![
                        mk(1, 100, ProcKind::Compute),
                        mk(2, 900, ProcKind::Graphics),
                    ],
                    compute_bytes: 100,
                    graphics_bytes: 900,
                },
                GpuSnapshot {
                    index: 1,
                    name: "G2".into(),
                    temperature: 0,
                    utilization: 0,
                    memory_used: 0,
                    memory_total: 0,
                    power_usage: 0,
                    power_limit: 0,
                    clock_freq: 0,
                    processes: vec![mk(3, 500, ProcKind::Compute)],
                    compute_bytes: 500,
                    graphics_bytes: 0,
                },
            ],
            history: Vec::new(),
            selected: None,
            sort_key: SortKey::Memory,
            sort_asc: false,
            last_message: None,
        };
        let rows = view.process_rows();
        let pids: Vec<u32> = rows.iter().map(|(_, p)| p.pid).collect();
        assert_eq!(pids, vec![2, 3, 1]); // 900, 500, 100
        assert_eq!(rows[0].0, 0); // gpu index of pid 2
        // A selection whose process is still listed survives revalidation.
        view.selected = Some((0, 2));
        view.revalidate_selection();
        assert_eq!(view.selected, Some((0, 2)));
        // A selection whose process is gone is dropped.
        view.selected = Some((0, 99));
        view.revalidate_selection();
        assert_eq!(view.selected, None);
    }

    #[test]
    fn test_toggle_sort_and_name_order() {
        use data::GpuProc;
        let mk = |pid: u32, name: &str| GpuProc {
            pid,
            used_gpu_memory: 0,
            kind: ProcKind::Graphics,
            name: name.into(),
            sm_util: None,
        };
        let mut view = GpuMonitor {
            rx: channel::<Vec<GpuSnapshot>>().1,
            snapshots: vec![GpuSnapshot {
                index: 0,
                name: "G".into(),
                temperature: 0,
                utilization: 0,
                memory_used: 0,
                memory_total: 0,
                power_usage: 0,
                power_limit: 0,
                clock_freq: 0,
                processes: vec![mk(1, "zeta"), mk(2, "Alpha"), mk(3, "mid")],
                compute_bytes: 0,
                graphics_bytes: 0,
            }],
            history: Vec::new(),
            selected: None,
            sort_key: SortKey::Memory,
            sort_asc: false,
            last_message: None,
        };
        // New column starts ascending (memory is the descending exception).
        view.toggle_sort(SortKey::Name);
        assert!(view.sort_asc);
        let names: Vec<&str> = view
            .process_rows()
            .iter()
            .map(|(_, p)| p.name.as_str())
            .collect();
        assert_eq!(names, vec!["Alpha", "mid", "zeta"]);
        // Same column flips to descending.
        view.toggle_sort(SortKey::Name);
        assert!(!view.sort_asc);
        let names: Vec<&str> = view
            .process_rows()
            .iter()
            .map(|(_, p)| p.name.as_str())
            .collect();
        assert_eq!(names, vec!["zeta", "mid", "Alpha"]);
        // Switching to memory starts descending.
        view.toggle_sort(SortKey::Memory);
        assert!(!view.sort_asc);
        // Utilization also reads best largest-first.
        view.toggle_sort(SortKey::Util);
        assert!(!view.sort_asc);
    }

    // --- stage 4: enrich seam, selection, kill target ---

    fn proc_info(pid: u32, bytes: u64) -> nvml::struct_wrappers::device::ProcessInfo {
        nvml::struct_wrappers::device::ProcessInfo {
            pid,
            used_gpu_memory: nvml::enums::device::UsedGpuMemory::Used(bytes),
            gpu_instance_id: None,
            compute_instance_id: None,
        }
    }

    fn view_with_pids(pids: &[u32]) -> GpuMonitor {
        let processes: Vec<GpuProc> = pids
            .iter()
            .map(|&pid| GpuProc {
                pid,
                used_gpu_memory: 0,
                kind: ProcKind::Graphics,
                name: format!("p{pid}"),
                sm_util: None,
            })
            .collect();
        GpuMonitor {
            rx: channel::<Vec<GpuSnapshot>>().1,
            snapshots: vec![GpuSnapshot {
                index: 0,
                name: "G".into(),
                temperature: 0,
                utilization: 0,
                memory_used: 0,
                memory_total: 0,
                power_usage: 0,
                power_limit: 0,
                clock_freq: 0,
                processes,
                compute_bytes: 0,
                graphics_bytes: 0,
            }],
            history: Vec::new(),
            selected: None,
            sort_key: SortKey::Pid,
            sort_asc: true,
            last_message: None,
        }
    }

    /// The live test process resolves on every platform, so `enrich_with` is
    /// drivable without hardware.
    #[test]
    fn test_enrich_with_empty_vram_falls_back_to_nvml() {
        let pid = std::process::id();
        let rows = enrich_with(
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            vec![proc_info(pid, 1234)],
            Vec::new(),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].pid, pid);
        assert_eq!(rows[0].used_gpu_memory, 1234);
        assert_eq!(rows[0].sm_util, None);
    }

    #[test]
    fn test_enrich_with_vram_map_wins_over_nvml() {
        let pid = std::process::id();
        let vram = std::collections::HashMap::from([(pid, 9999u64)]);
        let utils = std::collections::HashMap::from([(pid, 77u32)]);
        let rows = enrich_with(&vram, &utils, vec![proc_info(pid, 1234)], Vec::new());
        assert_eq!(rows[0].used_gpu_memory, 9999);
        assert_eq!(rows[0].sm_util, Some(77));
    }

    #[test]
    fn test_step_selection_clamps() {
        let mut view = view_with_pids(&[1, 2, 3]);
        // From no selection, down enters at the first row.
        view.step_selection(1);
        assert_eq!(view.selected, Some((0, 1)));
        view.step_selection(1);
        assert_eq!(view.selected, Some((0, 2)));
        view.step_selection(10);
        assert_eq!(view.selected, Some((0, 3)));
        view.step_selection(-10);
        assert_eq!(view.selected, Some((0, 1)));
        // From no selection, up enters at the last row.
        view.selected = None;
        view.step_selection(-1);
        assert_eq!(view.selected, Some((0, 3)));
        // Empty list: nothing to select.
        let mut empty = view_with_pids(&[]);
        empty.step_selection(1);
        assert_eq!(empty.selected, None);
    }

    #[test]
    fn test_kill_target_follows_selection() {
        let mut view = view_with_pids(&[11, 22, 33]);
        view.selected = Some((0, 11));
        assert_eq!(view.kill_target().unwrap().0, 11);
        view.selected = Some((0, 33));
        let (pid, name) = view.kill_target().unwrap();
        assert_eq!(pid, 33);
        assert_eq!(name, "p33");
        // Empty list or a dead selection: no target, kill is a no-op.
        assert!(view_with_pids(&[]).kill_target().is_none());
        view.selected = Some((0, 44));
        assert!(view.kill_target().is_none());
    }

    /// The final-review finding: selection must follow the process across
    /// re-sorts, not the display row it happened to occupy — otherwise `x`
    /// can kill whatever moved into that row between polls.
    #[test]
    fn test_selection_follows_process_across_resort() {
        let mut view = view_with_pids(&[11, 22, 33]);
        view.sort_key = SortKey::Memory;
        view.sort_asc = false;
        view.snapshots[0].processes[0].used_gpu_memory = 300;
        view.snapshots[0].processes[1].used_gpu_memory = 200;
        view.snapshots[0].processes[2].used_gpu_memory = 100;
        view.selected = Some((0, 33)); // last row under memory-desc
        assert_eq!(view.selected_row_index(), Some(2));
        // pid 33's memory grows past the others: it moves to row 0, and the
        // pid-keyed selection moves with it.
        view.snapshots[0].processes[2].used_gpu_memory = 999;
        assert_eq!(view.selected_row_index(), Some(0));
        assert_eq!(view.kill_target().unwrap().0, 33);
    }

    #[test]
    fn test_kill_message() {
        let (msg, ok) = kill_message(Ok::<(), std::io::Error>(()), 7, "ollama");
        assert!(ok);
        assert_eq!(msg, "killed 7 (ollama)");
        let (msg, ok) = kill_message(Err::<(), _>(std::io::Error::other("denied")), 7, "ollama");
        assert!(!ok);
        assert_eq!(msg, "kill 7 failed: denied");
    }

    // --- final review: snapshot degrade seam ---

    struct FakeSource {
        name: Option<String>,
        temperature: Option<u32>,
        utilization: Option<u32>,
        memory: Option<(u64, u64)>,
        power_usage: Option<u32>,
        power_limit: Option<u32>,
        clock: Option<u32>,
        compute: Option<Vec<ProcessInfo>>,
        graphics: Option<Vec<ProcessInfo>>,
        utils: Option<Vec<ProcessUtilizationSample>>,
    }

    impl DeviceSource for FakeSource {
        fn name(&self) -> Option<String> {
            self.name.clone()
        }

        fn temperature(&self) -> Option<u32> {
            self.temperature
        }

        fn utilization(&self) -> Option<u32> {
            self.utilization
        }

        fn memory(&self) -> Option<(u64, u64)> {
            self.memory
        }

        fn power_usage(&self) -> Option<u32> {
            self.power_usage
        }

        fn power_limit(&self) -> Option<u32> {
            self.power_limit
        }

        fn clock(&self) -> Option<u32> {
            self.clock
        }

        fn running_compute_processes(&self) -> Option<Vec<ProcessInfo>> {
            self.compute.clone()
        }

        fn running_graphics_processes(&self) -> Option<Vec<ProcessInfo>> {
            self.graphics.clone()
        }

        fn process_utils(&self) -> Option<Vec<ProcessUtilizationSample>> {
            self.utils.clone()
        }
    }

    fn no_vram(_: &[u32]) -> std::collections::HashMap<u32, u64> {
        std::collections::HashMap::new()
    }

    fn util_sample(pid: u32, sm_util: u32) -> ProcessUtilizationSample {
        ProcessUtilizationSample {
            pid,
            timestamp: 0,
            sm_util,
            mem_util: 0,
            enc_util: 0,
            dec_util: 0,
        }
    }

    /// Every failed read degrades to its neutral value while the process
    /// list still assembles — pins the poll-loop degrade invariant without
    /// hardware.
    #[test]
    fn test_snapshot_for_degrades_per_field_on_errors() {
        let pid = std::process::id();
        let src = FakeSource {
            name: None,
            temperature: None,
            utilization: None,
            memory: None,
            power_usage: None,
            power_limit: None,
            clock: None,
            compute: Some(vec![proc_info(pid, 1234)]),
            graphics: None,
            utils: None,
        };
        let snap = snapshot_for(&src, 0, &no_vram);
        assert_eq!(snap.name, "GPU");
        assert_eq!(snap.temperature, 0);
        assert_eq!(snap.utilization, 0);
        assert_eq!((snap.memory_used, snap.memory_total), (0, 0));
        assert_eq!((snap.power_usage, snap.power_limit), (0, 0));
        assert_eq!(snap.clock_freq, 0);
        assert_eq!(snap.processes.len(), 1);
        assert_eq!(snap.processes[0].used_gpu_memory, 1234);
    }

    #[test]
    fn test_snapshot_for_reads_flow_through() {
        let src = FakeSource {
            name: Some("RTX".into()),
            temperature: Some(61),
            utilization: Some(42),
            memory: Some((2_000_000_000, 8_000_000_000)),
            power_usage: Some(120_000),
            power_limit: Some(250_000),
            clock: Some(1800),
            compute: None,
            graphics: None,
            utils: None,
        };
        let snap = snapshot_for(&src, 1, &no_vram);
        assert_eq!(snap.name, "RTX");
        assert_eq!(snap.temperature, 61);
        assert_eq!(snap.utilization, 42);
        assert_eq!(snap.memory_used, 2_000_000_000);
        assert_eq!(snap.memory_total, 8_000_000_000);
        assert_eq!(snap.power_usage, 120); // mW → W
        assert_eq!(snap.power_limit, 250);
        assert_eq!(snap.clock_freq, 1800);
        assert_eq!(snap.index, 1);
    }

    /// NVML's per-pid utilization samples land on the matching row; pids
    /// without a sample degrade to `None`.
    #[test]
    fn test_snapshot_for_maps_util_to_pids() {
        let pid = std::process::id();
        let src = FakeSource {
            name: None,
            temperature: None,
            utilization: None,
            memory: None,
            power_usage: None,
            power_limit: None,
            clock: None,
            compute: Some(vec![proc_info(pid, 10)]),
            graphics: None,
            utils: Some(vec![util_sample(pid, 77)]),
        };
        let snap = snapshot_for(&src, 0, &no_vram);
        assert_eq!(snap.processes.len(), 1);
        assert_eq!(snap.processes[0].sm_util, Some(77));
        // A process absent from the samples has no utilization figure.
        let empty = FakeSource {
            utils: Some(vec![util_sample(999_999, 5)]),
            ..src
        };
        let snap = snapshot_for(&empty, 0, &no_vram);
        assert_eq!(snap.processes[0].sm_util, None);
    }
}
