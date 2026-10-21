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
    KeyBinding, actions, prelude::*, px, size, App, Application, Bounds, Context, TitlebarOptions,
    WindowBounds, WindowOptions,
};
use nvml::Nvml;

use crate::Result;
use crate::utils::system::get_process_identity;
use data::{GpuProc, MemSample, ProcKind, SortKey, push_samples};

actions!(nviwatch, [Quit]);

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

/// Merge the NVML process lists into typed rows: name/user resolved per
/// platform, kind decided by name, VRAM from D3DKMT with the NVML figure as
/// fallback (D3DKMT is Windows-only; NVML reports "NA" there on WDDM). Dead
/// processes drop silently.
fn enrich(
    compute: Vec<nvml::struct_wrappers::device::ProcessInfo>,
    graphics: Vec<nvml::struct_wrappers::device::ProcessInfo>,
) -> Vec<GpuProc> {
    let mut pids: Vec<u32> = Vec::new();
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
        if !pids.contains(&info.pid) {
            pids.push(info.pid);
        }
    }
    let vram = vram::process_vram_bytes(&pids);
    pids.into_iter()
        .filter_map(|pid| {
            let (name, user) = get_process_identity(pid)?;
            Some(GpuProc {
                pid,
                used_gpu_memory: vram
                    .get(&pid)
                    .copied()
                    .or_else(|| nvml_mem.get(&pid).copied())
                    .unwrap_or(0),
                kind: data::classify_by_name(&name),
                name,
                user,
            })
        })
        .collect()
}

fn poll_loop(nvml: Nvml, watch_ms: u64, tx: std::sync::mpsc::Sender<Vec<GpuSnapshot>>) {
    loop {
        let mut snapshots = Vec::new();
        if let Ok(count) = nvml.device_count() {
            for index in 0..count {
                let Ok(device) = nvml.device_by_index(index) else {
                    continue;
                };
                let name = device.name().unwrap_or_else(|_| "GPU".to_string());
                let temperature = device
                    .temperature(nvml::enum_wrappers::device::TemperatureSensor::Gpu)
                    .unwrap_or(0);
                let utilization = device.utilization_rates().map(|u| u.gpu).unwrap_or(0);
                let (memory_used, memory_total) = device
                    .memory_info()
                    .map(|m| (m.used, m.total))
                    .unwrap_or((0, 0));
                let power_usage = device.power_usage().unwrap_or(0) / 1000;
                let power_limit = device.enforced_power_limit().unwrap_or(0) / 1000;
                let clock_freq = device
                    .clock_info(nvml::enum_wrappers::device::Clock::Graphics)
                    .unwrap_or(0);
                let compute = device.running_compute_processes().unwrap_or_default();
                let graphics = device.running_graphics_processes().unwrap_or_default();
                let processes = enrich(compute, graphics);
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
                snapshots.push(GpuSnapshot {
                    index: index as usize,
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
                });
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
    /// Row index into the flattened, sorted process list.
    pub(crate) selected: usize,
    /// Current column sort of the process list (clickable headers).
    pub(crate) sort_key: SortKey,
    pub(crate) sort_asc: bool,
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
                            view.clamp_selection();
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
                SortKey::User => a.1.user.cmp(&b.1.user),
            };
            if asc { ord } else { ord.reverse() }
        });
        rows
    }

    /// Header click: same column flips direction, a new column starts in its
    /// natural direction (memory reads best largest-first).
    pub(crate) fn toggle_sort(&mut self, key: SortKey) {
        if self.sort_key == key {
            self.sort_asc = !self.sort_asc;
        } else {
            self.sort_key = key;
            self.sort_asc = !matches!(key, SortKey::Memory);
        }
    }

    fn clamp_selection(&mut self) {
        let total = self.process_rows().len();
        if total == 0 {
            self.selected = 0;
        } else if self.selected >= total {
            self.selected = total - 1;
        }
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
        cx.bind_keys([KeyBinding::new("q", Quit, Some("GpuMonitor"))]);
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
                        selected: 0,
                        sort_key: SortKey::Memory,
                        sort_asc: false,
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
            user: "u".into(),
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
                    processes: vec![mk(1, 100, ProcKind::Compute), mk(2, 900, ProcKind::Graphics)],
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
            selected: 0,
            sort_key: SortKey::Memory,
            sort_asc: false,
        };
        let rows = view.process_rows();
        let pids: Vec<u32> = rows.iter().map(|(_, p)| p.pid).collect();
        assert_eq!(pids, vec![2, 3, 1]); // 900, 500, 100
        assert_eq!(rows[0].0, 0); // gpu index of pid 2
        view.clamp_selection();
        assert_eq!(view.selected, 0);
        view.selected = 99;
        view.clamp_selection();
        assert_eq!(view.selected, 2);
    }

    #[test]
    fn test_toggle_sort_and_name_order() {
        use data::GpuProc;
        let mk = |pid: u32, name: &str| GpuProc {
            pid,
            used_gpu_memory: 0,
            kind: ProcKind::Graphics,
            name: name.into(),
            user: "u".into(),
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
            selected: 0,
            sort_key: SortKey::Memory,
            sort_asc: false,
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
    }
}
