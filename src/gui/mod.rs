//! gpui window app: NVIDIA GPU monitoring without the terminal.
//!
//! A background thread owns `Nvml` and pushes snapshots over a channel; the
//! window drains it on a timer loop. NVML stays off the UI executor so a slow
//! driver call can never stall a frame.

use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

use gpui::{
    KeyBinding, actions, div, prelude::*, px, rgb, size, App, Application, Bounds, Context,
    TitlebarOptions, Window, WindowBounds, WindowOptions,
};
use nvml::Nvml;

use crate::Result;

actions!(nviwatch, [Quit]);

/// One poll of one GPU. Process detail arrives in the data-layer stage.
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

struct GpuMonitor {
    rx: Receiver<Vec<GpuSnapshot>>,
    snapshots: Vec<GpuSnapshot>,
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
                            view.snapshots = snapshots;
                            updated = true;
                        }
                        if updated {
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
}

impl Render for GpuMonitor {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let bg = rgb(0x0e1116);
        let card_border = rgb(0x2a2f3a);
        let text = rgb(0xd8dee9);
        let dim = rgb(0x8b93a1);

        let mut root = div()
            .key_context("GpuMonitor")
            .on_action(cx.listener(|_, _: &Quit, _, cx| cx.quit()))
            .flex()
            .flex_col()
            .size_full()
            .bg(bg)
            .text_color(text)
            .font_family("Segoe UI")
            .p_4()
            .gap_2();

        if self.snapshots.is_empty() {
            root = root.child(
                div()
                    .flex()
                    .flex_1()
                    .items_center()
                    .justify_center()
                    .text_color(dim)
                    .child("Waiting for GPU data…"),
            );
            return root;
        }

        for gpu in &self.snapshots {
            let header = format!(
                "GPU {} — {}   {}°C   {}%   {}/{}   {}/{}W   {}MHz",
                gpu.index,
                gpu.name,
                gpu.temperature,
                gpu.utilization,
                format_memory(gpu.memory_used),
                format_memory(gpu.memory_total),
                gpu.power_usage,
                gpu.power_limit,
                gpu.clock_freq,
            );
            root = root.child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .border_1()
                    .border_color(card_border)
                    .rounded_md()
                    .p_3()
                    .child(div().text_lg().child(header))
                    .child(
                        div()
                            .text_color(dim)
                            .child("graphs and process list land in the next stages"),
                    ),
            );
        }
        root
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

#[cfg(test)]
mod tests {
    use super::*;

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
