//! Rendering for the gpui GPU monitor window.

use gpui::{
    Context, Div, PathBuilder, Pixels, Render, Rgba, SharedString, Stateful, Window, canvas, div,
    point, prelude::*, px, rgb,
};

use super::GpuMonitor;
use super::data::{MemSample, ProcKind, SortKey};
use super::format_memory;
use super::graph::{area_geometry, grid_label, grid_labels, grid_ticks, stacked_geometry};

const BG: u32 = 0x0e1116;
const CARD: u32 = 0x161b22;
const BORDER: u32 = 0x2a2f3a;
const TEXT: u32 = 0xd8dee9;
const DIM: u32 = 0x8b93a1;
/// Compute/LLM segment and tag color.
const LLM: u32 = 0xff9f43;
/// Graphics/video segment and tag color.
const VIDEO: u32 = 0x4fc3f7;
const SELECT_BG: u32 = 0x2d3b4e;
/// Unaccounted VRAM (driver, desktop) in the memory graph.
const OTHER: u32 = 0x3a4150;
const UTIL: u32 = 0xc792ea;
const POWER: u32 = 0xe5c07b;
/// Footer kill feedback: success / failure.
const OK: u32 = 0x98c379;
const ERR: u32 = 0xe06c75;

fn kind_label(kind: ProcKind) -> &'static str {
    match kind {
        ProcKind::Compute => "LLM",
        ProcKind::Graphics => "VIDEO",
    }
}

fn kind_color(kind: ProcKind) -> Rgba {
    match kind {
        ProcKind::Compute => rgb(LLM),
        ProcKind::Graphics => rgb(VIDEO),
    }
}

impl Render for GpuMonitor {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut root = div()
            .key_context("GpuMonitor")
            .on_action(cx.listener(|_, _: &super::Quit, _, cx| cx.quit()))
            .on_action(cx.listener(GpuMonitor::select_up))
            .on_action(cx.listener(GpuMonitor::select_down))
            .on_action(cx.listener(GpuMonitor::kill_selected))
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(BG))
            .text_color(rgb(TEXT))
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
                    .text_color(rgb(DIM))
                    .child("Waiting for GPU data…"),
            );
            return root;
        }

        for gpu in &self.snapshots {
            let samples: Vec<MemSample> = self
                .history
                .get(gpu.index)
                .map(|ring| ring.iter().copied().collect())
                .unwrap_or_default();
            root = root.child(render_gpu_card(gpu, &samples));
        }

        let rows = self.process_rows();
        root = root.child(render_process_list(
            cx,
            &rows,
            self.selected_row_index(),
            self.sort_key,
            self.sort_asc,
        ));

        let mut footer = div().flex().justify_between().text_color(rgb(DIM));
        match &self.last_message {
            Some((message, ok)) => {
                footer = footer
                    .child(
                        div()
                            .text_color(rgb(if *ok { OK } else { ERR }))
                            .child(message.clone()),
                    )
                    .child(format!("{} processes", rows.len()));
            }
            None => {
                footer = footer
                    .child(div())
                    .child(format!("{} processes", rows.len()));
            }
        }
        root = root.child(footer);
        root
    }
}

fn render_gpu_card(gpu: &super::GpuSnapshot, samples: &[MemSample]) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .bg(rgb(CARD))
        .border_1()
        .border_color(rgb(BORDER))
        .rounded_md()
        .p_3()
        .child(
            div()
                .flex()
                .flex_wrap()
                .items_baseline()
                .gap_3()
                .child(
                    div()
                        .text_lg()
                        .text_color(rgb(TEXT))
                        .child(format!("GPU {} — {}", gpu.index, gpu.name)),
                )
                .child(div().text_color(rgb(DIM)).child(format!(
                    "{}°C   {}%   {}/{}W   {}MHz",
                    gpu.temperature,
                    gpu.utilization,
                    gpu.power_usage,
                    gpu.power_limit,
                    gpu.clock_freq,
                )))
                .child(div().flex_1())
                .child(div().text_color(rgb(TEXT)).child(format!(
                    "{}/{}",
                    format_memory(gpu.memory_used),
                    format_memory(gpu.memory_total)
                ))),
        )
        .child(render_memory_graph(samples))
        .child(render_small_graphs(gpu, samples))
}

/// Stacked VRAM graph: video (bottom), LLM, other — with 500MiB gridlines,
/// GiB axis labels and a legend. The used/total figure lives in the card
/// header, not here.
fn render_memory_graph(samples: &[MemSample]) -> impl IntoElement {
    let samples = samples.to_vec();
    // Same y-scale rule as graph.rs stacked_geometry — keep the two in sync
    // or the axis label stops matching the drawn polygons.
    let y_max_label = samples
        .iter()
        .map(|s| s.mem_total.max(s.compute + s.graphics + s.other))
        .max()
        .unwrap_or(0);
    // Axis labels in a column to the RIGHT of the bar: the y_max figure
    // carries the unit at the very top, 0 sits at the bottom, bare numbers
    // pin to their gridlines in between.
    let mut tick_labels: Vec<_> = if y_max_label > 0 {
        grid_labels(y_max_label)
            .into_iter()
            .filter(|&v| v != y_max_label)
            .chain(std::iter::once(0))
            .map(|v| {
                let y = 150.0 - (v as f32 / y_max_label as f32) * 150.0;
                axis_label((y - 7.0).max(0.0), grid_label(v))
            })
            .collect()
    } else {
        Vec::new()
    };
    tick_labels.push(axis_label(0.0, format_memory(y_max_label)));
    div()
        .flex()
        .flex_col()
        .gap_0p5()
        .child(
            div()
                .flex()
                .gap_1()
                .child(
                    div().h(px(150.)).flex_1().relative().child(
                        canvas(
                            |_, _, _| {},
                            move |bounds, _, window, _| {
                                paint_stacked(bounds, &samples, window);
                            },
                        )
                        .size_full(),
                    ),
                )
                .child(
                    div()
                        .h(px(150.))
                        .w(px(48.))
                        .relative()
                        .children(tick_labels),
                ),
        )
        .child(
            div()
                .flex()
                .gap_3()
                .text_xs()
                .text_color(rgb(DIM))
                .child(legend_swatch(rgb(VIDEO), "video / processen"))
                .child(legend_swatch(rgb(LLM), "LLM (ollama)"))
                .child(legend_swatch(rgb(OTHER), "overig")),
        )
}

fn legend_swatch(color: Rgba, label: &'static str) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap_1()
        .child(div().size_2().rounded_sm().bg(color))
        .child(label)
}

fn axis_label(top: f32, text: impl IntoElement) -> Div {
    div()
        .absolute()
        .top(px(top))
        .text_xs()
        .text_color(rgb(DIM))
        .child(text)
}

fn paint_stacked(bounds: gpui::Bounds<gpui::Pixels>, samples: &[MemSample], window: &mut Window) {
    let width = f32::from(bounds.size.width);
    let height = f32::from(bounds.size.height);
    let Some(geo) = stacked_geometry(samples, width, height) else {
        return;
    };
    let origin = bounds.origin;
    let to_point = |(x, y): (f32, f32)| point(origin.x + px(x), origin.y + px(y));
    let mut paint_layer = |vertices: &[(f32, f32)], color: Rgba| {
        let points: Vec<_> = vertices.iter().copied().map(to_point).collect();
        let mut builder = PathBuilder::fill();
        builder.add_polygon(&points, true);
        if let Ok(path) = builder.build() {
            window.paint_path(path, color);
        }
    };
    paint_layer(&geo.graphics, rgb(VIDEO));
    paint_layer(&geo.compute, rgb(LLM));
    paint_layer(&geo.other, rgb(OTHER));
    // Horizontal gridlines every 500 MiB, 1px, over the bands.
    for v in grid_ticks(geo.y_max) {
        let y = height - (v as f32 / geo.y_max as f32) * height;
        paint_layer(
            &[(0.0, y), (width, y), (width, y + 1.0), (0.0, y + 1.0)],
            rgb(BORDER),
        );
    }
}

/// Utilization and power side by side under the memory graph. The power
/// scale is the enforced power limit — a stable ceiling, unlike a
/// history-derived max that jumps as soon as the card idles.
fn render_small_graphs(gpu: &super::GpuSnapshot, samples: &[MemSample]) -> impl IntoElement {
    let util: Vec<f32> = samples.iter().map(|s| s.util as f32).collect();
    let power: Vec<f32> = samples.iter().map(|s| s.power as f32).collect();
    let power_max = if gpu.power_limit > 0 {
        gpu.power_limit as f32
    } else {
        power.iter().copied().fold(0.0_f32, f32::max).max(1.0)
    };
    let power_label = if gpu.power_limit > 0 {
        format!("{}/{}W", gpu.power_usage, gpu.power_limit)
    } else {
        format!("{}W", gpu.power_usage)
    };
    div()
        .flex()
        .gap_2()
        .child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .gap_0p5()
                .child(
                    div()
                        .text_color(rgb(TEXT))
                        .child(format!("{}%", gpu.utilization)),
                )
                .child(
                    div().h(px(56.)).w_full().child(
                        canvas(
                            |_, _, _| {},
                            move |bounds, _, window, _| {
                                paint_area(bounds, &util, 100.0, rgb(UTIL), window);
                            },
                        )
                        .size_full(),
                    ),
                )
                .child(div().text_xs().text_color(rgb(DIM)).child("utilization")),
        )
        .child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .gap_0p5()
                .child(div().text_color(rgb(TEXT)).child(power_label))
                .child(
                    div().h(px(56.)).w_full().child(
                        canvas(
                            |_, _, _| {},
                            move |bounds, _, window, _| {
                                paint_area(bounds, &power, power_max, rgb(POWER), window);
                            },
                        )
                        .size_full(),
                    ),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(rgb(DIM))
                        .child(format!("power (scale {power_max:.0}W)")),
                ),
        )
}

fn paint_area(
    bounds: gpui::Bounds<gpui::Pixels>,
    values: &[f32],
    v_max: f32,
    color: Rgba,
    window: &mut Window,
) {
    let width = f32::from(bounds.size.width);
    let height = f32::from(bounds.size.height);
    let Some(vertices) = area_geometry(values, v_max, width, height) else {
        return;
    };
    let origin = bounds.origin;
    let points: Vec<_> = vertices
        .iter()
        .copied()
        .map(|(x, y)| point(origin.x + px(x), origin.y + px(y)))
        .collect();
    let mut builder = PathBuilder::fill();
    builder.add_polygon(&points, true);
    if let Ok(path) = builder.build() {
        window.paint_path(path, color);
    }
}

fn render_process_list(
    cx: &mut Context<GpuMonitor>,
    rows: &[(usize, &super::GpuProc)],
    selected: Option<usize>,
    sort_key: SortKey,
    sort_asc: bool,
) -> impl IntoElement {
    let mut list = div()
        .id("process-list")
        .flex()
        .flex_col()
        .flex_1()
        .min_h_0()
        .bg(rgb(CARD))
        .border_1()
        .border_color(rgb(BORDER))
        .rounded_md()
        .p_2()
        .overflow_y_scroll();

    if rows.is_empty() {
        list = list
            .flex_1()
            .items_center()
            .justify_center()
            .text_color(rgb(DIM))
            .child("No GPU processes");
        return list;
    }

    list = list.child(
        div()
            .flex()
            .gap_3()
            .text_color(rgb(DIM))
            .border_b_1()
            .border_color(rgb(BORDER))
            .pb_1()
            .child(sort_header(
                cx,
                "hdr-pid",
                "PID",
                SortKey::Pid,
                sort_key,
                sort_asc,
                Some(px(70.)),
            ))
            .child(sort_header(
                cx,
                "hdr-mem",
                "GPU-Mem",
                SortKey::Memory,
                sort_key,
                sort_asc,
                Some(px(80.)),
            ))
            .child(sort_header(
                cx,
                "hdr-kind",
                "Type",
                SortKey::Kind,
                sort_key,
                sort_asc,
                Some(px(64.)),
            ))
            .child(sort_header(
                cx,
                "hdr-name",
                "Name",
                SortKey::Name,
                sort_key,
                sort_asc,
                Some(px(180.)),
            ))
            .child(sort_header(
                cx,
                "hdr-util",
                "GPU%",
                SortKey::Util,
                sort_key,
                sort_asc,
                Some(px(56.)),
            )),
    );

    for (index, (gpu_index, process)) in rows.iter().enumerate() {
        let mem = if process.used_gpu_memory == 0 {
            "—".to_string()
        } else {
            format_memory(process.used_gpu_memory)
        };
        let util = match process.sm_util {
            Some(u) => format!("{u}%"),
            None => "—".to_string(),
        };
        // Selection is keyed by (gpu, pid), not row: the row a process
        // occupies can change between polls.
        let (row_gpu, row_pid) = (*gpu_index, process.pid);
        let row = div()
            .id(SharedString::from(format!("proc-row-{index}")))
            .on_click(cx.listener(move |view, _, _, cx| {
                view.selected = Some((row_gpu, row_pid));
                cx.notify();
            }))
            .cursor_pointer()
            .flex()
            .gap_3()
            .py_0p5()
            .when(selected == Some(index), |el| el.bg(rgb(SELECT_BG)))
            .child(div().w(px(70.)).child(process.pid.to_string()))
            .child(div().w(px(80.)).child(mem))
            .child(
                div()
                    .w(px(64.))
                    .text_color(kind_color(process.kind))
                    .child(kind_label(process.kind)),
            )
            .child(
                div()
                    .w(px(180.))
                    .text_ellipsis()
                    .child(process.name.clone()),
            )
            .child(div().w(px(56.)).child(util));
        list = list.child(row);
    }
    list
}

/// Clickable column header; the active column shows its sort direction.
fn sort_header(
    cx: &mut Context<GpuMonitor>,
    id: &'static str,
    label: &'static str,
    key: SortKey,
    active: SortKey,
    asc: bool,
    width: Option<Pixels>,
) -> Stateful<Div> {
    let arrow = if active == key {
        if asc { " ▲" } else { " ▼" }
    } else {
        ""
    };
    let mut cell = div()
        .id(id)
        .on_click(cx.listener(move |view, _, _, _| view.toggle_sort(key)))
        .cursor_pointer();
    if let Some(w) = width {
        cell = cell.w(w);
    }
    cell.child(format!("{label}{arrow}"))
}
