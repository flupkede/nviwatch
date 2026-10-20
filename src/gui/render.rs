//! Rendering for the gpui GPU monitor window.

use gpui::{Context, Render, Rgba, Window, div, prelude::*, px, rgb};

use super::GpuMonitor;
use super::data::ProcKind;
use super::format_memory;

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
            root = root.child(render_gpu_card(gpu));
        }

        let rows = self.process_rows();
        root = root.child(render_process_list(&rows, self.selected));

        root = root.child(
            div()
                .flex()
                .justify_between()
                .text_color(rgb(DIM))
                .child("up/down select · x kill · q quit")
                .child(format!("{} processes", rows.len())),
        );
        root
    }
}

fn render_gpu_card(gpu: &super::GpuSnapshot) -> impl IntoElement {
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
                    gpu.temperature, gpu.utilization, gpu.power_usage, gpu.power_limit, gpu.clock_freq,
                ))),
        )
        .child(
            div()
                .text_color(rgb(TEXT))
                .child(format!(
                    "VRAM {}/{}   LLM {}   video {}",
                    format_memory(gpu.memory_used),
                    format_memory(gpu.memory_total),
                    format_memory(gpu.compute_bytes),
                    format_memory(gpu.graphics_bytes),
                )),
        )
}

fn render_process_list(rows: &[(usize, &super::GpuProc)], selected: usize) -> impl IntoElement {
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
            .child(div().w(px(70.)).child("PID"))
            .child(div().w(px(80.)).child("GPU-Mem"))
            .child(div().w(px(64.)).child("Type"))
            .child(div().w(px(180.)).child("Name"))
            .child(div().child("User")),
    );

    for (index, (gpu_index, process)) in rows.iter().enumerate() {
        let mem = if process.used_gpu_memory == 0 {
            "—".to_string()
        } else {
            format_memory(process.used_gpu_memory)
        };
        let row = div()
            .flex()
            .gap_3()
            .py_0p5()
            .when(index == selected, |el| el.bg(rgb(SELECT_BG)))
            .child(div().w(px(70.)).child(process.pid.to_string()))
            .child(div().w(px(80.)).child(mem))
            .child(
                div()
                    .w(px(64.))
                    .text_color(kind_color(process.kind))
                    .child(kind_label(process.kind)),
            )
            .child(div().w(px(180.)).text_ellipsis().child(process.name.clone()))
            .child(
                div()
                    .text_color(rgb(DIM))
                    .child(format!("gpu{gpu_index} · {}", process.user)),
            );
        list = list.child(row);
    }
    list
}
