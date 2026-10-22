//! Snapshot data types and pure classification logic for the gpui app.
//!
//! Everything hardware-free lives here so the NVML per-field degrade
//! invariant and the compute/graphics split are testable without a GPU.

use std::collections::VecDeque;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum ProcKind {
    /// CUDA/compute — ollama and other LLM runtimes.
    Compute,
    /// Graphics — video playback, games, browsers.
    Graphics,
}

/// One GPU process row after classification; name resolved per platform.
#[derive(Clone, Debug)]
pub(crate) struct GpuProc {
    pub pid: u32,
    pub used_gpu_memory: u64,
    pub kind: ProcKind,
    pub name: String,
    /// NVML SM (3D/compute) utilization for this pid, when reported.
    pub sm_util: Option<u32>,
}

/// Process-name fragments that indicate an LLM runtime. Separators are
/// stripped from the name before matching, so "LM Studio" hits `lmstudio`.
const LLM_NAME_HINTS: &[&str] = &[
    "ollama", "llama", "vllm", "lmstudio", "kobold", "exllama", "gpt4all", "tabby", "comfyui",
    "sglang", "msty",
];

/// Name-based classification. NVML's compute/graphics lists cannot decide
/// this on WDDM — both contain every GPU process — so the display name does.
pub(crate) fn classify_by_name(name: &str) -> ProcKind {
    let normalized: String = name
        .to_lowercase()
        .chars()
        .filter(|c| !matches!(c, ' ' | '-' | '_' | '.'))
        .collect();
    if LLM_NAME_HINTS.iter().any(|hint| normalized.contains(hint)) {
        ProcKind::Compute
    } else {
        ProcKind::Graphics
    }
}

/// Column the process list is sorted by (clickable headers).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SortKey {
    Pid,
    Memory,
    Kind,
    Name,
    Util,
}

/// One history point for the memory graph (stage 3 renders it).
// Fields are written by push_samples now and read by the stage-3 graph.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MemSample {
    pub mem_total: u64,
    pub compute: u64,
    pub graphics: u64,
    /// VRAM the per-process lists do not account for (driver, desktop).
    pub other: u64,
    pub util: u32,
    pub power: u32,
}

/// ~72s of samples at the default 300ms poll.
pub(crate) const HISTORY_CAP: usize = 240;

/// Append one sample per GPU index, dropping the oldest beyond `HISTORY_CAP`.
pub(crate) fn push_samples(
    history: &mut Vec<VecDeque<MemSample>>,
    snapshots: &[super::GpuSnapshot],
) {
    for gpu in snapshots {
        let other = gpu
            .memory_used
            .saturating_sub(gpu.compute_bytes + gpu.graphics_bytes);
        let sample = MemSample {
            mem_total: gpu.memory_total,
            compute: gpu.compute_bytes,
            graphics: gpu.graphics_bytes,
            other,
            util: gpu.utilization,
            power: gpu.power_usage,
        };
        if history.len() <= gpu.index {
            history.resize(gpu.index + 1, VecDeque::new());
        }
        let ring = &mut history[gpu.index];
        ring.push_back(sample);
        while ring.len() > HISTORY_CAP {
            ring.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_by_name_llm_hints() {
        assert_eq!(classify_by_name("ollama"), ProcKind::Compute);
        assert_eq!(classify_by_name("Ollama.exe"), ProcKind::Compute);
        assert_eq!(classify_by_name("llama-server"), ProcKind::Compute);
        assert_eq!(classify_by_name("LM Studio.exe"), ProcKind::Compute);
        assert_eq!(classify_by_name("vllm serve"), ProcKind::Compute);
    }

    #[test]
    fn test_classify_by_name_defaults_graphics() {
        assert_eq!(classify_by_name("chrome.exe"), ProcKind::Graphics);
        assert_eq!(classify_by_name("explorer.EXE"), ProcKind::Graphics);
        assert_eq!(classify_by_name(""), ProcKind::Graphics);
        // Substring must not fire inside unrelated words.
        assert_eq!(classify_by_name("collimator"), ProcKind::Graphics);
    }

    #[test]
    fn test_push_samples_grows_and_caps() {
        use crate::gui::GpuSnapshot;
        let mut history: Vec<VecDeque<MemSample>> = Vec::new();
        let snap = GpuSnapshot {
            index: 0,
            name: "G".into(),
            temperature: 0,
            utilization: 10,
            memory_used: 900,
            memory_total: 1000,
            power_usage: 5,
            power_limit: 10,
            clock_freq: 0,
            processes: Vec::new(),
            compute_bytes: 300,
            graphics_bytes: 400,
        };
        for _ in 0..(HISTORY_CAP + 5) {
            push_samples(&mut history, std::slice::from_ref(&snap));
        }
        assert_eq!(history[0].len(), HISTORY_CAP);
        let last = history[0].back().unwrap();
        assert_eq!(last.compute, 300);
        assert_eq!(last.graphics, 400);
        assert_eq!(last.other, 200); // 900 - 300 - 400
        assert_eq!(last.mem_total, 1000);
    }
}
