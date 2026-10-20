//! Snapshot data types and pure classification logic for the gpui app.
//!
//! Everything hardware-free lives here so the NVML per-field degrade
//! invariant and the compute/graphics split are testable without a GPU.

use std::collections::VecDeque;

use nvml::enums::device::UsedGpuMemory;
use nvml::struct_wrappers::device::ProcessInfo;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ProcKind {
    /// CUDA/compute — ollama and other LLM runtimes.
    Compute,
    /// Graphics — video playback, games, browsers.
    Graphics,
}

/// One GPU process row after classification; name/user resolved per platform.
#[derive(Clone, Debug)]
pub(crate) struct GpuProc {
    pub pid: u32,
    pub used_gpu_memory: u64,
    pub kind: ProcKind,
    pub name: String,
    pub user: String,
}

/// Pure NVML→rows classification. A pid in both lists keeps the larger memory
/// figure; Compute wins the tie (the LLM case this app exists for).
pub(crate) fn classify_processes(
    compute: &[ProcessInfo],
    graphics: &[ProcessInfo],
) -> Vec<(u32, u64, ProcKind)> {
    let mut rows: Vec<(u32, u64, ProcKind)> = Vec::new();
    let push = |pid: u32, bytes: u64, kind: ProcKind, rows: &mut Vec<(u32, u64, ProcKind)>| {
        if let Some(existing) = rows.iter_mut().find(|(p, _, _)| *p == pid) {
            if bytes > existing.1 {
                existing.1 = bytes;
                existing.2 = kind;
            }
        } else {
            rows.push((pid, bytes, kind));
        }
    };
    for p in compute {
        push(p.pid, used_bytes(&p.used_gpu_memory), ProcKind::Compute, &mut rows);
    }
    for p in graphics {
        push(p.pid, used_bytes(&p.used_gpu_memory), ProcKind::Graphics, &mut rows);
    }
    rows
}

fn used_bytes(memory: &UsedGpuMemory) -> u64 {
    match memory {
        UsedGpuMemory::Used(bytes) => *bytes,
        UsedGpuMemory::Unavailable => 0,
    }
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
pub(crate) fn push_samples(history: &mut Vec<VecDeque<MemSample>>, snapshots: &[super::GpuSnapshot]) {
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

    fn proc(pid: u32, bytes: u64) -> ProcessInfo {
        ProcessInfo {
            pid,
            used_gpu_memory: UsedGpuMemory::Used(bytes),
            compute_instance_id: None,
            gpu_instance_id: None,
        }
    }

    #[test]
    fn test_classify_separate_lists() {
        let rows = classify_processes(&[proc(1, 100)], &[proc(2, 200)]);
        assert_eq!(
            rows,
            vec![
                (1, 100, ProcKind::Compute),
                (2, 200, ProcKind::Graphics),
            ]
        );
    }

    #[test]
    fn test_classify_overlap_keeps_larger_compute_wins_tie() {
        // Same pid in both lists: larger figure wins; equal → Compute.
        let rows = classify_processes(&[proc(7, 300)], &[proc(7, 200)]);
        assert_eq!(rows, vec![(7, 300, ProcKind::Compute)]);
        let rows = classify_processes(&[proc(7, 100)], &[proc(7, 500)]);
        assert_eq!(rows, vec![(7, 500, ProcKind::Graphics)]);
        let rows = classify_processes(&[proc(7, 100)], &[proc(7, 100)]);
        assert_eq!(rows, vec![(7, 100, ProcKind::Compute)]);
    }

    #[test]
    fn test_classify_unavailable_memory_is_zero() {
        let unavailable = ProcessInfo {
            pid: 9,
            used_gpu_memory: UsedGpuMemory::Unavailable,
            compute_instance_id: None,
            gpu_instance_id: None,
        };
        let rows = classify_processes(&[unavailable], &[]);
        assert_eq!(rows, vec![(9, 0, ProcKind::Compute)]);
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
            push_samples(&mut history, &[snap.clone()]);
        }
        assert_eq!(history[0].len(), HISTORY_CAP);
        let last = history[0].back().unwrap();
        assert_eq!(last.compute, 300);
        assert_eq!(last.graphics, 400);
        assert_eq!(last.other, 200); // 900 - 300 - 400
        assert_eq!(last.mem_total, 1000);
    }
}
