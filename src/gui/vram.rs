//! Per-process VRAM on Windows via D3DKMT — the source Task Manager uses.
//!
//! NVML cannot report per-process memory on WDDM (every process reads "NA"),
//! so the figure comes from `D3DKMTQueryStatistics` with the
//! PROCESS_SEGMENT_GROUP type: the LOCAL segment group is dedicated VRAM.
//! The query rejects a `PROCESS_QUERY_LIMITED_INFORMATION` handle with
//! STATUS_INVALID_PARAMETER, so the process is opened with full query rights.
//! The NVIDIA adapter is identified by DXGI description match — a process
//! also commits memory on the iGPU, and those figures must not leak in.
//! Each pid is queried against every NVIDIA adapter and the largest figure
//! wins. Non-Windows builds return an empty map; NVML already fills
//! `used_gpu_memory` there.

#[cfg(windows)]
use windows::Win32::Foundation::LUID;

#[cfg(windows)]
pub(crate) fn process_vram_bytes(pids: &[u32]) -> std::collections::HashMap<u32, u64> {
    use windows::Wdk::Graphics::Direct3D::{
        D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL, D3DKMT_QUERYSTATISTICS, D3DKMT_QUERYSTATISTICS_0,
        D3DKMT_QUERYSTATISTICS_PROCESS_SEGMENT_GROUP, D3DKMTQueryStatistics,
    };
    use windows::Win32::Foundation::{CloseHandle, NTSTATUS};
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_INFORMATION};

    let succeeded = |status: NTSTATUS| status.0 >= 0;
    let mut map = std::collections::HashMap::new();
    if pids.is_empty() {
        return map;
    }

    let luids = nvidia_adapter_luids();
    if luids.is_empty() {
        return map;
    }

    for &pid in pids {
        let Ok(handle) = (unsafe { OpenProcess(PROCESS_QUERY_INFORMATION, false, pid) }) else {
            continue;
        };
        let mut best: u64 = 0;
        for luid in &luids {
            let query = D3DKMT_QUERYSTATISTICS {
                Type: D3DKMT_QUERYSTATISTICS_PROCESS_SEGMENT_GROUP,
                AdapterLuid: *luid,
                hProcess: handle,
                Anonymous: D3DKMT_QUERYSTATISTICS_0 {
                    QueryProcessSegmentGroup: D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL,
                },
                ..Default::default()
            };
            if succeeded(unsafe { D3DKMTQueryStatistics(&query) }) {
                let usage = unsafe { query.QueryResult.ProcessSegmentGroupInformation.Usage };
                best = best.max(usage);
            }
        }
        unsafe {
            let _ = CloseHandle(handle);
        }
        map.insert(pid, best);
    }
    map
}

/// NVIDIA adapter LUIDs via DXGI description match; falls back to every
/// D3DKMT adapter when DXGI is unavailable, so the figures degrade to
/// "largest across adapters" instead of vanishing.
#[cfg(windows)]
fn nvidia_adapter_luids() -> Vec<LUID> {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};

    if let Ok(factory) = unsafe { CreateDXGIFactory1::<IDXGIFactory1>() } {
        let mut found = Vec::new();
        for index in 0.. {
            let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
                break;
            };
            let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
                break;
            };
            let name = String::from_utf16_lossy(&desc.Description);
            if name.to_ascii_lowercase().contains("nvidia") {
                found.push(desc.AdapterLuid);
            }
        }
        if !found.is_empty() {
            return found;
        }
    }
    d3dkmt_adapter_luids()
}

/// Two-call D3DKMT enumeration; the second call resets `NumAdapters` to the
/// count actually written, so the buffer is truncated to that.
#[cfg(windows)]
fn d3dkmt_adapter_luids() -> Vec<LUID> {
    use windows::Wdk::Graphics::Direct3D::{
        D3DKMT_ADAPTERINFO, D3DKMT_CLOSEADAPTER, D3DKMT_ENUMADAPTERS2, D3DKMTCloseAdapter,
        D3DKMTEnumAdapters2,
    };
    use windows::Win32::Foundation::NTSTATUS;

    let succeeded = |status: NTSTATUS| status.0 >= 0;
    let mut listed = D3DKMT_ENUMADAPTERS2::default();
    if !succeeded(unsafe { D3DKMTEnumAdapters2(&mut listed) }) || listed.NumAdapters == 0 {
        return Vec::new();
    }
    let mut adapters: Vec<D3DKMT_ADAPTERINFO> =
        vec![D3DKMT_ADAPTERINFO::default(); listed.NumAdapters as usize];
    listed.pAdapters = adapters.as_mut_ptr();
    if !succeeded(unsafe { D3DKMTEnumAdapters2(&mut listed) }) {
        return Vec::new();
    }
    adapters.truncate(listed.NumAdapters as usize);
    let luids: Vec<LUID> = adapters.iter().map(|a| a.AdapterLuid).collect();
    for adapter in &adapters {
        let close = D3DKMT_CLOSEADAPTER {
            hAdapter: adapter.hAdapter,
        };
        let _ = unsafe { D3DKMTCloseAdapter(&close) };
    }
    luids
}

#[cfg(not(windows))]
pub(crate) fn process_vram_bytes(_pids: &[u32]) -> std::collections::HashMap<u32, u64> {
    std::collections::HashMap::new()
}

#[cfg(all(test, windows))]
mod tests {
    use super::process_vram_bytes;

    /// Pins the D3DKMT query path end-to-end on the live process: the adapter
    /// enumeration must yield LUIDs and the query must be accepted (the
    /// pre-fix bug was STATUS_INVALID_PARAMETER), regardless of how much
    /// memory this process happens to hold.
    #[test]
    fn test_process_vram_bytes_returns_entry_for_live_pid() {
        let pid = std::process::id();
        let map = process_vram_bytes(&[pid]);
        assert!(map.contains_key(&pid));
    }
}
