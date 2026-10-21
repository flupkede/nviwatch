//! Per-process VRAM on Windows via D3DKMT — the source Task Manager uses.
//!
//! NVML cannot report per-process memory on WDDM (every process reads "NA"),
//! so the figure comes from `D3DKMTQueryStatistics` with the
//! PROCESS_SEGMENT_GROUP type: the LOCAL segment group is dedicated VRAM.
//! Each pid is queried against every adapter and the largest figure wins —
//! a process commits memory on the adapter it renders on, so no NVML↔DXGI
//! index mapping is needed. Non-Windows builds return an empty map; NVML
//! already fills `used_gpu_memory` there.

#[cfg(windows)]
pub(crate) fn process_vram_bytes(pids: &[u32]) -> std::collections::HashMap<u32, u64> {
    use windows::Wdk::Graphics::Direct3D::{
        D3DKMT_ADAPTERINFO, D3DKMT_CLOSEADAPTER, D3DKMT_ENUMADAPTERS2,
        D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL, D3DKMT_QUERYSTATISTICS, D3DKMT_QUERYSTATISTICS_0,
        D3DKMT_QUERYSTATISTICS_PROCESS_SEGMENT_GROUP, D3DKMTCloseAdapter, D3DKMTEnumAdapters2,
        D3DKMTQueryStatistics,
    };
    use windows::Win32::Foundation::{CloseHandle, NTSTATUS};
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    let succeeded = |status: NTSTATUS| status.0 >= 0;
    let mut map = std::collections::HashMap::new();
    if pids.is_empty() {
        return map;
    }

    // Two-call enumeration: first pass returns the count, the second fills a
    // caller-owned buffer.
    let mut listed = D3DKMT_ENUMADAPTERS2::default();
    if !succeeded(unsafe { D3DKMTEnumAdapters2(&mut listed) }) || listed.NumAdapters == 0 {
        return map;
    }
    let mut adapters: Vec<D3DKMT_ADAPTERINFO> =
        vec![D3DKMT_ADAPTERINFO::default(); listed.NumAdapters as usize];
    listed.pAdapters = adapters.as_mut_ptr();
    if !succeeded(unsafe { D3DKMTEnumAdapters2(&mut listed) }) {
        return map;
    }

    for &pid in pids {
        let Ok(handle) = (unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) })
        else {
            continue;
        };
        let mut best: u64 = 0;
        for adapter in &adapters {
            let query = D3DKMT_QUERYSTATISTICS {
                Type: D3DKMT_QUERYSTATISTICS_PROCESS_SEGMENT_GROUP,
                AdapterLuid: adapter.AdapterLuid,
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

    // Enumerated adapter handles are released so repeated polls cannot leak.
    for adapter in &adapters {
        let close = D3DKMT_CLOSEADAPTER {
            hAdapter: adapter.hAdapter,
        };
        let _ = unsafe { D3DKMTCloseAdapter(&close) };
    }
    map
}

#[cfg(not(windows))]
pub(crate) fn process_vram_bytes(_pids: &[u32]) -> std::collections::HashMap<u32, u64> {
    std::collections::HashMap::new()
}
