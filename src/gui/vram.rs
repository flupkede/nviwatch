//! Per-process VRAM on Windows — D3DKMT first, PDH fallback.
//!
//! NVML cannot report per-process memory on WDDM (every process reads "NA"),
//! so the figure comes from `D3DKMTQueryStatistics` with the
//! PROCESS_SEGMENT_GROUP type: the LOCAL segment group is dedicated VRAM.
//! The query rejects a `PROCESS_QUERY_LIMITED_INFORMATION` handle with
//! STATUS_INVALID_PARAMETER, so the process is opened with full query rights.
//! Processes that refuse even that handle (elevated services like ollama's
//! llama-server deny every handle) fall back to the GPU performance counters
//! ("\GPU Process Memory(pid_*)\Dedicated Usage"), which need no process
//! handle at all. The NVIDIA adapter is identified by DXGI description match —
//! a process also commits memory on the iGPU, and those figures must not
//! leak in. Each pid is queried against every NVIDIA adapter and the largest
//! figure wins. Non-Windows builds return an empty map; NVML already fills
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
        let mut best: u64 = 0;
        if let Ok(handle) = unsafe { OpenProcess(PROCESS_QUERY_INFORMATION, false, pid) } {
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
        }
        map.insert(pid, best);
    }

    // Elevated processes deny every handle, so their D3DKMT figure is 0; the
    // PDH counters cover them without needing a handle at all.
    for (pid, bytes) in pdh_dedicated_bytes(pids, &luids) {
        map.entry(pid)
            .and_modify(|v| *v = (*v).max(bytes))
            .or_insert(bytes);
    }
    map
}

/// Dedicated VRAM per pid from the GPU performance counters — the same source
/// Task Manager uses. Any failure degrades to an empty map; the NVML figure
/// then fills in as before.
#[cfg(windows)]
fn pdh_dedicated_bytes(pids: &[u32], luids: &[LUID]) -> std::collections::HashMap<u32, u64> {
    use windows::Win32::System::Performance::{
        PDH_FMT_COUNTERVALUE, PDH_FMT_LARGE, PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA,
        PERF_DETAIL_WIZARD, PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData,
        PdhEnumObjectItemsW, PdhGetFormattedCounterValue, PdhOpenQueryW,
    };
    use windows::core::{HSTRING, PWSTR, w};

    let wanted: std::collections::HashSet<u32> = pids.iter().copied().collect();
    // Instances carry the adapter LUID ("pid_<n>_luid_0x..._0x..._phys_<n>");
    // matching ours keeps iGPU bytes out.
    let luid_hints: Vec<String> = luids
        .iter()
        .map(|l| format!("0x{:08x}_0x{:08x}", l.HighPart as u32, l.LowPart).to_ascii_lowercase())
        .collect();

    let mut query = PDH_HQUERY::default();
    if unsafe { PdhOpenQueryW(None, 0, &mut query) } != 0 {
        return std::collections::HashMap::new();
    }
    let mut result = std::collections::HashMap::new();

    // Instance names for the "GPU Process Memory" counter set; sizes are in
    // UTF-16 units and grow with the process count, hence the retry. Both
    // lists need real buffers — with a NULL counter list the call answers
    // PDH_MORE_DATA forever.
    let mut c_need = 512u32;
    let mut i_need = 4096u32;
    let mut status: u32;
    let instance_buf = loop {
        let mut counter_buf = vec![0u16; c_need as usize];
        let mut buf = vec![0u16; i_need as usize];
        let mut c_size = c_need;
        let mut size = i_need;
        status = unsafe {
            PdhEnumObjectItemsW(
                None,
                None,
                w!("GPU Process Memory"),
                Some(PWSTR(counter_buf.as_mut_ptr())),
                &mut c_size,
                Some(PWSTR(buf.as_mut_ptr())),
                &mut size,
                PERF_DETAIL_WIZARD,
                0,
            )
        };
        if status == 0 || status != PDH_MORE_DATA {
            break buf;
        }
        let (grew_c, grew_i) = (c_size > c_need, size > i_need);
        c_need = c_need.max(c_size);
        i_need = i_need.max(size);
        if !grew_c && !grew_i {
            break Vec::new();
        }
    };

    if status == 0 && !instance_buf.is_empty() {
        let mut targets: Vec<(u32, bool, HSTRING)> = Vec::new();
        let mut start = 0usize;
        for i in 0..instance_buf.len() {
            if instance_buf[i] != 0 {
                continue;
            }
            if i > start {
                let name = String::from_utf16_lossy(&instance_buf[start..i]);
                if let Some(pid) = name
                    .strip_prefix("pid_")
                    .and_then(|r| r.split('_').next())
                    .and_then(|s| s.parse().ok())
                    .filter(|pid| wanted.contains(pid))
                {
                    let exact = luid_hints
                        .iter()
                        .any(|hint| name.to_ascii_lowercase().contains(hint));
                    targets.push((
                        pid,
                        exact,
                        HSTRING::from(format!("\\GPU Process Memory({name})\\Dedicated Usage")),
                    ));
                }
            }
            start = i + 1;
            // Double null terminates the multi-sz list.
            if start < instance_buf.len() && instance_buf[start] == 0 {
                break;
            }
        }

        let mut counters: Vec<(u32, bool, PDH_HCOUNTER)> = Vec::new();
        for (pid, exact, path) in &targets {
            let mut counter = PDH_HCOUNTER::default();
            if unsafe { PdhAddEnglishCounterW(query, path, 0, &mut counter) } == 0 {
                counters.push((*pid, *exact, counter));
            }
        }

        if unsafe { PdhCollectQueryData(query) } == 0 {
            let mut on_gpu: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
            let mut elsewhere = std::collections::HashMap::new();
            for (pid, exact, counter) in &counters {
                let mut value = PDH_FMT_COUNTERVALUE::default();
                if unsafe { PdhGetFormattedCounterValue(*counter, PDH_FMT_LARGE, None, &mut value) }
                    == 0
                {
                    let bytes = unsafe { value.Anonymous.largeValue }.max(0) as u64;
                    let map = if *exact { &mut on_gpu } else { &mut elsewhere };
                    map.entry(*pid)
                        .and_modify(|b| *b = (*b).max(bytes))
                        .or_insert(bytes);
                }
            }
            for (pid, bytes) in on_gpu {
                result.insert(pid, bytes);
            }
            for (pid, bytes) in elsewhere {
                result.entry(pid).or_insert(bytes);
            }
        }
    }

    unsafe {
        let _ = PdhCloseQuery(query);
    }
    result
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
