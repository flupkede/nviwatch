# nviwatch

gpui GUI for monitoring NVIDIA GPU processes on Windows (WDDM).

## GPU memory accounting (hybrid graphics)

- NVML per-process memory reads "NA" on WDDM. Per-process VRAM comes from
  D3DKMT (`PROCESS_SEGMENT_GROUP`, LOCAL group, needs a full
  `PROCESS_QUERY_INFORMATION` handle) and, for processes that deny every
  handle — elevated services like ollama's llama-server refuse even
  `PROCESS_QUERY_LIMITED_INFORMATION` — from the PDH counters
  `GPU Process Memory(pid_*) Dedicated Usage`, matched against the NVIDIA
  adapter LUIDs from DXGI. PDH needs no process handle.
- The Intel iGPU has no dedicated VRAM: its Dedicated counter reads 0 and
  all allocations sit in Shared = system RAM (dwm holds several GiB there,
  growing with display count).
- dwm mirrors composition surfaces onto the NVIDIA adapter, so its
  per-process Dedicated figure over-reports and the per-process sum can
  exceed the adapter total. The graphics band is therefore clamped to
  `memory_used − compute`; the compute (LLM) band stays exact — CUDA
  allocations are real.
- LLM vs video classification is name-based (`classify_by_name`): NVML's
  compute/graphics lists are useless on WDDM, both contain every pid.
- The ollama CLI legitimately shows 0 VRAM; the model lives in the
  llama-server child process, which NVML lists separately.
