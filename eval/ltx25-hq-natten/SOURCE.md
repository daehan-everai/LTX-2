# Upstream pin

This tree is the **LTX-2.5 eval overlay we actually run**, not the LTX-2.3
`packages/` at the repo root.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/Lightricks/LTX-2 |
| Commit | `9bf45d0c6e6360f1e0919b7e6cb8bea4492025bf` (`main`, 2026-08-12) |
| Message | `fix(env): scope training bundle to Linux CUDA` |
| Packages | `packages/ltx-core`, `packages/ltx-pipelines` |
| Runner | EverAI `run_official.py` / `run_common.py` (HQ + natten, no SGLang) |

Do **not** use `packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py`
at the repo root for 2.5 eval. That file is the older 2.3 HQ pipeline (321 lines).
This overlay’s HQ pipeline is 409 lines and is what infer pods import.
