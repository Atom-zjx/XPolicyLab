## Policy

- Name / paper / upstream repo: InternW0_delta / not released /
  evaluation-only inference closure vendored under `wam_runtime/`
- Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`,
  `action_type=joint`
- Training support: eval-only (training release ETA: before the end of October 2026)

## Components

- [x] `install.sh`
- [x] `model.py` + `__init__.py`
- [x] images remain RGB; no adapter-side decode or channel swap
- [x] `deploy.yml` with the standard key set and `protocol: ws`
- [x] `deploy.py` aligned with `demo_policy`
- [x] `eval.sh`, `setup_eval_policy_server.sh`, and
  `setup_eval_env_client.sh`
- [x] eval-only exception declared for omitted `process_data.sh` / `train.sh`
- [x] policy README with installation, assets, eval, and configuration

## Testing

- [x] `bash -n`, `py_compile`, and `git diff --check` pass
- [x] official decode, encode, channel-swap, and path greps are clean
- [x] raw-RGB and encoded-RGB WebSocket debug closed loops pass
- [x] z-score normalization and the 14D-to-80D scatter mapping pass
- [x] real-weight GPU forward loads Wan2.2 + RynnBrain and returns the expected
  10-action replan chunk
- [x] simulator evaluation: 54 tasks, 1,444 / 6,300 successes (22.92%),
  mean score 30.35

## Checkpoint (required for leaderboard evaluation)

No weights are committed. `download_assets.sh` downloads Wan2.2 and RynnBrain
from ModelScope. The evaluation checkpoint is `robodojo.pt` in
[InternRobotics/InternW0-Delta-RoboDojo](https://huggingface.co/InternRobotics/InternW0-Delta-RoboDojo):

```bash
bash policy/InternW0_delta/download_checkpoint.sh
```

## Limitations / notes

- Evaluation contract: horizon 32, replan 10, 10 denoising steps,
  `eval_batch: false`.
- Uses the checked-in z-score `global_mean/global_std` statistics.
- Discrete Action RoPE is enabled; physical-time Action RoPE and fan-in
  calibration are disabled; the recent-KV-cache mask fix is included.
- Camera processing is a resize-only three-camera T canvas with no crop or
  color jitter.
- Wan2.2, RynnBrain, and the post-trained checkpoint are intentionally excluded
  from Git and source archives.
