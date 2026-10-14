# Biostrike fork of OpenCap

Fork of [stanfordnmbl/opencap-core](https://github.com/stanfordnmbl/opencap-core) for
running the OpenCap pipeline on our own hardware instead of Stanford's servers.
It produces the `.mot` / `.trc` files the swing-optimizer-pro analyzer consumes.

**Upstream code is unchanged.** Everything we add lives in `biostrike/`, so pulling
Stanford's updates stays a clean merge:

```bash
git remote add upstream https://github.com/stanfordnmbl/opencap-core.git
git fetch upstream && git merge upstream/main
```

## How it differs from upstream

| | Upstream | This fork |
|---|---|---|
| Entry point | `app.py` polls Stanford's API queue for trials | `biostrike/capture_local.py` runs a local session folder; no Stanford calls |
| Pose detector | Defaults to OpenPose | **MMPose (HRNet) only**. OpenPose is licensed for non-commercial use. MMPose is also what OpenCap's own cloud uses (`posemodel: hrnet`) |
| Outputs | Written into the session folder | Written to `MarkerData/mmpose_0.8/biostrike` and `OpenSimData/mmpose_0.8/biostrike` inside a *copy*, so OpenCap's originals stay as the reference |

## Phase 1 acceptance test: do we reproduce OpenCap?

Take a session already processed by OpenCap, run it through this fork, and diff
the results:

```bash
python biostrike/capture_local.py --session ~/Downloads/OpenCapData_<id> \
    --trial Quinn_1_60fps_1 --workdir runs --reuse-opencap-keypoints
```

`--reuse-opencap-keypoints` feeds OpenCap's own 2-D keypoints into the pipeline,
which then skips pose detection. **No GPU is needed.** It validates everything
after pose detection: video sync, triangulation, marker augmentation and OpenSim
IK. The run compares its output against OpenCap's automatically. To compare files
yourself:

```bash
python biostrike/compare_to_opencap.py ours.trc reference.trc   # per-marker RMSE, mm
python biostrike/compare_to_opencap.py ours.mot reference.mot   # per-coordinate RMSE, deg
```

Pass thresholds default to 10 mm per marker and 2° per joint. A faithful
reproduction should sit far below them. On real data the comparison tool reads
0.00 mm for a file against itself and ~170 mm between two different trials of
the same athlete, so it clearly separates the two cases.

Dropping `--reuse-opencap-keypoints` runs MMPose pose detection too. That is the
step that needs an NVIDIA GPU, and it is the second half of the acceptance test.

Run `--check-only` first: it lists anything missing from the session folder
without running anything.

## Requirements

- Python dependencies from `requirements.txt` (TensorFlow for the marker augmenter)
- OpenSim 4.x via conda (inverse kinematics)
- `ffmpeg` on PATH (the pipeline writes a rotated copy of each video, even when
  pose detection is skipped)
- For pose detection only: MMPose and an NVIDIA GPU (see `docker/`)

## Session folder

Upload or point at the **whole** downloaded session folder. The two cameras' files
have identical names (`Videos/Cam0/...` vs `Videos/Cam1/...`), so once pulled out
of their folders they can't be matched back up.

## Licensing notes

- opencap-core, OpenSim, TensorFlow and MMPose code are Apache 2.0.
- OpenPose is non-commercial only. This fork does not use it.
- Pretrained pose-model weights (HRNet / COCO-WholeBody) can carry terms separate
  from the code. Confirm before commercial use.

## Tests

```bash
python biostrike/test_compare.py
python biostrike/test_capture_local.py
```

Neither needs TensorFlow, OpenSim or a GPU. The upstream pipeline call is
replaced with a stand-in, so they test our session handling, not Stanford's code.
