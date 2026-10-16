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

### Result on a real swing (Quinn_1_60fps, 2 cameras)

- Markers: max difference 0.0005 mm over 63 markers × 343 frames.
- Joint angles: median RMSE 0.14° over 33 coordinates.
- The only coordinates that exceed the tolerance are the lead (left) arm's,
  for about 10 frames (t ≈ 4.18–4.35 s) right after the elbow reaches full
  extension. The cause is not the OpenSim version: Stanford's own image
  (`stanfordnmbl/opensim-python:4.3`, OpenSim 4.3.1) gives the same result
  as 4.6. At 0° the elbow sits on its joint limit, and the markers can't tell
  which way the arm leaves it. IK then jumps branches in a single frame
  (OpenCap: 2° → 67°; ours: 1° → 101°, 10 frames later). Both jumps are
  physically impossible. **Treat lead-elbow and forearm angles and velocities
  in the frames after full extension as unreliable in OpenCap's output as
  well as ours.**

To rerun IK with Stanford's OpenSim version (Docker, no Python needed):

```bash
docker run --rm -v "$PWD":"$PWD" -w "$PWD" stanfordnmbl/opensim-python:4.3 \
    opensim-cmd -o error run-tool path/to/Setup_IK_<trial>.xml
```

Dropping `--reuse-opencap-keypoints` runs MMPose pose detection too. That is the
step that needs an NVIDIA GPU, and it is the second half of the acceptance test.

Run `--check-only` first: it lists anything missing from the session folder
without running anything.

## Recording with our own phones (capture server)

`biostrike/capture_server` replaces OpenCap's recording flow: press Record once
and every connected phone records. No app install is needed; phones use a web
page.

```bash
conda create -n biostrike python=3.11 -y
conda activate biostrike
pip install -r biostrike/requirements-capture.txt
brew install ffmpeg
uvicorn biostrike.capture_server.server:app --port 8000
```

Or do it all in one command (after `brew install cloudflared`):

```bash
python -m biostrike.capture_server.run
```

This starts the server and a free `https://…trycloudflare.com` link for the
phones, waits until the link answers, keeps the Mac awake, and opens the control
page at `http://localhost:8000`. It prints a passcode, which is asked only of
browsers coming in over the internet. Ctrl+C stops everything. Commands here carry no trailing `# comments`: macOS's zsh
passes them to the command as arguments.

1. **Home page:** enter the athlete and the checkerboard size, then create a session.
   That opens the **control page**, which has a QR code. Keep the control link
   private: it starts and stops recordings.
2. **Each phone** scans the QR code, picks its iPhone model and taps Start camera.
   Phones must be mounted upright (portrait). The control page lists every
   phone with its resolution, frame rate and clock-sync accuracy.
3. Press **Calibrate**: hold the checkerboard still, seen *at an angle* by
   every phone. Then press **Neutral** with the athlete standing still, then
   **Record / Stop** for each swing.
4. Each phone uploads its clip. The server trims the clips to a shared window,
   converts them to the exact video layout of OpenCap's iPhone app, and files
   them into a session folder (`~/biostrike-capture-data/Data/BiostrikeSession_<id>`).
   Process that folder with the command the control page shows:
   `capture_local.py --session <folder> --trial swing_1 --rescale`. This first
   solves the calibration from the checkerboard trial, then sizes the model
   from the neutral trial.

**Control page on the server machine:** open `http://localhost:8000` there and
start the server with the tunnel link, so the QR code still hands phones a link
they can open:
`CAPTURE_PUBLIC_URL=https://xxxx.trycloudflare.com uvicorn biostrike.capture_server.server:app --port 8000`

**HTTPS is required.** Phones only allow camera access on `https://` pages.
Run the server behind Render, `tailscale serve`, or a Cloudflare tunnel. Plain
`http://localhost` works only on the computer itself.

**Sync:** the server tells every phone to start at the same server time. Each
phone measures its own clock offset first, so the phones start within a few
frames of each other. The pipeline's own `synchronizeVideos` then aligns them
to the frame from the body motion, exactly as OpenCap does. No clap or audio
is needed.

**Access:** session links are unguessable. The camera link can only join and
upload that phone's own clips. Start/stop needs the control key. Set
`CAPTURE_PASSCODE` to require a passcode before anyone can create a session.
Sessions live in memory; the recorded files persist, but restarting the server
ends live sessions.

### Validated

- Real-browser run (Chromium, two fake portrait cameras plus a control page):
  join, clock sync, calibrate → neutral → record → stop, uploads, conversion.
- Calibration round trip: a checkerboard rendered through the iPhone X lens
  model, recorded the way a browser does and converted, then solved by
  OpenCap's own code. The camera came back 2–39 mm from its true position at
  1.3–2 m (≤2% of distance). A full two-camera `main.py` calibration run on
  such a session put both cameras within 8 and 21 mm.
- Viewed exactly head-on, a flat board is ambiguous: the two candidate solutions
  sit about ±240 mm either side of the truth at 1.5 m. Angle the phones to the board.

### iPhone app (120/240 fps)

Safari cannot reach the iPhone's slow-motion camera modes; OpenCap's own app
records 120/240 fps through AVFoundation. `biostrike/ios` is a native app for
the same server: frame-exact start/stop from sensor timestamps, stabilization
off, focus locked at calibration, model detected automatically. It builds free
with an Apple ID; see `biostrike/ios/README.md`. Not yet compiled: written
where no Xcode is available.

The server keeps each camera's real frame rate (30/60/120/240). Cameras running
at different rates are all reduced to the slowest by keeping every n-th frame;
frames are never duplicated to raise a rate.

### First real-phone session (iPhone 14 + iPhone 11 Pro, Safari)

- Recording, upload and conversion worked on both phones. Safari tagged some
  clips as rotated 180 deg; they still came out upright.
- The phones' measured clocks were within ±10–14 ms (round trips of 20–28 ms).
  Each phone reported starting within 2 ms of the other.
- **Frame rate:** the iPhone 14 gave 60 fps; the iPhone 11 Pro gave only 30 fps
  when the page merely asked for 60. The page now requires ≥55 fps first.
- **Calibration:** OpenCap's code found the board in both cameras at about 3 m.
  But each square was only ~10 px, and in one view the board was at the frame
  edge, so the two pose solutions disagreed (cameras 30° vs 65° apart). Use the
  board at 1.5–2 m, centred where the hitter stands, or print bigger squares.
- The neutral pose had been recorded before calibration, with one phone aimed
  elsewhere. The server now requires Calibrate first, as OpenCap does.

### Not yet tested on real iPhones

- **Frame rate.** Browsers may give 30 fps instead of 60, and the control page
  warns when they do. If iPhones won't give 60 fps in the browser, the next
  step is a native app talking to this same server.
- **Field of view.** Whether Safari's camera image is the same, uncropped
  field of view that the deployed lens calibrations were measured on. The
  server refuses anything that isn't 16:9, and calibration accuracy on a real
  session will confirm the rest.
- **Pose detection.** It still needs MMPose installed (GPU, or slow on CPU)
  before these sessions produce `.mot` files.

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
python biostrike/capture_server/test_capture_server.py
python biostrike/test_calibration_synthetic.py
```

The last two need ffmpeg and the capture requirements (OpenCV 4.x). The first two
need neither TensorFlow, OpenSim nor a GPU. The upstream pipeline call is
replaced with a stand-in, so they test our session handling, not Stanford's code.
