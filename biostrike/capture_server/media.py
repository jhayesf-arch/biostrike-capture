"""Turn a phone's browser recording into the video OpenCap's pipeline expects.

OpenCap's iPhone app writes 1280x720 landscape frames tagged to display upright
in portrait (QuickTime orientation 90, display matrix -90 deg), at 60 fps. The
deployed lens calibration (CameraIntrinsics/<model>/Deployed) is measured in
that frame, and the pipeline reads the orientation tag to decide whether to
rotate it. A browser recording differs on every count: portrait pixels,
variable frame rate, often WebM or a different resolution.

normalize_video re-encodes it so the pipeline sees exactly what it would get
from OpenCap's app:
  - trimmed to the session's common start time
  - stored as 1280x720 landscape, tagged to display upright in portrait
  - constant 60 fps (browsers record variable frame rate)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

TARGET_FPS = 60
TARGET_DISPLAY_W, TARGET_DISPLAY_H = 720, 1280      # portrait, as displayed


class VideoError(ValueError):
    """The recording can't be used; the message is shown to the user."""


def _ffmpeg() -> str:
    exe = shutil.which('ffmpeg')
    if not exe:
        raise RuntimeError('ffmpeg is not on PATH')
    return exe


def probe(path: str) -> dict:
    """Displayed width/height (rotation applied), frame rate and duration."""
    exe = shutil.which('ffprobe')
    if not exe:
        raise RuntimeError('ffprobe is not on PATH')
    out = subprocess.run(
        [exe, '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_format',
         '-of', 'json', path], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise VideoError('The uploaded file is not a readable video.')
    meta = json.loads(out.stdout or '{}')
    streams = meta.get('streams') or []
    if not streams:
        raise VideoError('The uploaded file has no video track.')
    s = streams[0]
    w, h = int(s['width']), int(s['height'])
    rotation = 0
    for sd in s.get('side_data_list') or []:
        if 'rotation' in sd:
            rotation = int(round(float(sd['rotation'])))
    if rotation % 180:
        w, h = h, w
    num, _, den = (s.get('avg_frame_rate') or '0/1').partition('/')
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    duration = float((meta.get('format') or {}).get('duration') or s.get('duration') or 0.0)
    return {'display_w': w, 'display_h': h, 'fps': fps, 'duration': duration,
            'rotation': rotation}


def normalize_video(src: str, dst: str, trim_start_s: float = 0.0,
                    duration_s: float | None = None) -> dict:
    """Write dst (.mov) in OpenCap's layout. Returns the source probe."""
    info = probe(src)
    w, h = info['display_w'], info['display_h']
    if w > h:
        raise VideoError('This phone recorded in landscape. Mount phones upright '
                         '(portrait), as OpenCap does; the camera calibration '
                         'assumes it.')
    if abs(w / h - TARGET_DISPLAY_W / TARGET_DISPLAY_H) > 0.01:
        raise VideoError(f'The video is {w}x{h}, not 16:9. The phone cropped its '
                         'camera image, so the stored lens calibration would not fit.')

    ff = _ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = os.path.join(tmp, 'stage.mov')
        # Decoding applies the source's own rotation, so frames arrive as
        # displayed (portrait). Scale to 720x1280 (same field of view, so the
        # intrinsics still hold), resample to constant 60 fps, then turn them
        # 90 deg counter-clockwise into the landscape storage iPhones use.
        cmd = [ff, '-v', 'error', '-y']
        if trim_start_s > 0:
            cmd += ['-ss', f'{trim_start_s:.4f}']
        cmd += ['-i', src]
        if duration_s:
            cmd += ['-t', f'{duration_s:.4f}']
        cmd += ['-an', '-vf',
                f'scale={TARGET_DISPLAY_W}:{TARGET_DISPLAY_H},fps={TARGET_FPS},transpose=2',
                # A keyframe every second, as in OpenCap's iPhone files: the
                # calibration step grabs its checkerboard image with
                # `-skip_frame nokey`, so it can only see keyframes.
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17',
                '-g', str(TARGET_FPS), stage]
        _run(cmd)
        # Tag it to display upright, exactly as OpenCap's .mov files are:
        # display matrix -90 deg plus the QuickTime orientation the pipeline
        # reads (utilsChecker.getVideoRotation).
        _run([ff, '-v', 'error', '-y', '-display_rotation', '-90', '-i', stage,
              '-c', 'copy', '-metadata', 'com.apple.quicktime.video-orientation=90',
              '-movflags', 'use_metadata_tags', dst])
    return info


def _run(cmd: list) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise VideoError('Video conversion failed: ' + (r.stderr.strip().splitlines() or ['?'])[-1])
