"""Biostrike capture server: press Record once, every connected phone records.

    uvicorn biostrike.capture_server.server:app --host 0.0.0.0 --port 8000

Works like OpenCap's own capture flow, without Stanford's servers:

  1. A coach opens the server's home page and creates a session. The control
     page shows a QR code.
  2. Each phone scans it. The camera page opens in the browser, picks its
     back camera and measures how far its clock is from the server's.
  3. The coach presses Calibrate, Neutral or Record on the control page. The
     server tells every phone to start at the same server time, then stop.
  4. Each phone uploads its clip. Once all are in, the server trims them to a
     common start, converts them to OpenCap's video layout (media.py) and files
     them into a session folder that capture_local.py processes like a
     downloaded OpenCap session.

Sync is coarse here by design, exactly as in OpenCap: the phones start within a
few frames of each other, and the pipeline's own synchronizeVideos step lines
them up to the frame from the body motion.

Phones only allow camera access on HTTPS pages (localhost excepted), so the
server must be reached over HTTPS: deploy it behind Render, `tailscale serve`,
or a Cloudflare tunnel.

Access: the camera link only lets a phone join and upload its own clips. The
control link carries a separate key needed to start and stop recordings. Set
CAPTURE_PASSCODE to require a passcode to create sessions from anywhere but
the server machine itself. `python -m biostrike.capture_server.run` sets one up
automatically, together with the tunnel.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from . import media
except ImportError:                      # run as a plain script / in tests
    import media  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STATIC = os.path.join(HERE, 'static')
# Outside the repo by default, so recordings can never be committed by accident.
DATA_DIR = os.path.abspath(os.environ.get('BIOSTRIKE_CAPTURE_DATA',
                                          os.path.expanduser('~/biostrike-capture-data')))
PASSCODE = os.environ.get('CAPTURE_PASSCODE', '')
# The https address phones use (e.g. a cloudflared link). Lets the control page
# run at http://localhost on the server itself and still hand phones a link
# they can open.
PUBLIC_URL = os.environ.get('CAPTURE_PUBLIC_URL', '').strip().rstrip('/')
if PUBLIC_URL and not re.match(r'^https://[A-Za-z0-9.-]+(:\d+)?$', PUBLIC_URL):
    raise SystemExit('CAPTURE_PUBLIC_URL must look like https://name.example.com')
MAX_UPLOAD_BYTES = int(os.environ.get('CAPTURE_MAX_UPLOAD_MB', '1024')) * 1024 * 1024
START_LEAD_S = 1.5       # phones get this long to schedule the start
STOP_LEAD_S = 0.3

ID_RE = re.compile(r'^[A-Za-z0-9_-]{8,64}$')
TRIAL_RE = re.compile(r'^[A-Za-z0-9_-]{1,48}$')
TRIAL_KINDS = ('calibration', 'neutral', 'dynamic')
# Rates a trial can ask for. 120/240 need the native app (slow-motion camera
# modes); browsers deliver at most 60 and the trial records what they give.
CAPTURE_FPS = (60, 120, 240)

# Apple identifiers that have deployed lens calibrations in CameraIntrinsics/.
IPHONE_NAMES = {
    'iPhone8,1': 'iPhone 6s', 'iPhone8,4': 'iPhone SE (1st gen)',
    'iPhone10,1': 'iPhone 8', 'iPhone10,4': 'iPhone 8', 'iPhone10,2': 'iPhone 8 Plus',
    'iPhone10,5': 'iPhone 8 Plus', 'iPhone10,3': 'iPhone X', 'iPhone10,6': 'iPhone X',
    'iPhone11,2': 'iPhone XS', 'iPhone11,4': 'iPhone XS Max', 'iPhone11,6': 'iPhone XS Max',
    'iPhone11,8': 'iPhone XR', 'iPhone12,1': 'iPhone 11', 'iPhone12,3': 'iPhone 11 Pro',
    'iPhone12,5': 'iPhone 11 Pro Max', 'iPhone12,8': 'iPhone SE (2nd gen)',
    'iPhone13,1': 'iPhone 12 mini', 'iPhone13,2': 'iPhone 12', 'iPhone13,3': 'iPhone 12 Pro',
    'iPhone13,4': 'iPhone 12 Pro Max', 'iPhone14,2': 'iPhone 13 Pro',
    'iPhone14,3': 'iPhone 13 Pro Max', 'iPhone14,4': 'iPhone 13 mini', 'iPhone14,5': 'iPhone 13',
    'iPhone14,6': 'iPhone SE (3rd gen)', 'iPhone14,7': 'iPhone 14', 'iPhone14,8': 'iPhone 14 Plus',
    'iPhone15,2': 'iPhone 14 Pro', 'iPhone15,3': 'iPhone 14 Pro Max', 'iPhone15,4': 'iPhone 15',
    'iPhone15,5': 'iPhone 15 Plus', 'iPhone16,1': 'iPhone 15 Pro', 'iPhone16,2': 'iPhone 15 Pro Max',
    'iPhone17,1': 'iPhone 16 Pro', 'iPhone17,2': 'iPhone 16 Pro Max', 'iPhone17,3': 'iPhone 16',
    'iPhone17,4': 'iPhone 16 Plus', 'iPhone17,5': 'iPhone 16e',
}

SESSION_DEFAULTS = {
    'augmentermodel': 'v0.3',
    'checkerBoard': {'black2BlackCornersHeight_n': 4, 'black2BlackCornersWidth_n': 5,
                     'placement': 'Perpendicular', 'squareSideLength_mm': 35.0},
    'filterfrequency': 'default',
    'gender_mf': None,
    'markerAugmentationSettings': {'markerAugmenterModel': 'LSTM'},
    'openSimModel': 'LaiUhlrich2022',
    'posemodel': 'hrnet',
    'scalingsetup': 'upright_standing_pose',
}


def available_models() -> List[dict]:
    root = os.path.join(REPO, 'CameraIntrinsics')
    ids = sorted((d for d in os.listdir(root) if d.startswith('iPhone')
                  and os.path.exists(os.path.join(root, d, 'Deployed', 'cameraIntrinsics.pickle'))),
                 key=lambda s: [int(x) for x in re.findall(r'\d+', s)]) if os.path.isdir(root) else []
    return [{'id': i, 'name': f"{IPHONE_NAMES.get(i, 'iPhone')} ({i})"} for i in ids]


# ── State ──────────────────────────────────────────────────────────────────

@dataclass
class Device:
    device_id: str
    token: str
    cam: int
    model: str = ''
    label: str = ''
    settings: dict = field(default_factory=dict)      # width/height/frameRate the browser reports
    clock_offset_s: Optional[float] = None            # server time - phone time
    clock_rtt_ms: Optional[float] = None
    state: str = 'connected'
    ws: Optional[WebSocket] = None

    def public(self) -> dict:
        return {'device_id': self.device_id, 'cam': f'Cam{self.cam}', 'model': self.model,
                'label': self.label, 'settings': self.settings,
                'clock_rtt_ms': self.clock_rtt_ms, 'state': self.state,
                'online': self.ws is not None}


@dataclass
class Trial:
    name: str
    kind: str
    start_at: float
    devices: List[str]
    stop_at: Optional[float] = None
    uploads: Dict[str, dict] = field(default_factory=dict)
    status: str = 'recording'          # recording → uploading → processing → ready | failed
    error: str = ''
    requested_fps: int = 60
    fps: Optional[int] = None          # rate the filed videos actually have
    warning: str = ''

    def public(self) -> dict:
        return {'name': self.name, 'kind': self.kind, 'status': self.status, 'error': self.error,
                'uploaded': sorted(self.uploads), 'devices': self.devices,
                'requested_fps': self.requested_fps, 'fps': self.fps, 'warning': self.warning}


@dataclass
class Session:
    sid: str
    key: str
    meta: dict
    devices: Dict[str, Device] = field(default_factory=dict)
    trials: Dict[str, Trial] = field(default_factory=dict)
    controllers: List[WebSocket] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f'BiostrikeSession_{self.sid}'

    @property
    def folder(self) -> str:
        return os.path.join(DATA_DIR, 'Data', self.name)

    def public(self) -> dict:
        return {'sid': self.sid, 'session_folder': self.folder, 'meta': self.meta,
                'devices': [d.public() for d in sorted(self.devices.values(), key=lambda d: d.cam)],
                'trials': [t.public() for t in self.trials.values()]}


SESSIONS: Dict[str, Session] = {}
app = FastAPI(title='Biostrike capture')
app.mount('/static', StaticFiles(directory=STATIC), name='static')


def _session(sid: str) -> Session:
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, 'No such session.')
    return s


def _require_key(s: Session, key: str) -> None:
    if not secrets.compare_digest(key or '', s.key):
        raise HTTPException(403, 'This link cannot control the session.')


def write_session_metadata(s: Session) -> None:
    """sessionMetadata.yaml in the layout OpenCap's download uses."""
    import yaml
    meta = dict(SESSION_DEFAULTS)
    meta.update({k: s.meta[k] for k in ('subjectID', 'mass_kg', 'height_m', 'checkerBoard')
                 if k in s.meta})
    meta['iphoneModel'] = {f'Cam{d.cam}': d.model for d in s.devices.values() if d.model}
    meta['cameraMapping'] = {f'Cam{d.cam}': {'device_id': d.device_id, 'phone_label': d.label}
                             for d in s.devices.values()}
    meta['biostrike'] = {'recorder': 'browser', 'session_id': s.sid}
    os.makedirs(s.folder, exist_ok=True)
    with open(os.path.join(s.folder, 'sessionMetadata.yaml'), 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=True)


async def _send(ws: Optional[WebSocket], msg: dict) -> None:
    if ws is None:
        return
    try:
        await ws.send_text(json.dumps(msg))
    except Exception:
        pass


async def _broadcast_status(s: Session) -> None:
    msg = {'type': 'status', **s.public()}
    for ws in list(s.controllers):
        await _send(ws, msg)


# ── Pages ──────────────────────────────────────────────────────────────────

@app.get('/', response_class=HTMLResponse)
def home():
    return FileResponse(os.path.join(STATIC, 'home.html'))


@app.get('/control/{sid}', response_class=HTMLResponse)
def control_page(sid: str):
    _session(sid)
    return FileResponse(os.path.join(STATIC, 'control.html'))


@app.get('/join/{sid}', response_class=HTMLResponse)
def camera_page(sid: str):
    _session(sid)
    return FileResponse(os.path.join(STATIC, 'camera.html'))


# ── API ────────────────────────────────────────────────────────────────────

@app.get('/api/models')
def models():
    return available_models()


def is_local_request(request: Request) -> bool:
    """True only for a browser on the server machine itself.

    Tunnel traffic also arrives from 127.0.0.1 (cloudflared connects locally),
    but Cloudflare always stamps it with CF-Connecting-IP, overwriting any
    value a client sends, so its presence marks a request from the internet.
    """
    host = request.client.host if request.client else ''
    return host in ('127.0.0.1', '::1') and 'cf-connecting-ip' not in request.headers


@app.get('/api/config')
def config(request: Request):
    return {'public_url': PUBLIC_URL,
            'passcode_required': bool(PASSCODE) and not is_local_request(request)}


@app.get('/api/sessions/{sid}/qr.svg')
def join_qr(sid: str, request: Request):
    """QR code for the phones' join link, drawn here so the control page
    needs nothing from outside sites."""
    from fastapi.responses import Response
    import segno
    _session(sid)
    base = PUBLIC_URL if (PUBLIC_URL and is_local_request(request)) else str(request.base_url).rstrip('/')
    buf = __import__('io').BytesIO()
    segno.make(f'{base}/join/{sid}', error='m').save(buf, kind='svg', scale=6, border=2)
    return Response(buf.getvalue(), media_type='image/svg+xml')


@app.get('/api/time')
def server_time():
    return {'t': time.time()}


@app.post('/api/sessions')
async def create_session(request: Request):
    body = await request.json()
    if (PASSCODE and not is_local_request(request)
            and not secrets.compare_digest(str(body.get('passcode', '')), PASSCODE)):
        raise HTTPException(403, 'Wrong passcode.')
    subject = str(body.get('subjectID') or 'athlete').strip()
    if not TRIAL_RE.match(subject):
        raise HTTPException(400, 'Athlete ID: letters, numbers, - and _ only.')
    try:
        mass, height = float(body['mass_kg']), float(body['height_m'])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, 'Mass (kg) and height (m) are required.')
    if not (20 <= mass <= 250 and 1.0 <= height <= 2.4):
        raise HTTPException(400, 'Mass must be 20-250 kg and height 1.0-2.4 m.')
    # Checkerboard: OpenCap's default print is 5x4 inner corners, 35 mm squares.
    # Larger squares calibrate more accurately at distance.
    try:
        board = {'black2BlackCornersWidth_n': int(body.get('board_cols', 5)),
                 'black2BlackCornersHeight_n': int(body.get('board_rows', 4)),
                 'squareSideLength_mm': float(body.get('square_mm', 35.0)),
                 'placement': 'Perpendicular'}
    except (TypeError, ValueError):
        raise HTTPException(400, 'Checkerboard size must be numbers.')
    if not (2 <= board['black2BlackCornersWidth_n'] <= 20 and 2 <= board['black2BlackCornersHeight_n'] <= 20
            and 10 <= board['squareSideLength_mm'] <= 300):
        raise HTTPException(400, 'Checkerboard: 2-20 inner corners each way, squares 10-300 mm.')
    s = Session(sid=secrets.token_urlsafe(9), key=secrets.token_urlsafe(18),
                meta={'subjectID': subject, 'mass_kg': mass, 'height_m': height,
                      'checkerBoard': board})
    SESSIONS[s.sid] = s
    write_session_metadata(s)
    return {'sid': s.sid, 'key': s.key, 'control_url': f'/control/{s.sid}?key={s.key}',
            'join_url': f'/join/{s.sid}'}


@app.get('/api/sessions/{sid}')
def session_status(sid: str, key: str = ''):
    s = _session(sid)
    _require_key(s, key)
    return s.public()


@app.post('/api/sessions/{sid}/trials')
async def start_trial(sid: str, request: Request):
    s = _session(sid)
    body = await request.json()
    _require_key(s, body.get('key', ''))
    kind = body.get('kind', 'dynamic')
    try:
        requested_fps = int(body.get('fps', 60))
    except (TypeError, ValueError):
        requested_fps = 0
    if requested_fps not in CAPTURE_FPS:
        raise HTTPException(400, f'fps must be one of {CAPTURE_FPS}')
    if kind not in TRIAL_KINDS:
        raise HTTPException(400, f'kind must be one of {TRIAL_KINDS}')
    name = kind if kind != 'dynamic' else str(body.get('name') or '').strip()
    if not TRIAL_RE.match(name) or (kind == 'dynamic' and name in ('calibration', 'neutral')):
        raise HTTPException(400, 'Trial name: letters, numbers, - and _ only.')
    if any(t.status == 'recording' for t in s.trials.values()):
        raise HTTPException(409, 'A recording is already running. Stop it first.')
    if name in s.trials and kind == 'dynamic':
        raise HTTPException(409, f'Trial {name} already exists.')
    ready = [d for d in s.devices.values() if d.ws is not None and d.model and d.clock_offset_s is not None]
    if len(ready) < 2:
        raise HTTPException(409, 'Need at least 2 phones connected, each with its model chosen.')
    # Same order OpenCap enforces: the neutral pose and the swings are only
    # usable from cameras that were calibrated where they now stand.
    if kind != 'calibration' and not _calibrated(s):
        raise HTTPException(409, 'Calibrate the cameras first.')
    t = Trial(name=name, kind=kind, start_at=time.time() + START_LEAD_S,
              devices=[d.device_id for d in ready], requested_fps=requested_fps)
    s.trials[name] = t           # re-recording calibration/neutral replaces it
    for d in ready:
        d.state = 'recording'
        await _send(d.ws, {'type': 'start', 'trial': name, 'kind': kind, 'at': t.start_at,
                           'fps': requested_fps})
    await _broadcast_status(s)
    return t.public()


@app.post('/api/sessions/{sid}/trials/{name}/stop')
async def stop_trial(sid: str, name: str, request: Request):
    s = _session(sid)
    body = await request.json()
    _require_key(s, body.get('key', ''))
    t = s.trials.get(name)
    if t is None or t.status != 'recording':
        raise HTTPException(409, 'That trial is not recording.')
    t.stop_at = time.time() + STOP_LEAD_S
    t.status = 'uploading'
    for did in t.devices:
        d = s.devices.get(did)
        if d:
            d.state = 'uploading'
            await _send(d.ws, {'type': 'stop', 'trial': name, 'at': t.stop_at})
    await _broadcast_status(s)
    return t.public()


@app.post('/api/sessions/{sid}/upload')
async def upload(sid: str, device_id: str = Form(...), token: str = Form(...),
                 trial: str = Form(...), started_at: float = Form(...),
                 stopped_at: float = Form(...), file: UploadFile = File(...),
                 background: BackgroundTasks = None):
    s = _session(sid)
    d = s.devices.get(device_id)
    if d is None or not secrets.compare_digest(token, d.token):
        raise HTTPException(403, 'Unknown phone for this session.')
    t = s.trials.get(trial)
    if t is None or device_id not in t.devices:
        raise HTTPException(400, 'This phone was not part of that recording.')
    if device_id in t.uploads:
        raise HTTPException(409, 'Already uploaded.')
    if not (stopped_at > started_at and abs(started_at - t.start_at) < 30):
        raise HTTPException(400, 'Recording times do not match the session clock.')

    raw_dir = os.path.join(DATA_DIR, 'raw', s.sid, trial)
    os.makedirs(raw_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or '')[1].lower()
    ext = ext if ext in ('.mp4', '.mov', '.webm', '.mkv') else '.bin'
    raw_path = os.path.join(raw_dir, f'Cam{d.cam}{ext}')
    written = 0
    with open(raw_path, 'wb') as out:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                os.remove(raw_path)
                raise HTTPException(413, 'Recording is too large.')
            out.write(chunk)
    t.uploads[device_id] = {'path': raw_path, 'started_at': started_at,
                            'stopped_at': stopped_at, 'bytes': written}
    d.state = 'idle'
    if set(t.uploads) >= set(t.devices):
        t.status = 'processing'
        background.add_task(_process_trial, s, t)
    await _broadcast_status(s)
    return {'ok': True}


def _calibrated(s: Session) -> bool:
    t = s.trials.get('calibration')
    return t is not None and t.status == 'ready'


def assemble_trial(s: Session, t: Trial) -> dict:
    """Trim every clip to the shared window and file it as OpenCap would.

    Each phone reports when it actually started and stopped, in server time.
    The shared window runs from the latest start to the earliest stop, so
    every camera covers the same span; frame-level alignment is left to the
    pipeline's synchronizeVideos, as in OpenCap.
    """
    ups = {did: t.uploads[did] for did in t.devices}
    # Every camera must end up at one frame rate. When phones ran at different
    # rates, the faster ones keep every n-th frame down to the slowest; the
    # slow ones are never padded with copies of their own frames.
    rates = {did: media.nominal_fps(media.probe(u['path'])['frame_rate']) for did, u in ups.items()}
    fps = min(rates.values())
    if len(set(rates.values())) > 1:
        slow = sorted({f'Cam{s.devices[d].cam}' for d, r in rates.items() if r == fps})
        t.warning = (f'Cameras ran at different frame rates ({", ".join(sorted({f"{r} fps" for r in rates.values()}))}); '
                     f'all were reduced to {fps} fps to match {", ".join(slow)}.')
    t.fps = fps
    common_start = max(u['started_at'] for u in ups.values())
    common_stop = min(u['stopped_at'] for u in ups.values())
    if common_stop - common_start < 0.5:
        raise media.VideoError('The phones did not record a common window of at least 0.5 s.')
    report = {'common_start': common_start, 'duration_s': common_stop - common_start,
              'fps': fps, 'requested_fps': t.requested_fps, 'warning': t.warning, 'cams': {}}
    for did, u in ups.items():
        d = s.devices[did]
        dst = os.path.join(s.folder, 'Videos', f'Cam{d.cam}', 'InputMedia', t.name, t.name + '.mov')
        trim = common_start - u['started_at']
        info = media.normalize_video(u['path'], dst, trim_start_s=trim,
                                     duration_s=common_stop - common_start, fps=fps)
        report['cams'][f'Cam{d.cam}'] = {'trim_s': round(trim, 4), 'source': info,
                                         'model': d.model, 'clock_rtt_ms': d.clock_rtt_ms}
    with open(os.path.join(s.folder, 'Videos', f'{t.name}_capture.json'), 'w') as f:
        json.dump(report, f, indent=1)
    write_session_metadata(s)
    return report


async def _process_trial(s: Session, t: Trial) -> None:
    try:
        await asyncio.to_thread(assemble_trial, s, t)
        t.status = 'ready'
        if t.kind == 'calibration':
            # A new calibration invalidates any camera parameters computed before.
            for cam_dir in os.listdir(os.path.join(s.folder, 'Videos')):
                p = os.path.join(s.folder, 'Videos', cam_dir, 'cameraIntrinsicsExtrinsics.pickle')
                if os.path.exists(p):
                    os.remove(p)
    except Exception as e:                    # shown on the control page
        t.status, t.error = 'failed', str(e)
    await _broadcast_status(s)


# ── WebSockets ─────────────────────────────────────────────────────────────

@app.websocket('/ws/{sid}/camera/{device_id}')
async def camera_ws(ws: WebSocket, sid: str, device_id: str):
    s = SESSIONS.get(sid)
    if s is None or not ID_RE.match(device_id):
        await ws.close(code=4404)
        return
    await ws.accept()
    d = s.devices.get(device_id)
    if d is None:
        d = Device(device_id=device_id, token=secrets.token_urlsafe(18),
                   cam=max((x.cam for x in s.devices.values()), default=-1) + 1)
        s.devices[device_id] = d
    d.ws, d.state = ws, 'connected'
    await _send(ws, {'type': 'welcome', 'cam': f'Cam{d.cam}', 'token': d.token,
                     'model': d.model, 'label': d.label})
    await _broadcast_status(s)
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get('type')
            if kind == 'ping':
                await _send(ws, {'type': 'pong', 't0': msg.get('t0'), 'ts': time.time()})
            elif kind == 'clock':
                d.clock_offset_s = float(msg['offset_s'])
                d.clock_rtt_ms = round(float(msg['rtt_ms']), 1)
                await _broadcast_status(s)
            elif kind == 'hello':
                model = str(msg.get('model', ''))
                if model in {m['id'] for m in available_models()}:
                    d.model = model
                d.label = str(msg.get('label', ''))[:40]
                reported = msg.get('settings') if isinstance(msg.get('settings'), dict) else {}
                d.settings = {k: reported.get(k) for k in ('width', 'height', 'frameRate', 'mimeType')}
                write_session_metadata(s)
                await _broadcast_status(s)
            elif kind == 'state':
                d.state = str(msg.get('state', ''))[:20]
                await _broadcast_status(s)
    except (WebSocketDisconnect, ValueError, KeyError, TypeError):
        pass
    finally:
        if d.ws is ws:
            d.ws = None
        await _broadcast_status(s)


@app.websocket('/ws/{sid}/control')
async def control_ws(ws: WebSocket, sid: str, key: str = ''):
    s = SESSIONS.get(sid)
    if s is None or not secrets.compare_digest(key, s.key):
        await ws.close(code=4403)
        return
    await ws.accept()
    s.controllers.append(ws)
    await _send(ws, {'type': 'status', **s.public()})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        s.controllers.remove(ws)
