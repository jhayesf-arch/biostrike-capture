"""End-to-end tests for the capture server with two simulated phones.

Needs fastapi, httpx, pyyaml and ffmpeg/ffprobe on PATH. Run:
    python biostrike/capture_server/test_capture_server.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_TMP = tempfile.mkdtemp(prefix='bs_capture_')
os.environ['BIOSTRIKE_CAPTURE_DATA'] = _TMP

import yaml  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import media  # noqa: E402
import server  # noqa: E402

client = TestClient(server.app)
MODEL = 'iPhone10,3'


def _clip(path, seconds=3.0, size='720x1280', rate=30, box='red'):
    """A browser-like portrait recording with a coloured box in the top-left."""
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f'testsrc2=size={size}:rate={rate},drawbox=x=0:y=0:w=160:h=80:color={box}:t=fill',
                    '-t', str(seconds), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', path], check=True)
    return path


def _new_session():
    r = client.post('/api/sessions', json={'subjectID': 'Quinn', 'mass_kg': 81.6, 'height_m': 1.75})
    assert r.status_code == 200, r.text
    return r.json()


def _join(ws, model=MODEL):
    welcome = json.loads(ws.receive_text())
    assert welcome['type'] == 'welcome'
    ws.send_text(json.dumps({'type': 'hello', 'model': model, 'label': welcome['cam'],
                             'settings': {'width': 720, 'height': 1280, 'frameRate': 60}}))
    ws.send_text(json.dumps({'type': 'ping', 't0': time.time()}))
    pong = json.loads(ws.receive_text())
    assert pong['type'] == 'pong'
    ws.send_text(json.dumps({'type': 'clock', 'offset_s': 0.0, 'rtt_ms': 12.0}))
    return welcome


def _wait(sid, key, trial, status='ready', timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        t = {x['name']: x for x in client.get(f'/api/sessions/{sid}?key={key}').json()['trials']}[trial]
        if t['status'] in (status, 'failed'):
            return t
        time.sleep(0.2)
    raise AssertionError(f'{trial} never reached {status}')


def _record(s, ws_by_cam, welcome_by_cam, kind, name=None, offsets=(0.0, 0.12), seconds=3.0):
    """Start, receive the start command on both phones, stop, upload."""
    r = client.post(f"/api/sessions/{s['sid']}/trials",
                    json={'key': s['key'], 'kind': kind, 'name': name})
    assert r.status_code == 200, r.text
    trial = r.json()['name']
    starts = {}
    for cam, ws in ws_by_cam.items():
        m = json.loads(ws.receive_text())
        assert m['type'] == 'start' and m['trial'] == trial
        starts[cam] = m['at']
    assert len(set(starts.values())) == 1, 'every phone gets the same start time'
    r = client.post(f"/api/sessions/{s['sid']}/trials/{trial}/stop", json={'key': s['key']})
    assert r.status_code == 200, r.text
    for ws in ws_by_cam.values():
        assert json.loads(ws.receive_text())['type'] == 'stop'
    with tempfile.TemporaryDirectory() as d:
        for i, (cam, w) in enumerate(welcome_by_cam.items()):
            # Each phone actually started a little late, by a different amount.
            started = starts[cam] + offsets[i]
            path = _clip(os.path.join(d, f'{cam}.mp4'), seconds=seconds)
            with open(path, 'rb') as f:
                r = client.post(f"/api/sessions/{s['sid']}/upload",
                                data={'device_id': w['device_id'], 'token': w['token'],
                                      'trial': trial, 'started_at': started,
                                      'stopped_at': started + seconds},
                                files={'file': (f'{trial}.mp4', f, 'video/mp4')})
            assert r.status_code == 200, r.text
    return _wait(s['sid'], s['key'], trial)


def test_full_session_produces_opencap_layout():
    s = _new_session()
    with client.websocket_connect(f"/ws/{s['sid']}/camera/phoneAAAA0001") as w0, \
         client.websocket_connect(f"/ws/{s['sid']}/camera/phoneBBBB0002") as w1:
        wa, wb = _join(w0), _join(w1)
        wa['device_id'], wb['device_id'] = 'phoneAAAA0001', 'phoneBBBB0002'
        assert (wa['cam'], wb['cam']) == ('Cam0', 'Cam1')
        wss, wel = {'Cam0': w0, 'Cam1': w1}, {'Cam0': wa, 'Cam1': wb}

        # Swings are refused until the cameras are calibrated.
        r = client.post(f"/api/sessions/{s['sid']}/trials",
                        json={'key': s['key'], 'kind': 'dynamic', 'name': 'swing_1'})
        assert r.status_code == 409 and 'Calibrate' in r.json()['detail']

        assert _record(s, wss, wel, 'calibration')['status'] == 'ready'
        assert _record(s, wss, wel, 'neutral')['status'] == 'ready'
        t = _record(s, wss, wel, 'dynamic', 'swing_1')
        assert t['status'] == 'ready', t

    folder = os.path.join(_TMP, 'Data', f"BiostrikeSession_{s['sid']}")
    meta = yaml.safe_load(open(os.path.join(folder, 'sessionMetadata.yaml')))
    assert meta['iphoneModel'] == {'Cam0': MODEL, 'Cam1': MODEL}
    assert meta['mass_kg'] == 81.6 and meta['checkerBoard']['squareSideLength_mm'] == 35.0
    for trial in ('calibration', 'neutral', 'swing_1'):
        for cam in ('Cam0', 'Cam1'):
            p = os.path.join(folder, 'Videos', cam, 'InputMedia', trial, trial + '.mov')
            info = media.probe(p)
            assert (info['display_w'], info['display_h'], info['fps']) == (720, 1280, 60.0), info
    report = json.load(open(os.path.join(folder, 'Videos', 'swing_1_capture.json')))
    # Cam1 started 120 ms after Cam0, so Cam0 is trimmed by 120 ms and Cam1 not
    # at all, and both cover the same shared window.
    assert abs(report['cams']['Cam0']['trim_s'] - 0.12) < 1e-3
    assert abs(report['cams']['Cam1']['trim_s']) < 1e-3
    assert abs(report['duration_s'] - 2.88) < 1e-3


def test_checkerboard_size_reaches_session_metadata():
    r = client.post('/api/sessions', json={'subjectID': 'Q', 'mass_kg': 80, 'height_m': 1.8,
                                           'board_cols': 7, 'board_rows': 5, 'square_mm': 60})
    assert r.status_code == 200, r.text
    folder = os.path.join(_TMP, 'Data', f"BiostrikeSession_{r.json()['sid']}")
    cb = yaml.safe_load(open(os.path.join(folder, 'sessionMetadata.yaml')))['checkerBoard']
    assert (cb['black2BlackCornersWidth_n'], cb['black2BlackCornersHeight_n'],
            cb['squareSideLength_mm']) == (7, 5, 60.0)
    assert client.post('/api/sessions', json={'subjectID': 'Q', 'mass_kg': 80, 'height_m': 1.8,
                                              'square_mm': 2}).status_code == 400


def test_passcode_only_asked_of_internet_visitors():
    body = {'subjectID': 'Q', 'mass_kg': 80, 'height_m': 1.8}
    local = TestClient(server.app, client=('127.0.0.1', 50000))
    old = server.PASSCODE
    server.PASSCODE = 'abc123'
    try:
        assert local.post('/api/sessions', json=body).status_code == 200
        # Through the tunnel: also from 127.0.0.1, but stamped by Cloudflare.
        r = local.post('/api/sessions', json=body, headers={'CF-Connecting-IP': '203.0.113.9'})
        assert r.status_code == 403
        assert client.post('/api/sessions', json=body).status_code == 403
        assert client.post('/api/sessions', json={**body, 'passcode': 'abc123'}).status_code == 200
        assert local.get('/api/config').json()['passcode_required'] is False
        assert client.get('/api/config').json()['passcode_required'] is True
    finally:
        server.PASSCODE = old


def test_tunnel_link_is_read_from_cloudflared_output():
    import run
    lines = ['INF Requesting new quick Tunnel on trycloudflare.com...',
             'INF |  Your quick Tunnel has been created! Visit it at  |',
             'INF |  https://won-door-murray-factors.trycloudflare.com   |']
    assert run.find_tunnel_url(iter(lines)) == 'https://won-door-murray-factors.trycloudflare.com'
    assert run.find_tunnel_url(iter(['nothing here'])) is None


def test_control_actions_need_the_control_key():
    s = _new_session()
    r = client.post(f"/api/sessions/{s['sid']}/trials", json={'key': 'wrong', 'kind': 'calibration'})
    assert r.status_code == 403
    assert client.get(f"/api/sessions/{s['sid']}?key=nope").status_code == 403


def test_upload_rejects_unknown_phone_and_bad_names():
    s = _new_session()
    r = client.post(f"/api/sessions/{s['sid']}/upload",
                    data={'device_id': 'x' * 12, 'token': 't', 'trial': 'calibration',
                          'started_at': 1, 'stopped_at': 2},
                    files={'file': ('a.mp4', b'not a video', 'video/mp4')})
    assert r.status_code == 403
    r = client.post('/api/sessions', json={'subjectID': '../etc', 'mass_kg': 80, 'height_m': 1.8})
    assert r.status_code == 400


def test_landscape_recording_is_refused_with_a_reason():
    with tempfile.TemporaryDirectory() as d:
        src = _clip(os.path.join(d, 'l.mp4'), seconds=1, size='1280x720')
        try:
            media.normalize_video(src, os.path.join(d, 'o.mov'))
        except media.VideoError as e:
            assert 'portrait' in str(e)
        else:
            raise AssertionError('landscape accepted')


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for fn in tests:
        fn(); print(f'✓ {fn.__name__}')
    print(f'{len(tests)} passed')
