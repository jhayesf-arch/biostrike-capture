"""Start a capture session in one command: server + public link + checks.

    python -m biostrike.capture_server.run

Runs the capture server, opens a Cloudflare quick tunnel (the free https link
phones need for camera access), waits until that link actually answers from
the internet, then opens the control page on this computer. It keeps the Mac
awake while running and stops everything on Ctrl+C.

The passcode it prints is only asked of browsers coming in over the internet;
the control page on this computer never needs it.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
TUNNEL_RE = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')


def find_tunnel_url(lines, timeout_s: float = 60.0) -> str | None:
    """Read cloudflared's log lines until it prints its quick-tunnel link."""
    end = time.time() + timeout_s
    for line in lines:
        m = TUNNEL_RE.search(line)
        if m and 'api.trycloudflare.com' not in m.group(0):
            return m.group(0)
        if time.time() > end:
            break
    return None


def wait_reachable(url: str, timeout_s: float = 90.0) -> bool:
    """Poll the public link until it serves our config endpoint."""
    ctx = ssl.create_default_context()
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(url + '/api/config', timeout=5, context=ctx) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _start_server(port: int) -> threading.Thread:
    import uvicorn
    sys.path.insert(0, HERE)
    import server  # noqa: F401  (configured through the environment below)
    t = threading.Thread(target=uvicorn.run, args=(server.app,),
                         kwargs={'host': '127.0.0.1', 'port': port, 'log_level': 'warning'},
                         daemon=True)
    t.start()
    return t


def main(argv=None) -> int:
    port = int(os.environ.get('CAPTURE_PORT', '8000'))
    exe = shutil.which('cloudflared')
    if not exe:
        print('cloudflared is not installed. Run:  brew install cloudflared')
        return 1
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        print('ffmpeg is not installed. Run:  brew install ffmpeg')
        return 1

    print('Starting the public link (Cloudflare)…')
    tunnel = subprocess.Popen([exe, 'tunnel', '--no-autoupdate', '--url', f'http://localhost:{port}'],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    url = find_tunnel_url(iter(tunnel.stdout.readline, ''))
    if not url:
        tunnel.terminate()
        print('Cloudflare did not give a link. Check the internet connection and try again.')
        return 1
    # Keep draining cloudflared's output so its pipe never fills and blocks it.
    threading.Thread(target=lambda: [None for _ in iter(tunnel.stdout.readline, '')],
                     daemon=True).start()

    passcode = os.environ.get('CAPTURE_PASSCODE') or secrets.token_hex(3)
    os.environ['CAPTURE_PUBLIC_URL'] = url
    os.environ['CAPTURE_PASSCODE'] = passcode
    _start_server(port)

    awake = None
    if sys.platform == 'darwin' and shutil.which('caffeinate'):
        awake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])

    print('Waiting for the link to come online (can take up to a minute)…')
    if wait_reachable(url, float(os.environ.get('CAPTURE_REACH_TIMEOUT', '90'))):
        print('Link is live.')
    else:
        print('The link is not answering yet from this computer. Phones may still reach it;\n'
              'if they show "Error 1033", stop this (Ctrl+C) and run it again.')

    local = f'http://localhost:{port}'
    print('\n' + '=' * 64)
    print(f'  Control page (this computer):  {local}')
    print(f'  Phones join via the QR code:   {url}')
    print(f'  Passcode for other computers:  {passcode}')
    print('  Keep this window open. Ctrl+C stops everything.')
    print('=' * 64 + '\n')
    webbrowser.open(local)
    try:
        while tunnel.poll() is None:
            time.sleep(1)
        print('The Cloudflare link stopped. Run this command again for a new one.')
        return 1
    except KeyboardInterrupt:
        print('\nStopping.')
        return 0
    finally:
        tunnel.terminate()
        if awake:
            awake.terminate()


if __name__ == '__main__':
    sys.exit(main())
