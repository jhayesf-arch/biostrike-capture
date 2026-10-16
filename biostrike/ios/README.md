# Biostrike Capture — iPhone app

A native camera for the capture server. It joins sessions the same way the
browser page does (scan the control page's QR code), but records like OpenCap's
app:

- **1280×720 at 60, 120 or 240 fps.** The slow-motion modes, which Safari cannot
  reach. Pick the rate on the control page before pressing Record.
- **Frame-exact start and stop.** Every frame's sensor timestamp is converted to
  server time, and recording starts on the first frame at or after the shared
  start instant. Phones line up to within the clock-sync accuracy, not to
  whenever a timer fired.
- **No stabilization; focus locks at calibration.** Either would shift the image
  and invalidate the lens calibration. Use *Refocus* only if you will calibrate
  again.
- **The phone model is detected automatically**, e.g. `iPhone14,7`. The app warns
  if there is no stored lens calibration for it.

## Build it for free (Apple ID, no paid developer account)

You need a Mac with Xcode, the iPhone and a USB cable.

1. Install **Xcode** from the Mac App Store (free). Open it once so it can
   finish installing its components.
2. In Terminal:
   ```
   brew install xcodegen
   ```
   ```
   cd ~/biostrike-capture
   ```
   ```
   git pull
   ```
   ```
   cd biostrike/ios
   ```
   ```
   xcodegen
   ```
   ```
   open BiostrikeCapture.xcodeproj
   ```
3. In Xcode: **Xcode → Settings → Accounts → +** and sign in with your Apple ID.
4. Click the blue **BiostrikeCapture** project at the top left, then the
   **BiostrikeCapture** target → **Signing & Capabilities**:
   - **Team:** choose *Your Name (Personal Team)*.
   - **Bundle Identifier:** change `CHANGE-ME` to something unique to you, e.g.
     `com.jetthayes.biostrikecapture`.
5. Plug in the iPhone and tap **Trust** on it. On the iPhone, turn on
   **Settings → Privacy & Security → Developer Mode** (it restarts).
6. In Xcode's top bar, pick your iPhone as the destination and press **▶︎** (Cmd+R).
7. The first launch is blocked until you trust yourself as a developer. On the
   iPhone: **Settings → General → VPN & Device Management →** your Apple ID →
   **Trust**. Then open Biostrike.

Repeat steps 5–7 for each phone.

**Free-account limits:** the app stops opening after **7 days**; plug the phone
in and press ▶︎ again to renew it. A free account can only put it on a few of
your own devices. Installing on coaches' phones without a cable (TestFlight)
needs the $99/year Apple Developer Program.

## Use it

1. On the Mac: `python -m biostrike.capture_server.run`. The control page opens.
2. On each iPhone: open **Biostrike → Scan QR code** and point it at the control
   page. The camera view appears with *Cam0/Cam1…*, the frame rate and clock
   accuracy.
3. On the control page, pick **60 / 120 / 240 fps**, then Calibrate → Neutral →
   Record. App phones and browser phones can be mixed, but all cameras are
   filed at the slowest one's rate.

240 fps needs bright light; indoors, 120 fps is usually the better trade.

## Status

Written against Apple's AVFoundation APIs but **not yet compiled**: the
environment it was written in has no Xcode. If the first build shows errors,
send them over. The server side it talks to is covered by
`capture_server/test_capture_server.py`, including a recording in exactly this
app's file layout at 120 fps.
