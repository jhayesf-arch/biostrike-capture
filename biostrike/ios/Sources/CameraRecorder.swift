import AVFoundation
import CoreMedia

/// Back camera at 1280x720 and 60/120/240 fps, recording frame-exact windows.
///
/// The camera runs continuously. A recording is a window in *server* time:
/// every frame's sensor timestamp is converted to server time (host clock ->
/// Unix time -> + the clock offset the session measured), and only frames
/// with start <= t < stop are written. So every phone starts on the first
/// frame at or after the shared start instant, instead of whenever a timer
/// happened to fire, and the reported start/stop times are the frames' own.
final class CameraRecorder: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()

    /// Called on the main queue with (file, trial, firstFrameServerTime,
    /// endServerTime, droppedFrames) when a window has been written.
    var onFinished: ((URL, String, Double, Double, Int) -> Void)?

    private let queue = DispatchQueue(label: "biostrike.camera")
    private let output = AVCaptureVideoDataOutput()
    private var device: AVCaptureDevice?
    private(set) var fps = 60
    private(set) var fieldOfView: Float = 0

    // Recording state — only touched on `queue`.
    private var trial: String?
    private var startAt = 0.0
    private var stopAt = Double.infinity
    private var hostToServer = 0.0          // add to host-clock seconds to get server time
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var fileURL: URL?
    private var firstFrame: Double?
    private var lastFrame = 0.0
    private var dropped = 0
    private var generation = 0              // identifies the current window for timers

    var settings: [String: Any] {
        ["width": 720, "height": 1280, "frameRate": fps, "mimeType": "video/quicktime (app)"]
    }

    // MARK: Setup

    func configure(fps: Int, completion: @escaping (String?) -> Void) {
        queue.async {
            let error = self.configureOnQueue(fps: fps)
            DispatchQueue.main.async { completion(error) }
        }
    }

    private func configureOnQueue(fps: Int) -> String? {
        if device == nil {
            guard let cam = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                return "No back camera found."
            }
            guard let camInput = try? AVCaptureDeviceInput(device: cam) else {
                return "Camera access is blocked. Allow it in Settings > Biostrike."
            }
            session.beginConfiguration()
            guard session.canAddInput(camInput), session.canAddOutput(output) else {
                session.commitConfiguration()
                return "Could not set up the camera."
            }
            session.addInput(camInput)
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
            // Recording, not live analysis: keep late frames rather than drop them.
            output.alwaysDiscardsLateVideoFrames = false
            output.setSampleBufferDelegate(self, queue: queue)
            session.addOutput(output)
            session.commitConfiguration()
            device = cam
        }
        guard let cam = device else { return "No camera." }
        guard let format = Self.bestFormat(for: cam, fps: fps) else {
            return "This iPhone cannot record 1280x720 at \(fps) fps."
        }
        do {
            try cam.lockForConfiguration()
            cam.activeFormat = format
            let frame = CMTime(value: 1, timescale: CMTimeScale(fps))
            cam.activeVideoMinFrameDuration = frame
            cam.activeVideoMaxFrameDuration = frame
            if cam.focusMode != .locked, cam.isFocusModeSupported(.continuousAutoFocus) {
                cam.focusMode = .continuousAutoFocus
            }
            cam.unlockForConfiguration()
        } catch {
            return "Could not configure the camera: \(error.localizedDescription)"
        }
        // OpenCap records without stabilization: it crops and shifts the image,
        // which would break the lens calibration.
        if let connection = output.connection(with: .video), connection.isVideoStabilizationSupported {
            connection.preferredVideoStabilizationMode = .off
        }
        self.fps = fps
        fieldOfView = format.videoFieldOfView
        if !session.isRunning { session.startRunning() }
        return nil
    }

    /// The 1280x720 format that reaches `fps` with the field of view closest to
    /// the standard 720p video mode, which the stored lens calibrations were
    /// measured on (OpenCap's 240 fps calibration matches its 60 fps one).
    static func bestFormat(for device: AVCaptureDevice, fps: Int) -> AVCaptureDevice.Format? {
        let yuv: Set<FourCharCode> = [kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                      kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        let hd = device.formats.filter { f in
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            return d.width == 1280 && d.height == 720
                && yuv.contains(CMFormatDescriptionGetMediaSubType(f.formatDescription))
        }
        let reaches = { (f: AVCaptureDevice.Format, rate: Double) in
            f.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= rate - 0.5 }
        }
        let candidates = hd.filter { reaches($0, Double(fps)) }
        guard !candidates.isEmpty else { return nil }
        let reference = hd.first { reaches($0, 30) && !reaches($0, 61) }?.videoFieldOfView
            ?? candidates[0].videoFieldOfView
        return candidates.min { abs($0.videoFieldOfView - reference) < abs($1.videoFieldOfView - reference) }
    }

    /// Fix the focus distance so the lens (and its calibration) cannot change
    /// mid-session. Done when calibration starts.
    func lockFocus() {
        queue.async {
            guard let cam = self.device, cam.isFocusModeSupported(.locked) else { return }
            try? cam.lockForConfiguration()
            cam.focusMode = .locked
            cam.unlockForConfiguration()
        }
    }

    func refocus() {
        queue.async {
            guard let cam = self.device, cam.isFocusModeSupported(.continuousAutoFocus) else { return }
            try? cam.lockForConfiguration()
            cam.focusMode = .continuousAutoFocus
            cam.unlockForConfiguration()
        }
    }

    func stopCamera() {
        queue.async { if self.session.isRunning { self.session.stopRunning() } }
    }

    // MARK: Recording windows

    /// Record frames from server time `startAt` until `stop(at:)`.
    /// `clockOffset` is server time minus this phone's Unix time.
    func arm(trial: String, startAt: Double, clockOffset: Double) {
        queue.async {
            self.resetWindow()
            self.generation += 1
            let hostNow = CMClockGetTime(CMClockGetHostTimeClock()).seconds
            self.hostToServer = Date().timeIntervalSince1970 - hostNow + clockOffset
            self.trial = trial
            self.startAt = startAt
            self.stopAt = .infinity
            // Open the file now, during the lead-in, so the first frame is not
            // spent setting up the encoder (at 240 fps that would drop frames).
            _ = self.prepareWriter(trial: trial)
            // Safety net: never record longer than two minutes.
            self.scheduleForcedFinish(after: startAt + 120)
        }
    }

    func stop(at serverTime: Double) {
        queue.async {
            guard self.trial != nil else { return }
            self.stopAt = serverTime
            // If frames stop arriving, finish anyway shortly after the stop time.
            self.scheduleForcedFinish(after: serverTime + 1.0)
        }
    }

    private func scheduleForcedFinish(after serverTime: Double) {
        let hostNow = CMClockGetTime(CMClockGetHostTimeClock()).seconds
        let delay = max(0, serverTime - (hostNow + hostToServer))
        let window = generation
        queue.asyncAfter(deadline: .now() + delay) {
            if self.generation == window, self.trial != nil { self.finish() }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let trial = trial else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let clock = session.synchronizationClock ?? CMClockGetHostTimeClock()
        let hostTime = CMSyncConvertTime(pts, from: clock, to: CMClockGetHostTimeClock()).seconds
        let t = hostTime + hostToServer

        if firstFrame == nil {
            guard t >= startAt else { return }
            guard writer != nil || prepareWriter(trial: trial), let w = writer else { return }
            w.startSession(atSourceTime: pts)
            firstFrame = t
        }
        if t >= stopAt {
            finish()
            return
        }
        if let input = input, input.isReadyForMoreMediaData {
            if !input.append(sampleBuffer) { dropped += 1 }
        } else {
            dropped += 1
        }
        lastFrame = t
    }

    private func prepareWriter(trial: String) -> Bool {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(trial)-\(UUID().uuidString).mov")
        guard let w = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return false }
        let bitrate = fps >= 240 ? 30_000_000 : fps >= 120 ? 20_000_000 : 12_000_000
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 1280,
            AVVideoHeightKey: 720,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoMaxKeyFrameIntervalKey: fps,          // a keyframe each second, as OpenCap
                AVVideoExpectedSourceFrameRateKey: fps,
            ],
        ]
        let inp = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        inp.expectsMediaDataInRealTime = true
        // Sensor frames are landscape; tag them to display upright in portrait,
        // exactly how the iPhone camera (and OpenCap's app) store video.
        inp.transform = CGAffineTransform(rotationAngle: .pi / 2)
        guard w.canAdd(inp) else { return false }
        w.add(inp)
        guard w.startWriting() else { return false }
        writer = w
        input = inp
        fileURL = url
        dropped = 0
        return true
    }

    private func finish() {
        guard let trial = trial else { return }
        self.trial = nil
        guard let w = writer, let inp = input, let url = fileURL, let first = firstFrame else {
            resetWindow()
            return
        }
        let end = lastFrame + 1.0 / Double(fps)
        let droppedFrames = dropped
        inp.markAsFinished()
        w.finishWriting { [weak self] in
            DispatchQueue.main.async {
                if w.status == .completed {
                    self?.onFinished?(url, trial, first, end, droppedFrames)
                }
            }
        }
        writer = nil
        input = nil
        fileURL = nil
        firstFrame = nil
    }

    private func resetWindow() {
        if let w = writer, w.status == .writing { w.cancelWriting() }
        writer = nil
        input = nil
        fileURL = nil
        firstFrame = nil
        trial = nil
        stopAt = .infinity
    }
}
