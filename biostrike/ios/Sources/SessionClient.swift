import Foundation

/// Talks to the capture server exactly like the browser camera page does:
/// joins over a WebSocket, measures the clock offset, follows start/stop
/// commands and uploads each recording.
final class SessionClient: NSObject, ObservableObject {
    @Published var status = "Connecting…"
    @Published var cam = ""
    @Published var clockText = "syncing…"
    @Published var recording = false
    @Published var modelWarning = ""
    @Published var fpsText = ""

    let link: JoinLink
    let recorder: CameraRecorder
    private let deviceId: String
    private var token: String?
    private var offset: Double?                     // server time − phone time, seconds
    private var pings: [(rtt: Double, offset: Double)] = []
    private var ws: URLSessionWebSocketTask?
    private var clockTimer: Timer?
    private var leaving = false
    private static let pingCount = 12

    init(link: JoinLink, recorder: CameraRecorder) {
        self.link = link
        self.recorder = recorder
        let key = "bs_dev_" + link.sid
        if let saved = UserDefaults.standard.string(forKey: key) {
            deviceId = saved
        } else {
            deviceId = UUID().uuidString
            UserDefaults.standard.set(deviceId, forKey: key)
        }
        super.init()
        recorder.onFinished = { [weak self] url, trial, start, end, dropped in
            self?.upload(file: url, trial: trial, startedAt: start, stoppedAt: end, dropped: dropped)
        }
        fpsText = "\(recorder.fps) fps"
    }

    private static func now() -> Double { Date().timeIntervalSince1970 }

    // MARK: Connection

    func connect() {
        guard !leaving, let url = link.webSocketURL(deviceId: deviceId) else { return }
        let task = URLSession.shared.webSocketTask(with: url)
        ws = task
        task.resume()
        receive(on: task)
    }

    func leave() {
        leaving = true
        clockTimer?.invalidate()
        ws?.cancel(with: .goingAway, reason: nil)
        recorder.stopCamera()
    }

    private func receive(on task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self = self, task === self.ws else { return }
            switch result {
            case .success(let message):
                if case .string(let text) = message,
                   let data = text.data(using: .utf8),
                   let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                    DispatchQueue.main.async { self.handle(obj) }
                }
                self.receive(on: task)
            case .failure:
                DispatchQueue.main.async {
                    guard !self.leaving else { return }
                    self.status = "Reconnecting…"
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { self.connect() }
                }
            }
        }
    }

    private func send(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let text = String(data: data, encoding: .utf8) else { return }
        ws?.send(.string(text)) { _ in }
    }

    private func sendState(_ state: String) { send(["type": "state", "state": state]) }

    // MARK: Messages

    private func handle(_ m: [String: Any]) {
        switch m["type"] as? String {
        case "welcome":
            token = m["token"] as? String
            cam = m["cam"] as? String ?? ""
            send(["type": "hello", "model": DeviceInfo.modelIdentifier, "label": cam,
                  "settings": recorder.settings])
            syncClock()
            clockTimer?.invalidate()
            clockTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
                self?.syncClock()
            }
            status = "Ready"
        case "hello_ack":
            let supported = m["model_supported"] as? Bool ?? false
            modelWarning = supported ? "" :
                "No lens calibration for this iPhone (\(DeviceInfo.modelIdentifier)). It cannot be used as a camera."
        case "pong":
            guard let t0 = m["t0"] as? Double, let ts = m["ts"] as? Double else { return }
            let t1 = Self.now()
            pings.append((rtt: t1 - t0, offset: ts - (t0 + t1) / 2))
            if pings.count == Self.pingCount, let best = pings.min(by: { $0.rtt < $1.rtt }) {
                offset = best.offset
                clockText = "±\(Int((best.rtt * 1000 / 2).rounded())) ms"
                send(["type": "clock", "offset_s": best.offset, "rtt_ms": best.rtt * 1000])
            }
        case "start":
            guard let offset = offset, let trial = m["trial"] as? String,
                  let at = m["at"] as? Double else { return }
            let fps = (m["fps"] as? Int) ?? 60
            if m["kind"] as? String == "calibration" { recorder.lockFocus() }
            status = "Starting \(trial)…"
            let begin = { [weak self] in
                guard let self = self else { return }
                self.recorder.arm(trial: trial, startAt: at, clockOffset: offset)
                self.recording = true
                self.status = "● Recording \(trial)"
                self.sendState("recording")
            }
            if fps != recorder.fps {
                recorder.configure(fps: fps) { [weak self] error in
                    guard let self = self else { return }
                    if let error = error {
                        self.status = error
                        self.sendState("error")
                        return
                    }
                    self.fpsText = "\(fps) fps"
                    self.send(["type": "hello", "model": DeviceInfo.modelIdentifier, "label": self.cam,
                               "settings": self.recorder.settings])
                    begin()
                }
            } else {
                begin()
            }
        case "stop":
            guard let at = m["at"] as? Double else { return }
            recorder.stop(at: at)
        default:
            break
        }
    }

    private func syncClock() {
        pings.removeAll()
        for i in 0..<Self.pingCount {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i) * 0.12) { [weak self] in
                self?.send(["type": "ping", "t0": Self.now()])
            }
        }
    }

    // MARK: Upload

    private func upload(file: URL, trial: String, startedAt: Double, stoppedAt: Double, dropped: Int) {
        recording = false
        guard let token = token else { status = "Upload failed: not joined"; return }
        status = "Uploading \(trial)…" + (dropped > 0 ? " (\(dropped) frames dropped)" : "")
        sendState("uploading")
        let boundary = "biostrike-\(UUID().uuidString)"
        let body = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).body")
        do {
            try Self.writeMultipart(to: body, boundary: boundary, fields: [
                "device_id": deviceId, "token": token, "trial": trial,
                "started_at": String(startedAt), "stopped_at": String(stoppedAt),
            ], fileField: "file", fileURL: file, fileName: "\(trial).mov", mimeType: "video/quicktime")
        } catch {
            status = "Upload failed: \(error.localizedDescription)"
            return
        }
        var request = URLRequest(url: link.uploadURL)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        URLSession.shared.uploadTask(with: request, fromFile: body) { [weak self] data, response, error in
            try? FileManager.default.removeItem(at: body)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            DispatchQueue.main.async {
                guard let self = self else { return }
                if code >= 200 && code < 300 {
                    try? FileManager.default.removeItem(at: file)
                    self.status = "Ready"
                    self.sendState("idle")
                } else {
                    let detail = data.flatMap { String(data: $0, encoding: .utf8) } ?? error?.localizedDescription ?? ""
                    self.status = "Upload failed (\(code)): \(detail.prefix(120))"
                    self.sendState("error")
                }
            }
        }.resume()
    }

    /// Streams the video into a multipart body on disk, so large 240 fps
    /// recordings are never held in memory twice.
    private static func writeMultipart(to url: URL, boundary: String, fields: [String: String],
                                       fileField: String, fileURL: URL, fileName: String,
                                       mimeType: String) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let out = try FileHandle(forWritingTo: url)
        defer { try? out.close() }
        func write(_ s: String) { out.write(s.data(using: .utf8)!) }
        for (name, value) in fields {
            write("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n")
        }
        write("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(fileField)\"; filename=\"\(fileName)\"\r\n")
        write("Content-Type: \(mimeType)\r\n\r\n")
        let src = try FileHandle(forReadingFrom: fileURL)
        defer { try? src.close() }
        while true {
            let chunk = src.readData(ofLength: 1 << 20)
            if chunk.isEmpty { break }
            out.write(chunk)
        }
        write("\r\n--\(boundary)--\r\n")
    }
}
