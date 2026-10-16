import Foundation

/// The join link the control page shows as a QR code:
/// https://<host>/join/<session id>
struct JoinLink {
    let base: URL       // https://<host>[:port]
    let sid: String

    init?(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), url.scheme == "https", let host = url.host else {
            return nil
        }
        let parts = url.pathComponents          // ["/", "join", "<sid>"]
        guard parts.count >= 3, parts[parts.count - 2] == "join" else { return nil }
        let sid = parts[parts.count - 1]
        guard sid.range(of: "^[A-Za-z0-9_-]{6,64}$", options: .regularExpression) != nil else {
            return nil
        }
        var c = URLComponents()
        c.scheme = "https"
        c.host = host
        c.port = url.port
        guard let base = c.url else { return nil }
        self.base = base
        self.sid = sid
    }

    var text: String { base.absoluteString + "/join/" + sid }

    func webSocketURL(deviceId: String) -> URL? {
        var c = URLComponents(url: base, resolvingAgainstBaseURL: false)
        c?.scheme = "wss"
        c?.path = "/ws/\(sid)/camera/\(deviceId)"
        return c?.url
    }

    var uploadURL: URL { base.appendingPathComponent("api/sessions/\(sid)/upload") }
}

enum DeviceInfo {
    /// Apple's hardware identifier, e.g. "iPhone14,7" for an iPhone 14. The
    /// server picks the matching lens calibration from it.
    static var modelIdentifier: String {
        if let sim = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
            return sim
        }
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
}
