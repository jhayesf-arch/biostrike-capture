import AVFoundation
import SwiftUI

struct ContentView: View {
    @State private var linkText = UserDefaults.standard.string(forKey: "lastJoinLink") ?? ""
    @State private var scanning = false
    @State private var client: SessionClient?
    @State private var error = ""
    @State private var joining = false

    var body: some View {
        if let client = client {
            SessionView(client: client) {
                client.leave()
                self.client = nil
            }
        } else {
            joinView
        }
    }

    private var joinView: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Biostrike Capture").font(.largeTitle.bold())
            Text("Scan the QR code on the control page, or paste its link.")
                .foregroundStyle(.secondary)
            Button {
                scanning = true
            } label: {
                Label("Scan QR code", systemImage: "qrcode.viewfinder")
                    .frame(maxWidth: .infinity).padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            TextField("https://….trycloudflare.com/join/…", text: $linkText)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.URL)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button(joining ? "Starting camera…" : "Join") { join(linkText) }
                .buttonStyle(.bordered)
                .disabled(joining || linkText.isEmpty)
            if !error.isEmpty { Text(error).foregroundStyle(.red) }
            Spacer()
            Text("This iPhone: \(DeviceInfo.modelIdentifier)").font(.footnote).foregroundStyle(.secondary)
        }
        .padding()
        .sheet(isPresented: $scanning) {
            QRScannerView { code in
                scanning = false
                linkText = code
                join(code)
            }
            .ignoresSafeArea()
        }
    }

    private func join(_ text: String) {
        error = ""
        guard let link = JoinLink(text) else {
            error = "That is not a Biostrike join link (https://…/join/…)."
            return
        }
        UserDefaults.standard.set(link.text, forKey: "lastJoinLink")
        joining = true
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                guard granted else {
                    joining = false
                    error = "Camera access is off. Turn it on in Settings > Biostrike."
                    return
                }
                let recorder = CameraRecorder()
                recorder.configure(fps: 60) { problem in
                    joining = false
                    if let problem = problem {
                        error = problem
                        return
                    }
                    let c = SessionClient(link: link, recorder: recorder)
                    c.connect()
                    client = c
                }
            }
        }
    }
}

struct SessionView: View {
    @ObservedObject var client: SessionClient
    let onLeave: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: client.recorder.session)
                .ignoresSafeArea()
                .overlay(
                    RoundedRectangle(cornerRadius: 0)
                        .stroke(client.recording ? Color.red : Color.clear, lineWidth: 8)
                        .ignoresSafeArea()
                )
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(client.cam.isEmpty ? "Joining…" : client.cam).font(.title2.bold())
                    Spacer()
                    Button("Leave", action: onLeave).buttonStyle(.bordered)
                }
                Text(client.status).font(.headline)
                    .foregroundStyle(client.recording ? Color.red : Color.white)
                Text("\(client.fpsText) · clock \(client.clockText) · \(DeviceInfo.modelIdentifier)")
                    .font(.footnote)
                if !client.modelWarning.isEmpty {
                    Text(client.modelWarning).font(.footnote).foregroundStyle(.yellow)
                }
                Spacer()
                HStack {
                    Button("Refocus") { client.recorder.refocus() }
                        .buttonStyle(.bordered)
                    Text("Focus locks at calibration. Refocusing means calibrating again.")
                        .font(.caption2)
                }
            }
            .foregroundStyle(.white)
            .padding()
            .background(LinearGradient(colors: [.black.opacity(0.55), .clear, .clear, .black.opacity(0.55)],
                                       startPoint: .top, endPoint: .bottom))
        }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
    }
}

/// Live view of the recording camera.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}
