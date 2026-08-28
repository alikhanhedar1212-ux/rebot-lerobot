
#!/usr/bin/env python3
"""Real-time visualization for all connected RealSense cameras.

The lerobot env's OpenCV is a headless build (no GTK/Qt), so cv2.imshow is
unavailable. Instead we capture the three cameras, composite them into one grid
(3 color on top, 3 colorized depth on bottom), and stream it as MJPEG over HTTP.

Open http://localhost:8765/ in a browser to watch the live feed. Ctrl+C to stop.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pyrealsense2 as rs

# All three cameras currently connected. Reorder / edit serials as needed.
CAMERAS = [
    {"serial": "260322276818", "label": "D405 #1"},
    {"serial": "260322275993", "label": "D405 #2"},
    {"serial": "408322071106", "label": "D435i"},
]

W, H, FPS = 640, 480, 30
MAX_DEPTH_M = 2.0          # clip depth colorization to this range
JPEG_QUALITY = 80
PORT = 8765

# Latest composite JPEG shared between the capture thread and the HTTP server.
_lock = threading.Lock()
_latest_jpeg = None


def colorize_depth(depth_frame):
    """Convert a z16 depth frame (mm) to a JET-colormapped BGR image."""
    d = np.asanyarray(depth_frame.get_data(), dtype=np.float32)
    d = np.clip(d, 0.0, MAX_DEPTH_M * 1000.0)
    d8 = cv2.convertScaleAbs(d, alpha=255.0 / (MAX_DEPTH_M * 1000.0))
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)


def _label(img, text):
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def capture_loop(pipes):
    global _latest_jpeg
    while True:
        colors, depths = [], []
        for cam, pipe in pipes:
            try:
                frames = pipe.wait_for_frames(timeout_ms=5000)
                color = np.asanyarray(frames.get_color_frame().get_data())
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                _label(color, f"{cam['label']}  color")
                colors.append(color)
                depth = colorize_depth(frames.get_depth_frame())
                _label(depth, f"{cam['label']}  depth")
                depths.append(depth)
            except Exception as e:  # noqa: BLE001
                print(f"[frame] {cam['label']} error: {e}")

        if colors and depths:
            grid = np.vstack([np.hstack(colors), np.hstack(depths)])
            ok, jpg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with _lock:
                    _latest_jpeg = jpg.tobytes()
        time.sleep(0.033)  # ~30 fps


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_html()
        elif self.path == "/stream":
            self._serve_mjpeg()
        else:
            self.send_error(404)

    def _serve_html(self):
        body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RealSense 3-camera view</title></head>
<body style="background:#111;color:#eee;font-family:sans-serif">
<h3>RealSense 3-camera view — top: color, bottom: depth (D405#1 / D405#2 / D435i)</h3>
<img src="/stream" style="max-width:100%">
</body></html>"""
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with _lock:
                    jpg = _latest_jpeg
                if jpg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):  # quiet
        pass


def main():
    pipes = []
    for cam in CAMERAS:
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(cam["serial"])
            cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
            cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
            pipe.start(cfg)
            pipes.append((cam, pipe))
            print(f"[open] {cam['label']} ({cam['serial']})")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {cam['label']} ({cam['serial']}): {e}")

    if not pipes:
        print("No cameras opened.")
        return

    print("[warmup] discarding first 30 frames...")
    for _ in range(30):
        for _, p in pipes:
            try:
                p.wait_for_frames(timeout_ms=5000)
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=capture_loop, args=(pipes,), daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[run] open http://localhost:{PORT}/ in a browser. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for _, p in pipes:
            p.stop()
        print("[done]")


if __name__ == "__main__":
    main()
