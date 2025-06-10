import cv2
import numpy as np
import subprocess
import time
import os
import sys
from collections import deque
from datetime import datetime
from openvino.runtime import Core
from utils import draw_enhanced_detections, VehicleTracker

class VehicleCounter:
    def __init__(self, xml_path, bin_path, video_path):
        self.core = Core()
        model = self.core.read_model(model=xml_path)
        self.compiled = self.core.compile_model(model, "CPU")
        self.input_layer = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)
        self.shape = self.input_layer.shape
        self.cap = None
        self.video_path = video_path
        self.initialize_video_capture()
        self.vehicle_tracker = VehicleTracker()
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)
        self.start_time = time.time()
        self.inference_times = deque(maxlen=100)
        self.ffmpeg_process = None

    def initialize_video_capture(self):
        backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
        for backend in backends:
            self.cap = cv2.VideoCapture(self.video_path, backend)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return
                self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
        raise RuntimeError("Failed to open video.")

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.shape[3], self.shape[2]))
        transposed = resized.transpose((2, 0, 1))
        return np.expand_dims(transposed, axis=0)

    def detect_vehicles(self, frame):
        input_tensor = self.preprocess(frame)
        result = self.compiled([input_tensor])[self.output_layer]
        detections = []
        for obj in result[0][0]:
            confidence = obj[2]
            if confidence > 0.5:
                x1 = int(obj[3] * frame.shape[1])
                y1 = int(obj[4] * frame.shape[0])
                x2 = int(obj[5] * frame.shape[1])
                y2 = int(obj[6] * frame.shape[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                if x2 > x1 and y2 > y1:
                    if area > 5000:
                        label = "Large Vehicle"
                    elif area > 2000:
                        label = "Car"
                    else:
                        label = "Small Vehicle"
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'type': label,
                        'area': area
                    })
        self.inference_times.append((time.time() - self.start_time) * 1000)
        return detections

    def calculate_fps(self):
        now = time.time()
        self.fps_counter.append(now)
        if len(self.fps_counter) < 2:
            return 0
        return (len(self.fps_counter) - 1) / (self.fps_counter[-1] - self.fps_counter[0])

    def get_statistics(self):
        return {
            'fps': self.calculate_fps(),
            'inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'uptime': time.time() - self.start_time,
            'frame_count': self.frame_count,
            'vehicle_stats': self.vehicle_tracker.get_stats()
        }

    def setup_ffmpeg_streaming(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        host = os.getenv('MEDIAMTX_HOST', 'localhost')
        port = os.getenv('MEDIAMTX_PORT', '8554')
        path = os.getenv('STREAM_PATH', 'live')
        url = f"rtsp://{host}:{port}/{path}"
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", "30", "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", "30", "-keyint_min", "30", "-bf", "0",
            "-maxrate", "2000k", "-bufsize", "4000k", "-f", "rtsp", url
        ]
        self.ffmpeg_process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=10**8
        )
        time.sleep(2)
        if self.ffmpeg_process.poll() is not None:
            self.cleanup()
            raise RuntimeError("FFmpeg failed to start")

    def run(self):
        try:
            self.setup_ffmpeg_streaming()
            while True:
                ret, frame = self.cap.read()
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if not ret:
                    self.cap.release()
                    time.sleep(0.1)
                    self.initialize_video_capture()
                    continue
                detections = self.detect_vehicles(frame)
                self.vehicle_tracker.update(detections)
                stats = self.get_statistics()
                output_frame = draw_enhanced_detections(frame, detections, stats)
                if self.ffmpeg_process.poll() is not None:
                    self.cleanup()
                    self.setup_ffmpeg_streaming()
                try:
                    self.ffmpeg_process.stdin.write(output_frame.tobytes())
                    self.ffmpeg_process.stdin.flush()
                except Exception:
                    self.cleanup()
                    self.setup_ffmpeg_streaming()
                self.frame_count += 1
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except Exception:
                self.ffmpeg_process.kill()

def main():
    xml_path = "model/vehicle-detection-adas-0002.xml"
    bin_path = "model/vehicle-detection-adas-0002.bin"
    video_path = "input/vid.mp4"
    if not all(os.path.exists(p) for p in [xml_path, bin_path, video_path]):
        print("Model or video file not found.")
        sys.exit(1)
    counter = VehicleCounter(xml_path, bin_path, video_path)
    counter.run()

if __name__ == "__main__":
    main()
