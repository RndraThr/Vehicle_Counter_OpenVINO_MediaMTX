import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict

class VehicleTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.total_count = 0
        self.vehicle_types = defaultdict(int)

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def register(self, detection):
        self.objects[self.next_id] = {
            'centroid': self._get_centroid(detection['bbox']),
            'type': detection['type'],
            'confidence': detection['confidence']
        }
        self.disappeared[self.next_id] = 0
        self.total_count += 1
        self.vehicle_types[detection['type']] += 1
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return

        input_centroids = [self._get_centroid(det['bbox']) for det in detections]
        if len(self.objects) == 0:
            for det in detections:
                self.register(det)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[i]['centroid'] for i in object_ids]
            used = set()
            for i, object_id in enumerate(object_ids):
                min_dist = float("inf")
                min_idx = -1
                for j, c in enumerate(input_centroids):
                    if j in used:
                        continue
                    dist = np.linalg.norm(np.array(object_centroids[i]) - np.array(c))
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = j
                if min_dist < 50 and min_idx != -1:
                    self.objects[object_id]['centroid'] = input_centroids[min_idx]
                    self.disappeared[object_id] = 0
                    used.add(min_idx)
                else:
                    self.disappeared[object_id] += 1
            for i, det in enumerate(detections):
                if i not in used:
                    self.register(det)
            for object_id in list(self.disappeared.keys()):
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

    def get_stats(self):
        return {
            'total_count': self.total_count,
            'current_count': len(self.objects),
            'types': dict(self.vehicle_types)
        }

def get_vehicle_color(vehicle_type):
    return {
        'Car': (0, 255, 0),
        'Large Vehicle': (0, 165, 255),
        'Small Vehicle': (255, 255, 0),
        'default': (255, 255, 255)
    }.get(vehicle_type, (255, 255, 255))

def create_gradient_background(w, h):
    g = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        alpha = i / h
        g[i, :] = [int(20 * (1 - alpha)), int(30 * (1 - alpha)), int(40 * (1 - alpha))]
    return g

def draw_enhanced_detections(frame, detections, stats):
    overlay = frame.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = get_vehicle_color(det['type'])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{det['type']} {det['confidence']:.2f}"
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    width = overlay.shape[1]
    stats_text = f"Live: {stats['vehicle_stats']['current_count']} Total: {stats['vehicle_stats']['total_count']} FPS: {stats['fps']:.1f}"
    cv2.putText(overlay, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return overlay