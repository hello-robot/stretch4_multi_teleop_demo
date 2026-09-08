#!/usr/bin/env python3
import rclpy
from teleop_interfaces.mediapipe_base import MediaPipeBaseNode
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

class HeadFaceTracker(MediaPipeBaseNode):
    def __init__(self, config_path):
        self._point_names = [f"pt_{i}" for i in range(478)]
        super().__init__('head_face_tracker', config_path)
        
        if not self.model_path:
            return
            
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame):
        if not hasattr(self, 'detector'):
            return None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
            
        face_landmarks = detection_result.face_landmarks[0]
        
        # Update self._landmarks
        for i, lm in enumerate(face_landmarks):
            name = f"pt_{i}"
            coords = [lm.x, lm.y, lm.z]
            self._landmarks[name] = coords
            
            friendly_map = {
                1: 'nose',
                13: 'mouth_top', 14: 'mouth_bottom', 61: 'mouth_left', 291: 'mouth_right',
                159: 'left_eye_top', 145: 'left_eye_bottom',
                133: 'left_eye_inner', 33: 'left_eye_outer',
                386: 'right_eye_top', 374: 'right_eye_bottom',
                362: 'right_eye_inner', 263: 'right_eye_outer',
                52: 'left_eyebrow', 282: 'right_eyebrow',
                468: 'left_iris', 473: 'right_iris'
            }
            if i in friendly_map:
                self._landmarks[friendly_map[i]] = coords
            
        # Estimate 6DOF
        nose = face_landmarks[1]
        x, y, z = nose.x, nose.y, nose.z
        
        le = face_landmarks[33]
        re = face_landmarks[263]
        dx = re.x - le.x
        dy = re.y - le.y
        roll = np.arctan2(dy, dx)
        
        yaw = (nose.x - (le.x + re.x)/2.0) / (re.x - le.x + 1e-6)
        
        top = face_landmarks[10]
        chin = face_landmarks[152]
        dy = chin.y - top.y
        dz = top.z - chin.z
        pitch = np.arctan2(dz, dy)
        
        return [x, y, z, roll, pitch, yaw]

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to config file')
    parsed_args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = None
    try:
        node = HeadFaceTracker(parsed_args.config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
