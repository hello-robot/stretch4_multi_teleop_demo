#!/usr/bin/env python3
"""Dual-hand MediaPipe tracker: publishes the midpoint of both palms as a 6DOF Joy signal."""
import rclpy
from teleop_interfaces.mediapipe_base import MediaPipeBaseNode, palm_center, palm_size, palm_angles
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

class HandsTracker(MediaPipeBaseNode):
    """Tracks both hands via MediaPipe HandLandmarker (near-duplicate of HandTracker)."""

    HAND_LMS = [
        'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
        'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
        'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
        'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
        'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
    ]

    def __init__(self, config_path, node_name='hands_tracker'):
        """Set up left_*/right_* landmark point names and, if model_path is set, the detector."""
        self._point_names = [f"left_{name.lower()}" for name in self.HAND_LMS] + \
                            [f"right_{name.lower()}" for name in self.HAND_LMS]
        super().__init__(node_name, config_path)

        if not self.model_path:
            return

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        """Detect both hands and derive a 6DOF pose from the palm(s) found.

        Returns:
            [x, y, z, roll, pitch, yaw], or None if no hand is detected.
            x/y are the midpoint of the detected palm centers and z the mean of
            each hand's 1 / palm size. roll is the angle of the line between
            the two palm centers ("steering wheel"), or that palm's own roll
            if only one hand is visible. pitch/yaw are the mean of each
            detected palm's angles (see mediapipe_base.palm_angles).
        """
        if not hasattr(self, 'detector'):
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            return None

        hand_found = {'left': False, 'right': False}

        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            label = detection_result.handedness[i][0].category_name.lower() # 'left' or 'right'
            hand_found[label] = True
            for j, lm in enumerate(hand_landmarks):
                l_name = f"{label}_{self.HAND_LMS[j].lower()}"
                self._landmarks[l_name] = [lm.x, lm.y, lm.z]

        h, w = frame.shape[:2]
        aspect = h / w

        # Per-hand pose for each detected hand
        centers, depths, angles = [], [], []
        for side in ['left', 'right']:
            if not hand_found[side]:
                continue
            prefix = f"{side}_"
            centers.append(palm_center(self._landmarks, prefix))
            depths.append(1.0 / max(palm_size(self._landmarks, aspect, prefix), 1e-6))
            angles.append(palm_angles(self._landmarks, aspect, prefix))

        if not centers:
            return None

        x, y, _ = np.mean(centers, axis=0)
        z = float(np.mean(depths))
        _, pitch, yaw = np.mean(angles, axis=0)

        if len(centers) == 2:
            # Order by image x rather than handedness label, so the sign doesn't
            # depend on whether MediaPipe's mirrored-image labels match reality.
            left_c, right_c = sorted(centers, key=lambda c: c[0])
            dx = right_c[0] - left_c[0]
            dy = (right_c[1] - left_c[1]) * aspect
            roll = np.arctan2(dy, dx)
        else:
            roll = angles[0][0]

        return [float(x), float(y), z, float(roll), float(pitch), float(yaw)]

def main(args=None):
    """Entry point: parse --config, construct HandsTracker, and spin until interrupted."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to config file')
    parsed_args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = None
    try:
        node = HandsTracker(parsed_args.config)
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
