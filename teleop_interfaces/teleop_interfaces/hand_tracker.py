#!/usr/bin/env python3
"""Single-hand MediaPipe tracker: publishes one hand's wrist pose as a 6DOF Joy signal."""
import rclpy
from teleop_interfaces.mediapipe_base import MediaPipeBaseNode
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

class HandTracker(MediaPipeBaseNode):
    """Tracks a single hand (left or right, selectable) via MediaPipe HandLandmarker."""

    def __init__(self, config_path):
        """Set up the 21 hand landmark point names and, if model_path is set, the detector."""
        hand_lms = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        self._point_names = [name.lower() for name in hand_lms]
        super().__init__('hand_tracker', config_path)
        
        self.declare_parameter('hand_to_track', 'left') # 'left' or 'right'
        
        if not self.model_path:
            return
            
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2) # Still detect 2 but pick one
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        """Detect hands, select the configured hand_to_track, and derive a 6DOF pose.

        Returns:
            [x, y, z, roll=0, pitch, yaw=0] from the wrist landmark, or None if
            no matching hand is detected. pitch is the image-plane angle of the
            wrist-to-middle-finger-MCP vector, not a true 3D pitch.
        """
        if not hasattr(self, 'detector'):
            return None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.hand_landmarks:
            return None
            
        target_hand = self.get_parameter('hand_to_track').value.lower()
        
        selected_landmarks = None
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            label = detection_result.handedness[i][0].category_name.lower()
            if label == target_hand:
                selected_landmarks = hand_landmarks
                break
        
        if not selected_landmarks:
            return None
            
        for j, lm in enumerate(selected_landmarks):
            l_name = self._point_names[j]
            self._landmarks[l_name] = [lm.x, lm.y, lm.z]

        wrist = self._landmarks['wrist']
        x, y, z = wrist
        
        m_mcp = self._landmarks['middle_finger_mcp']
        dx = m_mcp[0] - wrist[0]
        dy = m_mcp[1] - wrist[1]
        pitch = np.arctan2(dy, dx)
        
        roll, yaw = 0.0, 0.0
            
        return [x, y, z, roll, pitch, yaw]

def main(args=None):
    """Entry point: parse --config, construct HandTracker, and spin until interrupted."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to config file')
    parsed_args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = None
    try:
        node = HandTracker(parsed_args.config)
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
