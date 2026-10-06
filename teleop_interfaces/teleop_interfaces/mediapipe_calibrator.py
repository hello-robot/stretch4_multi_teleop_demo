#!/usr/bin/env python3
"""Interactive workspace calibration for the MediaPipe camera trackers.

Runs one of hand_tracker / hands_tracker / head_face_tracker in the same
OpenCV visualization window the tracker normally shows, plus a control strip
underneath with clickable buttons (and keyboard shortcuts):

    Zero (z)       Use the current pose as workspace_zero; min/max shift with it.
    Calibrate (c)  Guided procedure: hold a neutral pose (-> zero), then move
                   to each axis' min and max in turn. A translucent "ghost"
                   shows which way to move.
    Capture (spc)  Record the current step (the most extreme value reached).
    Skip (s)       Skip the current step, keeping the existing value.
    Save (w)       Write a ROS 2 params file with workspace_zero/min/max.
    Quit (q)

The node runs as '<tracker>_calibrator', so its Joy output isn't routed to any
control scheme. The saved file is keyed by the real tracker's node name, so
it can be passed straight to it:

    ros2 run teleop_interfaces mediapipe_calibrator --tracker hand \\
        --ros-args -p model_path:=/path/to/hand_landmarker.task -p hand_to_track:=right
    ros2 run teleop_interfaces hand_tracker --ros-args \\
        -p model_path:=... --params-file hand_tracker_workspace.yaml
"""
import argparse
import collections
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.parameter import Parameter

from teleop_interfaces.mediapipe_base import aspect_corrected, normalize_6dof
from teleop_interfaces.hand_tracker import HandTracker
from teleop_interfaces.hands_tracker import HandsTracker
from teleop_interfaces.head_face_tracker import HeadFaceTracker

AXIS_NAMES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

# Same Viridis-inspired palette (BGR) as MediaPipeBaseNode._draw_visualization
V_PURPLE = (84, 1, 68)
V_BLUE = (143, 71, 70)
V_GREEN = (89, 205, 109)
V_YELLOW = (37, 231, 253)
V_GRAY = (200, 200, 200)
V_DARK = (20, 20, 20)

# Instructions for each calibration step, as (axis index, side, text). Wording
# is in terms of the image, since the camera may or may not be mirrored.
# Sign conventions (shared by all trackers): x/y grow right/down in the image,
# z grows away from the camera, roll is clockwise-positive in the image, pitch
# is positive when the top (fingers / top of head) moves away from the camera,
# and yaw is positive when the image-right side moves away from the camera.
HAND_STEPS = [
    (0, 'min', "Move your hand toward the LEFT side of the image"),
    (0, 'max', "Move your hand toward the RIGHT side of the image"),
    (1, 'min', "Move your hand toward the TOP of the image"),
    (1, 'max', "Move your hand toward the BOTTOM of the image"),
    (2, 'min', "Move your hand CLOSER to the camera"),
    (2, 'max', "Move your hand FARTHER from the camera"),
    (3, 'min', "Tilt your hand COUNTER-CLOCKWISE (as seen in the image)"),
    (3, 'max', "Tilt your hand CLOCKWISE (as seen in the image)"),
    (4, 'min', "Tip your fingers TOWARD the camera"),
    (4, 'max', "Tip your fingers AWAY from the camera"),
    (5, 'min', "Turn your palm to face the LEFT side of the image"),
    (5, 'max', "Turn your palm to face the RIGHT side of the image"),
]

HANDS_STEPS = [
    (0, 'min', "Move both hands toward the LEFT side of the image"),
    (0, 'max', "Move both hands toward the RIGHT side of the image"),
    (1, 'min', "Move both hands toward the TOP of the image"),
    (1, 'max', "Move both hands toward the BOTTOM of the image"),
    (2, 'min', "Move both hands CLOSER to the camera"),
    (2, 'max', "Move both hands FARTHER from the camera"),
    (3, 'min', "Steer COUNTER-CLOCKWISE (left hand down, right hand up)"),
    (3, 'max', "Steer CLOCKWISE (left hand up, right hand down)"),
    (4, 'min', "Tip both hands' fingers TOWARD the camera"),
    (4, 'max', "Tip both hands' fingers AWAY from the camera"),
    (5, 'min', "Turn both palms to face the LEFT side of the image"),
    (5, 'max', "Turn both palms to face the RIGHT side of the image"),
]

FACE_STEPS = [
    (0, 'min', "Move your head toward the LEFT side of the image"),
    (0, 'max', "Move your head toward the RIGHT side of the image"),
    (1, 'min', "Move your head toward the TOP of the image"),
    (1, 'max', "Move your head toward the BOTTOM of the image"),
    (2, 'min', "Lean CLOSER to the camera"),
    (2, 'max', "Lean AWAY from the camera"),
    (3, 'min', "Tilt your head COUNTER-CLOCKWISE (as seen in the image)"),
    (3, 'max', "Tilt your head CLOCKWISE (as seen in the image)"),
    (4, 'min', "Tilt your head FORWARD (chin down)"),
    (4, 'max', "Tilt your head BACK (chin up)"),
    (5, 'min', "Turn your face toward the LEFT side of the image"),
    (5, 'max', "Turn your face toward the RIGHT side of the image"),
]

# Standard MediaPipe hand skeleton, by landmark name (without left_/right_ prefix)
HAND_CONNECTIONS = [
    ('wrist', 'thumb_cmc'), ('thumb_cmc', 'thumb_mcp'), ('thumb_mcp', 'thumb_ip'), ('thumb_ip', 'thumb_tip'),
    ('wrist', 'index_finger_mcp'), ('index_finger_mcp', 'index_finger_pip'),
    ('index_finger_pip', 'index_finger_dip'), ('index_finger_dip', 'index_finger_tip'),
    ('middle_finger_mcp', 'middle_finger_pip'), ('middle_finger_pip', 'middle_finger_dip'),
    ('middle_finger_dip', 'middle_finger_tip'),
    ('ring_finger_mcp', 'ring_finger_pip'), ('ring_finger_pip', 'ring_finger_dip'),
    ('ring_finger_dip', 'ring_finger_tip'),
    ('wrist', 'pinky_mcp'), ('pinky_mcp', 'pinky_pip'), ('pinky_pip', 'pinky_dip'), ('pinky_dip', 'pinky_tip'),
    ('index_finger_mcp', 'middle_finger_mcp'), ('middle_finger_mcp', 'ring_finger_mcp'),
    ('ring_finger_mcp', 'pinky_mcp'),
]

STRIP_HEIGHT = 120          # Control strip below the camera image, in pixels
ZERO_AVERAGE_FRAMES = 10    # Frames averaged when capturing a zero pose
GHOST_PERIOD_SEC = 2.0      # Duration of one ghost animation loop
GHOST_SHIFT = 0.25          # Ghost translation, as a fraction of the frame
GHOST_ANGLE = np.radians(45)


# --- Ghost geometry ---
#
# Ghost points live in aspect-corrected space (x, y * h/w, z), all in image
# widths, so rotations aren't distorted. Each rotation is written so a
# positive angle moves the ghost the same way a positive tracker value does.

def rotate_roll(p, angle):
    """Rotate Nx3 points in the image plane; positive = clockwise in the image."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([x * c - y * s, x * s + y * c, z], axis=1)


def rotate_pitch(p, angle):
    """Rotate about the image horizontal; positive = top moves away from the camera."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([x, y * c + z * s, -y * s + z * c], axis=1)


def rotate_yaw(p, angle):
    """Rotate about the image vertical; positive = image-right side moves away."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([x * c - z * s, y, x * s + z * c], axis=1)


def ghost_pose(points, axis, direction, t, aspect):
    """Move ghost points part of the way (t in [0, 1]) toward a step's target.

    Args:
        points: Nx3 aspect-corrected zero-pose points.
        axis: 0-5 (x, y, z, roll, pitch, yaw).
        direction: -1 for a 'min' step, +1 for 'max'.
        t: animation progress in [0, 1].
        aspect: image height / width.
    Returns:
        (moved Nx3 points, translation offset applied to the centroid).
    """
    center = points.mean(axis=0)
    rel = points - center
    offset = np.zeros(3)
    if axis == 0:
        offset[0] = direction * GHOST_SHIFT * t
    elif axis == 1:
        offset[1] = direction * GHOST_SHIFT * aspect * t
    elif axis == 2:
        # Closer (min) looks bigger, farther (max) looks smaller
        scale = 1.0 + 0.5 * t if direction < 0 else 1.0 - 0.4 * t
        rel = rel * scale
    elif axis == 3:
        rel = rotate_roll(rel, direction * GHOST_ANGLE * t)
    elif axis == 4:
        rel = rotate_pitch(rel, direction * GHOST_ANGLE * t)
    elif axis == 5:
        rel = rotate_yaw(rel, direction * GHOST_ANGLE * t)
    return center + offset + rel, offset


class CalibrationMixin:
    """Adds the calibration UI to a MediaPipeBaseNode tracker subclass.

    Use as the first base, e.g. `class HandCalibrator(CalibrationMixin,
    HandTracker)`. Subclasses set TRACKER_NAME (the real tracker's node name)
    and STEPS (the per-axis instructions).
    """

    TRACKER_NAME = None
    STEPS = []

    def __init__(self, config_path, output_path=None, margin=0.1):
        """Set up calibration state, then construct the tracker as '<TRACKER_NAME>_calibrator'.

        Args:
            config_path: tracker virtual-sensor config (same as the tracker's -c).
            output_path: where Save writes the params file.
            margin: fraction to pull each captured min/max back toward zero,
                so the full [-1, 1] range is reachable without straining.
        """
        # All UI state is set before the tracker constructor runs, because the
        # tracker starts capturing (and calling _show_frame) during __init__.
        self.output_path = output_path or f"{self.TRACKER_NAME}_workspace.yaml"
        self.margin = margin
        self.quit_requested = False

        self._recent_states = collections.deque(maxlen=ZERO_AVERAGE_FRAMES)
        self._ghost_points = None       # {landmark name: aspect-corrected xyz} at zero time
        self._step = None               # None = idle, 0 = neutral pose, 1.. = STEPS[step - 1]
        self._extreme = None            # Most extreme value seen in the current step
        self._status = "Press Calibrate (c) for guided setup, or Zero (z) to re-center."
        self._ui_buttons = []              # [(label, (x0, y0, x1, y1), callback, enabled)]
        self._mouse_ready = False
        self._last_aspect = None        # Image height / width, set on the first frame

        super().__init__(config_path, node_name=f"{self.TRACKER_NAME}_calibrator")

        if not self.get_parameter('visualize').value:
            self.get_logger().info("Calibrator needs the visualization window; enabling 'visualize'.")
            self.set_parameters([Parameter('visualize', Parameter.Type.BOOL, True)])
        if list(self.get_parameter('workspace_dims').value) != [1.0, 1.0, 1.0]:
            self.get_logger().warn(
                "'workspace_dims' is set, which overrides workspace_min/max for x/y/z. "
                "The saved file won't include it; unset it on the tracker to use this calibration.")

    # --- Workspace updates ---

    def _set_workspace(self, zero, ws_min, ws_max):
        """Write workspace_zero/min/max back to this node's parameters."""
        self.set_parameters([
            Parameter('workspace_zero', Parameter.Type.DOUBLE_ARRAY, [float(v) for v in zero]),
            Parameter('workspace_min', Parameter.Type.DOUBLE_ARRAY, [float(v) for v in ws_min]),
            Parameter('workspace_max', Parameter.Type.DOUBLE_ARRAY, [float(v) for v in ws_max]),
        ])

    def set_zero(self):
        """Use the recent average pose as the new zero, shifting min/max by the same offset."""
        if not self._recent_states or self._last_aspect is None:
            self._status = "No pose detected - can't set zero."
            return False
        new_zero = np.mean(self._recent_states, axis=0)
        zero, ws_min, ws_max = self._get_workspace()
        delta = new_zero - np.array(zero)
        self._set_workspace(new_zero, np.array(ws_min) + delta, np.array(ws_max) + delta)

        # Remember this pose for the ghost
        self._ghost_points = {name: aspect_corrected(self._landmarks[name], self._last_aspect)
                              for name in self._point_names if name in self._landmarks}

        self._status = "Zero set."
        self._log_workspace()
        return True

    def _capture_extreme(self):
        """Store the current step's extreme value (pulled in by margin) as its min or max."""
        axis, side, _ = self.STEPS[self._step - 1]
        zero, ws_min, ws_max = self._get_workspace()
        name = f"{AXIS_NAMES[axis]} {side}"

        moved_past_zero = (self._extreme is not None and
                           (self._extreme < zero[axis] if side == 'min' else self._extreme > zero[axis]))
        if not moved_past_zero:
            self._status = f"No movement past zero for {name}; kept previous value."
            return

        value = zero[axis] + (self._extreme - zero[axis]) * (1.0 - self.margin)
        if side == 'min':
            ws_min[axis] = value
        else:
            ws_max[axis] = value
        self._set_workspace(zero, ws_min, ws_max)
        self._status = f"Captured {name} = {value:.3f}"

    def _log_workspace(self):
        """Log the current workspace as a params-file YAML snippet."""
        self.get_logger().info(f"Current workspace ({self.TRACKER_NAME}):\n{self._workspace_yaml()}")

    def _workspace_yaml(self):
        """Format the current workspace as a ROS 2 params file for the real tracker node."""
        zero, ws_min, ws_max = self._get_workspace()
        params = {self.TRACKER_NAME: {'ros__parameters': {
            'workspace_zero': [round(float(v), 4) for v in zero],
            'workspace_min': [round(float(v), 4) for v in ws_min],
            'workspace_max': [round(float(v), 4) for v in ws_max],
        }}}
        return yaml.safe_dump(params, default_flow_style=None, sort_keys=False)

    def save(self):
        """Write the current workspace to output_path."""
        with open(self.output_path, 'w') as f:
            f.write(self._workspace_yaml())
        self._status = f"Saved to {self.output_path}"
        self._log_workspace()
        self.get_logger().info(f"Saved workspace to {self.output_path}")

    # --- Calibration procedure ---

    def start_calibration(self):
        """Begin the guided procedure at the neutral-pose step."""
        self._step = 0
        self._extreme = None
        self._status = "Calibration started."

    def capture(self):
        """Finish the current step (zero or extreme) and advance."""
        if self._step is None:
            return
        if self._step == 0:
            if not self.set_zero():
                return
        else:
            self._capture_extreme()
        self._advance()

    def skip(self):
        """Advance past the current step without changing anything."""
        if self._step is None:
            return
        if self._step == 0 and self._ghost_points is None:
            self._status = "Skipped zero - no ghost available until Zero is set."
        else:
            self._status = "Skipped."
        self._advance()

    def _advance(self):
        """Move to the next step, or finish the procedure."""
        self._step += 1
        self._extreme = None
        if self._step > len(self.STEPS):
            self._step = None
            self._status = f"Calibration done - Save (w) to write {self.output_path}"
            self._log_workspace()

    def _update_extreme(self, state):
        """Track the most extreme value reached in the current step's direction."""
        if not self._step:
            return
        axis, side, _ = self.STEPS[self._step - 1]
        val = state[axis]
        if self._extreme is None:
            self._extreme = val
        elif side == 'min':
            self._extreme = min(self._extreme, val)
        else:
            self._extreme = max(self._extreme, val)

    # --- Display ---

    def _show_frame(self, frame):
        """Overrides MediaPipeBaseNode._show_frame: add the ghost + control strip, handle input."""
        h, w = frame.shape[:2]
        self._last_aspect = h / w

        state = self._last_raw_state
        if state is not None:
            self._recent_states.append(list(state))
            self._update_extreme(state)

        if self._step and self._ghost_points:
            self._draw_ghost(frame)

        strip = np.full((STRIP_HEIGHT, w, 3), V_DARK, dtype=np.uint8)
        self._draw_strip(strip, state)
        canvas = np.vstack([frame, strip])

        window = f"{self.get_name()} Visualization"
        cv2.imshow(window, canvas)
        if not self._mouse_ready:
            cv2.setMouseCallback(window, self._on_mouse, param=h)
            self._mouse_ready = True
        self._handle_key(cv2.waitKey(1) & 0xFF)

    def _draw_ghost(self, frame):
        """Draw the zero pose as a translucent ghost moving toward the current step's target."""
        h, w = frame.shape[:2]
        axis, side, _ = self.STEPS[self._step - 1]
        direction = -1 if side == 'min' else 1
        t = (time.time() % GHOST_PERIOD_SEC) / GHOST_PERIOD_SEC

        names = list(self._ghost_points.keys())
        points = np.array([self._ghost_points[n] for n in names])
        moved, _ = ghost_pose(points, axis, direction, t, self._last_aspect)
        # Aspect-corrected y is in image widths, so both coordinates scale by w
        pixels = {n: (int(p[0] * w), int(p[1] * w)) for n, p in zip(names, moved)}

        overlay = frame.copy()
        if self._point_names[0].startswith('pt_'):
            # Face: sparse point cloud
            for n, px in pixels.items():
                if int(n[3:]) % 6 == 0:
                    cv2.circle(overlay, px, 2, V_YELLOW, -1)
        else:
            # Hand(s): skeleton, for each prefix present
            for prefix in ('', 'left_', 'right_'):
                for a, b in HAND_CONNECTIONS:
                    if prefix + a in pixels and prefix + b in pixels:
                        cv2.line(overlay, pixels[prefix + a], pixels[prefix + b], V_YELLOW, 3)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Arrow for translation steps, from the zero pose to the full target offset
        if axis in (0, 1):
            center = points.mean(axis=0)
            start = (int(center[0] * w), int(center[1] * w))
            _, full_offset = ghost_pose(points, axis, direction, 1.0, self._last_aspect)
            end = (int((center[0] + full_offset[0]) * w), int((center[1] + full_offset[1]) * w))
            cv2.arrowedLine(frame, start, end, V_GREEN, 2, tipLength=0.2)

    def _draw_strip(self, strip, state):
        """Draw instructions, live values, status, and buttons into the control strip."""
        w = strip.shape[1]

        # Line 1: instruction for the current step
        if self._step is None:
            instruction = "Idle"
            color = V_GRAY
        elif self._step == 0:
            instruction = "Hold a comfortable NEUTRAL pose, then Capture (space)"
            color = V_YELLOW
        else:
            instruction = self.STEPS[self._step - 1][2] + ", then Capture (space)"
            color = V_YELLOW
        cv2.putText(strip, instruction, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Line 2: step progress and live values
        if self._step:
            axis, side, _ = self.STEPS[self._step - 1]
            progress = f"Step {self._step}/{len(self.STEPS)}  {AXIS_NAMES[axis]} {side}"
            if state is None:
                values = "  (nothing detected)"
            else:
                zero, ws_min, ws_max = self._get_workspace()
                norm = normalize_6dof(state, zero, ws_min, ws_max)[axis]
                extreme = f"{self._extreme:.3f}" if self._extreme is not None else "-"
                values = f"  raw {state[axis]:.3f}  extreme {extreme}  norm {norm:+.2f}"
            cv2.putText(strip, progress + values, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, V_GRAY, 1, cv2.LINE_AA)
        elif self._step == 0:
            cv2.putText(strip, f"Step 0/{len(self.STEPS)}  zero", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, V_GRAY, 1, cv2.LINE_AA)

        # Line 3: last action / status
        cv2.putText(strip, self._status, (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, V_GREEN, 1, cv2.LINE_AA)

        # Button row
        calibrating = self._step is not None
        buttons = [
            ("Zero (z)", self.set_zero, True),
            ("Calibrate (c)", self.start_calibration, True),
            ("Capture (spc)", self.capture, calibrating),
            ("Skip (s)", self.skip, calibrating),
            ("Save (w)", self.save, True),
            ("Quit (q)", self.request_quit, True),
        ]
        pad = 6
        btn_w = (w - pad * (len(buttons) + 1)) // len(buttons)
        y0, y1 = 78, STRIP_HEIGHT - 8
        self._ui_buttons = []
        for i, (label, callback, enabled) in enumerate(buttons):
            x0 = pad + i * (btn_w + pad)
            x1 = x0 + btn_w
            cv2.rectangle(strip, (x0, y0), (x1, y1), V_BLUE if enabled else V_PURPLE, -1)
            cv2.rectangle(strip, (x0, y0), (x1, y1), V_YELLOW if enabled else V_GRAY, 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(strip, label, (x0 + (btn_w - tw) // 2, (y0 + y1 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, V_GRAY if enabled else (120, 120, 120), 1, cv2.LINE_AA)
            self._ui_buttons.append((label, (x0, y0, x1, y1), callback, enabled))

    # --- Input ---

    def _on_mouse(self, event, x, y, flags, frame_height):
        """OpenCV mouse callback: run a button's action on left click."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        strip_y = y - frame_height
        for _, (x0, y0, x1, y1), callback, enabled in self._ui_buttons:
            if enabled and x0 <= x <= x1 and y0 <= strip_y <= y1:
                callback()
                return

    def _handle_key(self, key):
        """Keyboard shortcuts matching the button labels."""
        actions = {
            ord('z'): self.set_zero,
            ord('c'): self.start_calibration,
            ord(' '): self.capture,
            ord('s'): self.skip,
            ord('w'): self.save,
            ord('q'): self.request_quit,
        }
        if key in actions:
            actions[key]()

    def request_quit(self):
        """Ask main() to stop spinning (logs the workspace so nothing is lost)."""
        self._log_workspace()
        self.quit_requested = True


class HandCalibrator(CalibrationMixin, HandTracker):
    TRACKER_NAME = 'hand_tracker'
    STEPS = HAND_STEPS


class HandsCalibrator(CalibrationMixin, HandsTracker):
    TRACKER_NAME = 'hands_tracker'
    STEPS = HANDS_STEPS


class HeadFaceCalibrator(CalibrationMixin, HeadFaceTracker):
    TRACKER_NAME = 'head_face_tracker'
    STEPS = FACE_STEPS


CALIBRATORS = {
    'hand': HandCalibrator,
    'hands': HandsCalibrator,
    'head_face': HeadFaceCalibrator,
}


def main(args=None):
    """Entry point: parse --tracker/--config/--output/--margin, then run the calibrator."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--tracker', required=True, choices=sorted(CALIBRATORS),
                        help='Which tracker to calibrate')
    parser.add_argument('-c', '--config', help="Tracker config file (same as the tracker's -c)")
    parser.add_argument('-o', '--output', help='Params file to write (default: ./<tracker>_workspace.yaml)')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Fraction to pull captured min/max back toward zero (default: 0.1)')
    parsed_args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = None
    try:
        node = CALIBRATORS[parsed_args.tracker](parsed_args.config, parsed_args.output, parsed_args.margin)
        while rclpy.ok() and not node.quit_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
