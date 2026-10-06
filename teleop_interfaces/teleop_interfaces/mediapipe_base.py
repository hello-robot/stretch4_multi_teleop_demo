"""Shared base node for MediaPipe camera trackers, plus a minimal numpy-only CvBridge.

Provides MediaPipeBaseNode (webcam/topic capture, 6DOF workspace normalization,
config-driven "virtual" axis/button distances, and OpenCV visualization) used by
hand_tracker, hands_tracker, and head_face_tracker.
"""
# Replaced cv_bridge with a pure Python/NumPy implementation to support NumPy 2.x
import rclpy
from multi_teleop.base import InputInterfaceNode
from sensor_msgs.msg import Image, Joy
import mediapipe as mp
import cv2
import numpy as np
import yaml
import json
import threading
import time
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.parameter import Parameter
from abc import abstractmethod

class CvBridge:
    """Minimal drop-in replacement for cv_bridge.CvBridge (bgr8-family decode only)."""

    def imgmsg_to_cv2(self, msg, desired_encoding="bgr8"):
        """Convert a sensor_msgs/Image to a bgr8 numpy array.

        Args:
            msg: sensor_msgs/Image with encoding in {rgb8, bgr8, rgba8, bgra8,
                mono8, mono16/16UC1}.
            desired_encoding: only "bgr8" is supported; anything else raises.
        Returns:
            HxWx3 uint8 numpy array in BGR order.
        """
        if desired_encoding != "bgr8":
            raise NotImplementedError(f"Encoding '{desired_encoding}' is not supported.")
        
        # Map ROS encoding to numpy parameters
        if msg.encoding == 'rgb8':
            channels = 3
            dtype = np.uint8
        elif msg.encoding == 'bgr8':
            channels = 3
            dtype = np.uint8
        elif msg.encoding == 'rgba8':
            channels = 4
            dtype = np.uint8
        elif msg.encoding == 'bgra8':
            channels = 4
            dtype = np.uint8
        elif msg.encoding == 'mono8':
            channels = 1
            dtype = np.uint8
        elif msg.encoding in ('mono16', '16UC1'):
            channels = 1
            dtype = np.uint16
        else:
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")

        # Convert the raw buffer to a numpy array
        try:
            arr = np.frombuffer(msg.data, dtype=dtype)
        except (TypeError, ValueError):
            arr = np.asarray(msg.data, dtype=dtype)

        # Reshape considering potential padding / step sizes
        itemsize = np.dtype(dtype).itemsize
        expected_step = msg.width * channels * itemsize
        
        if msg.step > expected_step:
            shape = (msg.height, msg.width, channels) if channels > 1 else (msg.height, msg.width)
            strides = (msg.step, channels * itemsize, itemsize) if channels > 1 else (msg.step, itemsize)
            arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        else:
            expected_size = msg.height * msg.width * channels
            arr = arr[:expected_size]
            if channels > 1:
                arr = arr.reshape((msg.height, msg.width, channels))
            else:
                arr = arr.reshape((msg.height, msg.width))

        # Convert to BGR format for OpenCV / MediaPipe
        if msg.encoding == 'rgb8':
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            return arr.copy()
        elif msg.encoding == 'mono8':
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif msg.encoding == 'rgba8':
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif msg.encoding == 'bgra8':
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif msg.encoding in ('mono16', '16UC1'):
            arr_8bit = (arr / 256).astype(np.uint8)
            return cv2.cvtColor(arr_8bit, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Cannot convert encoding {msg.encoding} to bgr8")


# --- Pose helpers shared by the trackers ---
#
# MediaPipe landmark x is normalized by image width and y by image height, so
# vectors built from raw landmarks are distorted on non-square frames. The
# helpers below first rescale y by the aspect ratio (h / w) so x, y, and z
# (which MediaPipe already scales roughly like x) all share "image widths" as
# their unit, then measure angles and sizes in that space.
#
# Depth: MediaPipe hand z is relative to the wrist (and face z to the head
# center), so no single landmark's z tracks distance to the camera. The
# trackers instead use 1 / (apparent size), which grows roughly linearly as the
# hand/face moves away from the camera.

PALM_POINTS = ['wrist', 'index_finger_mcp', 'middle_finger_mcp', 'ring_finger_mcp', 'pinky_mcp']


def aspect_corrected(landmark, aspect):
    """Return a normalized [x, y, z] landmark rescaled so all axes are in image widths.

    Args:
        landmark: [x, y, z] as produced by MediaPipe (x, y in [0, 1]).
        aspect: image height / image width.
    """
    return np.array([landmark[0], landmark[1] * aspect, landmark[2]])


def palm_center(landmarks, prefix=''):
    """Mean of the wrist and the four finger MCPs, in raw normalized coordinates."""
    return np.mean([landmarks[prefix + name] for name in PALM_POINTS], axis=0)


def palm_size(landmarks, aspect, prefix=''):
    """Mean in-plane wrist-to-MCP distance, in image widths (larger = closer)."""
    wrist = aspect_corrected(landmarks[prefix + 'wrist'], aspect)
    dists = [np.linalg.norm((aspect_corrected(landmarks[prefix + name], aspect) - wrist)[:2])
             for name in PALM_POINTS[1:]]
    return float(np.mean(dists))


def palm_angles(landmarks, aspect, prefix=''):
    """Estimate (roll, pitch, yaw) in radians from the palm plane.

    Uses the same conventions as head_face_tracker, all 0 for an upright hand
    with the palm facing the camera:
      roll:  in-plane tilt of the wrist->middle-MCP ("up") vector.
      pitch: that up vector tipping toward/away from the camera (fingers
             away from the camera is positive, like the face tracker's
             head-tilted-back).
      yaw:   depth difference across the palm (index MCP <-> pinky MCP),
             hand-right side minus hand-left side, like the face tracker's
             right-eye minus left-eye. The "hand-right" side is found from the
             up vector rather than the handedness label, so it works for
             either hand and stays correct as the hand rolls.
    """
    wrist = aspect_corrected(landmarks[prefix + 'wrist'], aspect)
    index = aspect_corrected(landmarks[prefix + 'index_finger_mcp'], aspect)
    middle = aspect_corrected(landmarks[prefix + 'middle_finger_mcp'], aspect)
    pinky = aspect_corrected(landmarks[prefix + 'pinky_mcp'], aspect)

    up = middle - wrist
    roll = np.arctan2(up[0], -up[1])
    pitch = np.arctan2(up[2], np.hypot(up[0], up[1]))

    across = pinky - index
    # In-plane direction pointing to the hand's right (image-right for an upright hand)
    hand_right = np.array([-up[1], up[0]])
    if np.dot(across[:2], hand_right) < 0:
        across = -across
    yaw = np.arctan2(across[2], np.hypot(across[0], across[1]))

    return float(roll), float(pitch), float(yaw)


def normalize_6dof(state, zero, ws_min, ws_max):
    """Map a raw 6DOF pose into [-1, 1] per axis, asymmetrically around `zero`.

    Values between zero and ws_max map to [0, 1]; values between ws_min and
    zero map to [-1, 0]. Results are clamped to [-1, 1].
    """
    normalized = []
    for val, z_val, min_val, max_val in zip(state, zero, ws_min, ws_max):
        if val < z_val:
            denom = z_val - min_val
        else:
            denom = max_val - z_val
        if abs(denom) < 1e-6:
            denom = 1e-6
        normalized.append(max(-1.0, min(1.0, (val - z_val) / denom)))
    return normalized


class MediaPipeBaseNode(InputInterfaceNode):
    """Base class for camera-driven trackers: capture, normalize to 6DOF, publish Joy.

    Subclasses set `self._point_names` before calling super().__init__() and
    implement `process_frame()` to return a 6DOF pose or None.
    """

    def __init__(self, node_name, config_path=None, **kwargs):
        """Load optional virtual-item config, declare parameters, and start capture.

        Args:
            node_name: ROS node name, forwarded to InputInterfaceNode.
            config_path: optional YAML path with a 'virtual_sensors' list.
        """
        # 6DOF base axes are always present: x, y, z, roll, pitch, yaw
        self._axes = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        self._buttons = []
        self._last_axes = [0.0 for n in self._axes]
        self._last_buttons = [0.0 for n in self._buttons]
        self._virtual_config = [] # List of {name, p1, p2, type, threshold/max_dist}
        # True once this Node (and its axis_names/button_names parameters)
        # exist, so add_virtual_item knows whether it can update those
        # declared parameters in place (it's also called before super().__init__()
        # below, while loading the initial config, when there is no Node yet).
        self._ros_ready = False

        # Load config if provided
        if config_path:
            self._load_config(config_path)

        super().__init__(node_name, self._axes, self._buttons, **kwargs)
        self._ros_ready = True

        self.declare_parameter('use_webcam', True)
        self.declare_parameter('webcam_id', 0)
        self.declare_parameter('image_topic', '/image_raw')
        # Workspace, per axis [x, y, z, roll, pitch, yaw]. x/y are normalized
        # image coordinates, z is 1 / apparent size (see pose helpers above),
        # and angles are radians. Z defaults were measured on a 640x480 webcam
        # (close ~4.5, comfortable ~7, far ~12) - run mediapipe_calibrator to
        # tune them for a given user/camera.
        half_pi = float(np.pi / 2)
        self.declare_parameter('workspace_zero', [0.5, 0.5, 7.0, 0.0, 0.0, 0.0])
        self.declare_parameter('workspace_dims', [1.0, 1.0, 1.0])
        self.declare_parameter('workspace_min', [0.0, 0.0, 4.5, -half_pi, -half_pi, -half_pi])
        self.declare_parameter('workspace_max', [1.0, 1.0, 12.0, half_pi, half_pi, half_pi])
        self.declare_parameter('model_path', '') # Required
        self.declare_parameter('visualize', True)
        # Flip each frame horizontally before tracking, so the image (and x,
        # roll, yaw) behave like a mirror. This also makes MediaPipe's hand
        # labels match the user's real hands, since it assumes mirrored input.
        self.declare_parameter('mirror', True)
        
        # Point names should be set by subclass BEFORE calling super().__init__
        point_names = getattr(self, '_point_names', [])
        self.declare_parameter('tracked_points', value=point_names, 
                               descriptor=ParameterDescriptor(read_only=True))
        
        self.model_path = self.get_parameter('model_path').value
        if not self.model_path:
            self.get_logger().error("REQUIRED PARAMETER MISSING: 'model_path'. Please provide the path to the MediaPipe .task file.")
            # We don't raise Exception here to allow rclpy to handle the node state if needed, 
            # but subclasses should check this.
        
        self._bridge = CvBridge()

        
        # Internal state for landmarks: {name: [x, y, z]}
        self._landmarks = {}
        # Most recent raw (un-normalized) 6DOF pose, or None if nothing tracked
        self._last_raw_state = None
        self._last_time = self.get_clock().now()
        self._fps = 0.0
        
        # Threading for latest-frame-only processing
        self._latest_msg = None
        self._msg_lock = threading.Lock()
        self._stop_event = threading.Event()
        
            
        self.add_on_set_parameters_callback(self._on_params_changed)
        
        if self.get_parameter('use_webcam').value:
            self._cap = cv2.VideoCapture(self.get_parameter('webcam_id').value)
            self.create_timer(0.033, self._timer_callback)
        else:
            from rclpy.qos import qos_profile_sensor_data
            self._sub = self.create_subscription(Image, self.get_parameter('image_topic').value, self._image_callback, qos_profile_sensor_data)
            self._proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._proc_thread.start()
        

    def _load_config(self, path):
        """Load 'virtual_sensors' entries from a YAML file, if present, via add_virtual_item."""
        if path is None:
            return
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                if 'virtual_sensors' in config:
                    for axis in config['virtual_sensors']:
                        self.add_virtual_item(axis)
        except Exception as e:
            print(f"WARNING: Failed to load config from {path}: {e}")

    def _on_params_changed(self, params):
        """ROS parameter callback: add a virtual item from an 'add_virtual_item_json' param."""
        for param in params:
            if param.name == 'add_virtual_item_json' and param.type_ == rclpy.Parameter.Type.STRING:
                try:
                    item = json.loads(param.value)
                    self.add_virtual_item(item)
                except Exception as e:
                    self.get_logger().error(f"Failed to add virtual item from JSON: {e}")
        return SetParametersResult(successful=True)

    def add_virtual_item(self, item):
        """Register a landmark-distance-derived virtual axis or button.

        If called after construction (e.g. from the 'add_virtual_item_json'
        parameter callback), also updates the declared 'axis_names'/
        'button_names' ROS parameters so tools like interface_monitor.py see
        the new name instead of a stale list.

        Args:
            item: dict with name, p1, p2, type ('axis'|'button'), and
                max_dist/min_dist (axis) or threshold (button).
        """
        name = item.get('name')
        if not name: return

        item_type = item.get('type')
        if item_type == 'axis':
            if name not in self._axes:
                self._axes.append(name)
                self._last_axes.append(0.0) # Initial value for distance-based axis
                self._virtual_config.append(item)
                self._sync_names_param('axis_names', self._axes)
        elif item_type == 'button':
            if name not in self._buttons:
                self._buttons.append(name)
                self._last_buttons.append(0)
                self._virtual_config.append(item)
                self._sync_names_param('button_names', self._buttons)
        else:
            self._warn_or_print(
                f"add_virtual_item: unrecognized item type {item_type!r} for "
                f"'{name}'; ignoring.")

    def _sync_names_param(self, param_name, names):
        """Update a declared axis_names/button_names parameter to match `names`.

        No-op if called before this node's parameters exist yet (during the
        initial config load in __init__, super().__init__() hasn't declared
        them - they'll already reflect `names` once it does).
        """
        if not self._ros_ready:
            return
        try:
            self.set_parameters([Parameter(param_name, Parameter.Type.STRING_ARRAY, list(names))])
        except Exception as e:
            self._warn_or_print(f"Failed to update '{param_name}' parameter: {e}")

    def _warn_or_print(self, msg):
        """Log a warning via the ROS logger if available, else print (pre-construction)."""
        if self._ros_ready:
            self.get_logger().warn(msg)
        else:
            print(f"WARNING: {msg}")


    @abstractmethod
    def process_frame(self, frame):
        """Should update self._landmarks and return 6DOF [x, y, z, r, p, y] or None."""
        pass

    def update_and_publish(self):
        """Timer callback (webcam mode): grab one frame and process/publish it."""
        if self.get_parameter('use_webcam').value:
            ret, frame = self._cap.read()
            if ret:
                self._update_fps()
                self._handle_frame(frame)

    def _timer_callback(self):
        # We now use update_and_publish for the timer
        self.update_and_publish()

    def _image_callback(self, msg):
        """Image subscription callback (topic mode): stash the latest message only."""
        with self._msg_lock:
            self._latest_msg = msg

    def _processing_loop(self):
        """Background-thread loop (topic mode): process the latest frame as it arrives."""
        while rclpy.ok() and not self._stop_event.is_set():
            msg = None
            with self._msg_lock:
                if self._latest_msg is not None:
                    msg = self._latest_msg
                    self._latest_msg = None # Clear so we wait for a new one
            
            if msg is None:
                time.sleep(0.001)
                continue
                
            # Latency check for logging purposes
            now = self.get_clock().now()
            msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
            latency = (now - msg_time).nanoseconds / 1e9
            if latency > 1.0:
                self.get_logger().warn(f"Processing delayed frame (latency: {latency:.3f}s)", throttle_duration_sec=2.0)

            try:
                frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
                self._update_fps()
                self._handle_frame(frame)
            except Exception as e:
                self.get_logger().error(f"Processing error: {e}")

    def _update_fps(self):
        """Update the exponential-moving-average FPS estimate used in visualization."""
        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds / 1e9
        if dt > 0:
            # 10-sample rolling average (approx)
            self._fps = self._fps * 0.9 + (1.0 / dt) * 0.1
        self._last_time = now

    def _get_workspace(self):
        """Read the workspace parameters and resolve them to 6DOF lists.

        Returns:
            (zero, ws_min, ws_max), each a 6-element list.
        """
        zero = list(self.get_parameter('workspace_zero').value)
        dims = list(self.get_parameter('workspace_dims').value)
        ws_min = list(self.get_parameter('workspace_min').value)
        ws_max = list(self.get_parameter('workspace_max').value)

        # Position-only (3-element) values get default orientation entries
        half_pi = float(np.pi / 2)
        if len(zero) == 3:
            zero += [0.0, 0.0, 0.0]
        if len(ws_min) == 3:
            ws_min += [-half_pi, -half_pi, -half_pi]
        if len(ws_max) == 3:
            ws_max += [half_pi, half_pi, half_pi]

        # Backwards compatibility with workspace_dims
        if dims != [1.0, 1.0, 1.0]:
            for i in range(3):
                ws_min[i] = zero[i] - dims[i] / 2.0
                ws_max[i] = zero[i] + dims[i] / 2.0

        return zero, ws_min, ws_max

    def _handle_frame(self, frame):
        """Run process_frame, normalize the 6DOF pose + virtual items, and publish Joy.

        Shows the raw frame (if visualize) when process_frame returns None.
        """
        if self.get_parameter('mirror').value:
            frame = cv2.flip(frame, 1)
        state_6dof = self.process_frame(frame)
        self._last_raw_state = state_6dof
        if state_6dof is None:
            if self.get_parameter('visualize').value:
                self._show_frame(frame)
            return

        zero, ws_min, ws_max = self._get_workspace()
        axes_values = normalize_6dof(state_6dof, zero, ws_min, ws_max)
        button_values = []
        
        # Process virtual items
        v_axes = []
        v_buttons = []
        v_viz_data = [] # Store data for visualization
        
        for item in self._virtual_config:
            p1_coords = self._landmarks.get(item['p1'])
            p2_coords = self._landmarks.get(item['p2'])
            
            dist = 0.0
            if p1_coords is not None and p2_coords is not None:
                dist = np.linalg.norm(np.array(p1_coords) - np.array(p2_coords))
            
            val = 0.0
            pressed = False
            if item['type'] == 'axis':
                max_d = item.get('max_dist', 1.0)
                min_d = item.get('min_dist', 0.0)
                denom = max_d - min_d
                if abs(denom) < 1e-6: denom = 1e-6
                
                val = 2.0 * (dist - min_d) / denom - 1.0
                val = max(-1.0, min(1.0, val))
                v_axes.append(val)
            else:
                thresh = item.get('threshold', 0.1)
                pressed = dist < thresh
                v_buttons.append(1 if pressed else 0)
                
            v_viz_data.append({
                'item': item,
                'p1': p1_coords,
                'p2': p2_coords,
                'val': val,
                'pressed': pressed
            })
                
        self.publish_input(axes_values + v_axes, button_values + v_buttons)

        # Visualization
        if self.get_parameter('visualize').value:
            self._draw_visualization(frame, state_6dof, zero, ws_min, ws_max, v_viz_data)

    def _draw_visualization(self, frame, state, zero, ws_min, ws_max, v_viz_data):
        """Render the debug overlay (workspace box, 6DOF panel, virtual items) via OpenCV."""
        h, w, _ = frame.shape
        
        # Viridis-inspired Palette (BGR)
        V_PURPLE = (84, 1, 68)
        V_BLUE = (143, 71, 70)
        V_GREEN = (89, 205, 109)
        V_YELLOW = (37, 231, 253)
        
        # 1. Draw Workspace
        x_min = int(ws_min[0] * w)
        y_min = int(ws_min[1] * h)
        x_max = int(ws_max[0] * w)
        y_max = int(ws_max[1] * h)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), V_BLUE, 2)
        # Label above the box's top-left corner when there's room. Otherwise (e.g.
        # the default full-frame workspace) put it inside the bottom-right corner,
        # clear of the 6DOF panel and virtual-item overlay on the left.
        if y_min >= 20 and x_min >= 0:
            label_pos = (x_min, y_min - 10)
        else:
            label_pos = (min(x_max, w) - 80, min(y_max, h) - 8)
        cv2.putText(frame, "WORKSPACE", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, V_BLUE, 1)
        
        # 1.5 Draw FPS
        cv2.putText(frame, f"FPS: {self._fps:.1f}", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, V_YELLOW, 2)
        
        # 1.6 Draw 6DOF Pose Panel
        overlay = frame.copy()
        panel_w = 200
        panel_h = 145
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), V_BLUE, 1)
        
        norm_vals = normalize_6dof(state, zero, ws_min, ws_max)

        cv2.putText(frame, "6DOF POSE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, V_YELLOW, 1, cv2.LINE_AA)
        color_lbl = (200, 200, 200)
        cv2.putText(frame, f"X:     {norm_vals[0]:.2f} ({state[0]:.3f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_lbl, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Y:     {norm_vals[1]:.2f} ({state[1]:.3f})", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_lbl, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Z:     {norm_vals[2]:.2f} ({state[2]:.3f})", (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_lbl, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Roll:  {norm_vals[3]:.2f} ({state[3]:.3f})", (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_lbl, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Pitch: {norm_vals[4]:.2f} ({state[4]:.3f})", (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_lbl, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Yaw:   {norm_vals[5]:.2f} ({state[5]:.3f})", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_lbl, 1, cv2.LINE_AA)
        
        # 2. Draw Target
        px = int(state[0] * w)
        py = int(state[1] * h)
        inside = (x_min <= px <= x_max) and (y_min <= py <= y_max)
        target_color = V_GREEN if inside else V_YELLOW
        cv2.circle(frame, (px, py), 8, target_color, -1)
        cv2.putText(frame, "TARGET", (px+12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, target_color, 1)
        
        # 3. Draw Virtual Items on frame
        for data in v_viz_data:
            item = data['item']
            p1, p2 = data['p1'], data['p2']
            if p1 is None or p2 is None: continue
            
            pt1 = (int(p1[0]*w), int(p1[1]*h))
            pt2 = (int(p2[0]*w), int(p2[1]*h))
            
            if item['type'] == 'axis':
                cv2.line(frame, pt1, pt2, V_GREEN, 1)
                mid_pt = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
                cv2.putText(frame, item['name'], mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, V_GREEN, 1)
            else:
                if data['pressed']:
                    mid_pt = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
                    cv2.circle(frame, mid_pt, 12, V_YELLOW, -1)
                    cv2.putText(frame, item['name'], (mid_pt[0]+15, mid_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, V_YELLOW, 1)
                else:
                    cv2.circle(frame, pt1, 3, V_PURPLE, -1)
                    cv2.circle(frame, pt2, 3, V_PURPLE, -1)

        # 4. Prepare data strings and calculate dimensions
        data_lines = []
        max_w = 150 # Minimum width
        for data in v_viz_data:
            item = data['item']
            dist = np.linalg.norm(np.array(data['p1']) - np.array(data['p2'])) if data['p1'] is not None and data['p2'] is not None else 0.0
            
            if item['type'] == 'axis':
                text = f"{item['name']}: {data['val']:.2f} ({dist:.3f}m)"
                color = V_GREEN
            else:
                text = f"{item['name']}: {'ON' if data['pressed'] else 'OFF'} ({dist:.3f}m)"
                color = V_YELLOW if data['pressed'] else (200, 200, 200)
            
            data_lines.append((text, color))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            max_w = max(max_w, tw + 40)

        # 5. Create Dynamic Bottom Overlay
        if data_lines:
            bar_height = 15 + (len(data_lines) * 20)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - bar_height), (max_w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # 6. List data in the overlay
            for i, (text, txt_color) in enumerate(data_lines):
                cv2.putText(frame, text, (20, h - bar_height + 20 + i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1)

        self._show_frame(frame)

    def _show_frame(self, frame):
        """Display a (possibly annotated) frame. Overridden by mediapipe_calibrator."""
        cv2.imshow(f"{self.get_name()} Visualization", frame)
        cv2.waitKey(1)
