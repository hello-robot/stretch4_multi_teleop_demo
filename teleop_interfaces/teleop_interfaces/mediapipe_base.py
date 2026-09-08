# Replaced cv_bridge with a pure Python/NumPy implementation to support NumPy 2.x
import rclpy
from multi_teleop.base import InputInterfaceNode
from sensor_msgs.msg import Image, Joy

class CvBridge:
    def imgmsg_to_cv2(self, msg, desired_encoding="bgr8"):
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

class MediaPipeBaseNode(InputInterfaceNode):
    def __init__(self, node_name, config_path=None, **kwargs):
        # 6DOF base axes are always present: x, y, z, roll, pitch, yaw
        self._axes = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        self._buttons = []
        self._last_axes = [0.0 for n in self._axes]
        self._last_buttons = [0.0 for n in self._buttons]
        self._virtual_config = [] # List of {name, p1, p2, type, threshold/max_dist}
        
        # Load config if provided
        if config_path:
            self._load_config(config_path)

        super().__init__(node_name, self._axes, self._buttons, **kwargs)
        
        self.declare_parameter('use_webcam', True)
        self.declare_parameter('webcam_id', 0)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('workspace_zero', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('workspace_center', [0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('workspace_dims', [1.0, 1.0, 1.0])
        self.declare_parameter('workspace_min', [0.0, 0.0, -0.5, -1.0, -1.0, -1.0])
        self.declare_parameter('workspace_max', [1.0, 1.0, 0.5, 1.0, 1.0, 1.0])
        self.declare_parameter('model_path', '') # Required
        self.declare_parameter('visualize', True)
        
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
        for param in params:
            if param.name == 'add_virtual_item_json' and param.type_ == rclpy.Parameter.Type.STRING:
                try:
                    item = json.loads(param.value)
                    self.add_virtual_item(item)
                except Exception as e:
                    self.get_logger().error(f"Failed to add virtual item from JSON: {e}")
        return SetParametersResult(successful=True)

    def add_virtual_item(self, item):
        """
        item: {name, p1, p2, type: 'axis'|'button', max_dist/threshold}
        """
        name = item.get('name')
        if not name: return
        
        if item['type'] == 'axis':
            if name not in self._axes:
                self._axes.append(name)
                self._last_axes.append(0.0) # Initial value for distance-based axis
                self._virtual_config.append(item)
        elif item['type'] == 'button':
            if name not in self._buttons:
                self._buttons.append(name)
                self._last_buttons.append(0)
                self._virtual_config.append(item)


    @abstractmethod
    def process_frame(self, frame):
        """Should update self._landmarks and return 6DOF [x, y, z, r, p, y] or None."""
        pass

    def update_and_publish(self):
        """Implementation of InputInterfaceNode's abstract method."""
        if self.get_parameter('use_webcam').value:
            ret, frame = self._cap.read()
            if ret:
                self._update_fps()
                self._handle_frame(frame)

    def _timer_callback(self):
        # We now use update_and_publish for the timer
        self.update_and_publish()

    def _image_callback(self, msg):
        with self._msg_lock:
            self._latest_msg = msg

    def _processing_loop(self):
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
        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds / 1e9
        if dt > 0:
            # 10-sample rolling average (approx)
            self._fps = self._fps * 0.9 + (1.0 / dt) * 0.1
        self._last_time = now

    def _handle_frame(self, frame):
        state_6dof = self.process_frame(frame)
        if state_6dof is None:
            if self.get_parameter('visualize').value:
                cv2.imshow(f"{self.get_name()} Visualization", frame)
                cv2.waitKey(1)
            return
            
        # Fetch 6DOF parameters
        zero = self.get_parameter('workspace_zero').value
        dims = self.get_parameter('workspace_dims').value
        ws_min = self.get_parameter('workspace_min').value
        ws_max = self.get_parameter('workspace_max').value
        
        # Resolve zero to 6DOF
        if len(zero) == 3:
            zero = list(zero) + [0.0, 0.0, 0.0]
            
        # Resolve ws_min/ws_max to 6DOF
        if len(ws_min) == 3:
            ws_min = list(ws_min) + [-1.0, -1.0, -1.0]
        if len(ws_max) == 3:
            ws_max = list(ws_max) + [1.0, 1.0, 1.0]
            
        # Backwards compatibility with workspace_dims
        if dims != [1.0, 1.0, 1.0]:
            for i in range(3):
                ws_min[i] = zero[i] - dims[i] / 2.0
                ws_max[i] = zero[i] + dims[i] / 2.0

        # Normalize 6DOF asymmetric position & orientation
        axes_values = []
        for i in range(6):
            val = state_6dof[i]
            z_val = zero[i]
            min_val = ws_min[i]
            max_val = ws_max[i]
            
            if val < z_val:
                denom = z_val - min_val
                if abs(denom) < 1e-6: denom = 1e-6
                n_val = (val - z_val) / denom
            else:
                denom = max_val - z_val
                if abs(denom) < 1e-6: denom = 1e-6
                n_val = (val - z_val) / denom
                
            axes_values.append(max(-1.0, min(1.0, n_val)))
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
        cv2.putText(frame, "WORKSPACE", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, V_BLUE, 1)
        
        # 1.5 Draw FPS
        cv2.putText(frame, f"FPS: {self._fps:.1f}", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, V_YELLOW, 2)
        
        # 1.6 Draw 6DOF Pose Panel
        overlay = frame.copy()
        panel_w = 200
        panel_h = 145
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), V_BLUE, 1)
        
        # Calculate asymmetric 6DOF normalized position & orientation using zero
        norm_vals = []
        for i in range(6):
            val = state[i]
            z_val = zero[i]
            min_val = ws_min[i]
            max_val = ws_max[i]
            
            if val < z_val:
                denom = z_val - min_val
                if abs(denom) < 1e-6: denom = 1e-6
                n_val = (val - z_val) / denom
            else:
                denom = max_val - z_val
                if abs(denom) < 1e-6: denom = 1e-6
                n_val = (val - z_val) / denom
                
            norm_vals.append(max(-1.0, min(1.0, n_val)))
        
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

        cv2.imshow(f"{self.get_name()} Visualization", frame)
        cv2.waitKey(1)
