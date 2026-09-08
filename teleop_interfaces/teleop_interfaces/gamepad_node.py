#!/usr/bin/env python3
import rclpy
from multi_teleop.base import InputInterfaceNode
import evdev
from evdev import ecodes
import threading
import time

class GamepadNode(InputInterfaceNode):
    def __init__(self):
        # Default Xbox 360 controller layouts
        axis_names = [
            'Left Stick X', 'Left Stick Y', 'Left Trigger',
            'Right Stick X', 'Right Stick Y', 'Right Trigger',
            'D-pad X', 'D-pad Y'
        ]
        button_names = [
            'A Button', 'B Button', 'X Button', 'Y Button',
            'Left Bumper', 'Right Bumper',
            'Select Button', 'Start Button', 'Xbox Button',
            'Left Stick Click', 'Right Stick Click'
        ]
        super().__init__('gamepad', axis_names, button_names)

        self.declare_parameter('device_path', '')
        self.declare_parameter('poll_rate', 0.02)  # 50 Hz default publication rate
        self.declare_parameter('deadzone', 0.05)   # Default deadzone for analog sticks

        self._current_axes = [0.0] * len(axis_names)
        self._current_buttons = [0] * len(button_names)
        
        self.lock = threading.Lock()
        self.device = None
        self._stop_thread = False

        # Xbox 360 mapping tables
        self.BUTTON_MAP = {
            304: 0,  # BTN_A
            305: 1,  # BTN_B
            307: 2,  # BTN_X
            308: 3,  # BTN_Y
            310: 4,  # BTN_TL / LB
            311: 5,  # BTN_TR / RB
            314: 6,  # BTN_SELECT
            315: 7,  # BTN_START
            316: 8,  # BTN_MODE (Xbox Guide)
            317: 9,  # BTN_THUMBL (L Stick Click)
            318: 10, # BTN_THUMBR (R Stick Click)
        }

        self.AXIS_MAP = {
            0: (0, 32768.0, False),  # ABS_X: Left Stick X
            1: (1, 32768.0, True),   # ABS_Y: Left Stick Y (invert)
            2: (2, 255.0, False),    # ABS_Z: Left Trigger
            3: (3, 32768.0, False),  # ABS_RX: Right Stick X
            4: (4, 32768.0, True),   # ABS_RY: Right Stick Y (invert)
            5: (5, 255.0, False),    # ABS_RZ: Right Trigger
            16: (6, 1.0, False),     # ABS_HAT0X: D-pad X
            17: (7, 1.0, True),      # ABS_HAT0Y: D-pad Y (invert)
        }

        # Start device connection thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        # Timer to publish state
        poll_rate = self.get_parameter('poll_rate').value
        self.timer = self.create_timer(poll_rate, self.update_and_publish)

    def _connect_gamepad(self):
        device_path = self.get_parameter('device_path').value
        if device_path:
            try:
                self.device = evdev.InputDevice(device_path)
                self.get_logger().info(f"Connected to gamepad at {device_path} ({self.device.name})")
            except Exception as e:
                self.get_logger().warn(f"Failed to open gamepad at {device_path}: {e}")
                self.device = None
        else:
            # Auto-detect Xbox / Pad / Controller
            try:
                for path in evdev.list_devices():
                    dev = evdev.InputDevice(path)
                    if any(kw in dev.name.lower() for kw in ['pad', 'xbox', 'controller']):
                        self.device = dev
                        self.get_logger().info(f"Auto-detected gamepad at {path} ({dev.name})")
                        break
            except Exception as e:
                self.get_logger().error(f"Error listing input devices: {e}")

    def _run(self):
        while rclpy.ok() and not self._stop_thread:
            if self.device is None:
                self._connect_gamepad()
                if self.device is None:
                    time.sleep(2.0)
                    continue

            try:
                for event in self.device.read_loop():
                    if self._stop_thread:
                        break
                    self._process_event(event)
            except Exception as e:
                self.get_logger().warn(f"Lost connection to gamepad: {e}")
                self.device = None
                with self.lock:
                    self._current_axes = [0.0] * len(self._axes)
                    self._current_buttons = [0] * len(self._buttons)
                time.sleep(1.0)

    def _process_event(self, event):
        if event.type == ecodes.EV_KEY:
            if event.code in self.BUTTON_MAP:
                idx = self.BUTTON_MAP[event.code]
                with self.lock:
                    self._current_buttons[idx] = 1 if event.value > 0 else 0
        elif event.type == ecodes.EV_ABS:
            if event.code in self.AXIS_MAP:
                idx, scale, invert = self.AXIS_MAP[event.code]
                val = float(event.value) / scale
                if invert:
                    val = -val
                val = max(-1.0, min(1.0, val))
                
                # Apply hardware-level deadzone for analog sticks
                if idx in [0, 1, 3, 4]:
                    deadzone = self.get_parameter('deadzone').value
                    if abs(val) < deadzone:
                        val = 0.0
                    else:
                        import math
                        val = math.copysign((abs(val) - deadzone) / (1.0 - deadzone), val)
                
                with self.lock:
                    self._current_axes[idx] = val

    def update_and_publish(self):
        with self.lock:
            # Make a copy to avoid race conditions
            axes = list(self._current_axes)
            buttons = list(self._current_buttons)
        self.publish_input(axes, buttons)

    def stop(self):
        self._stop_thread = True

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GamepadNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in GamepadNode: {e}")
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
