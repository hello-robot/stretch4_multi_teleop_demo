"""Minimal tutorial reference for writing a new InputInterfaceNode subclass."""
import rclpy
from multi_teleop.base import InputInterfaceNode
from sensor_msgs.msg import Joy
from pynput import keyboard
import threading

class KeyboardInputInterface(InputInterfaceNode):
    """Tutorial input interface: arrow keys accumulate two axes [0,1];
    w/a/s/d are momentary buttons."""

    def __init__(self):
        # Pass axis and button names to the super constructor
        # This will declare them as read-only ROS parameters
        super().__init__(
            node_name='keyboard_input_interface',
            axis_names=["up-down", "left-right"],
            button_names=["w", "a", "s", "d"]
        )
        
        # Local state to track accumulation for axes
        self.axis_accumulator = {"up-down": 0.5, "left-right": 0.5}
        self.button_states = {"w": 0, "a": 0, "s": 0, "d": 0}
        
        # Key states
        self.pressed_keys = set()
        
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        
        # Timer to update and publish
        self.timer = self.create_timer(0.05, self.update_and_publish) # 20Hz
        
        self.get_logger().info(f"KeyboardInputInterface initialized. Axes: {self.control_axes}, Buttons: {self.control_buttons}")

    def _on_press(self, key):
        """pynput callback: record a pressed key and set its button if tracked."""
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        self.pressed_keys.add(k)

        if k in self.button_states:
            self.button_states[k] = 1

    def _on_release(self, key):
        """pynput callback: forget a released key and clear its button if tracked."""
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        
        if k in self.pressed_keys:
            self.pressed_keys.remove(k)
        
        if k in self.button_states:
            self.button_states[k] = 0

    def update_and_publish(self):
        """Timer callback (20Hz): integrate arrow-key axes and publish_input."""
        # Update axes based on held keys
        # "holding down left decreases the value in left-right, and holding right increases it"
        # "down decreases, up increases"
        step = 0.01
        
        if 'Key.up' in self.pressed_keys:
            self.axis_accumulator["up-down"] = min(1.0, self.axis_accumulator["up-down"] + step)
        if 'Key.down' in self.pressed_keys:
            self.axis_accumulator["up-down"] = max(0.0, self.axis_accumulator["up-down"] - step)
            
        if 'Key.right' in self.pressed_keys:
            self.axis_accumulator["left-right"] = min(1.0, self.axis_accumulator["left-right"] + step)
        if 'Key.left' in self.pressed_keys:
            self.axis_accumulator["left-right"] = max(0.0, self.axis_accumulator["left-right"] - step)
            
        # Create lists matching the name order
        axes = [self.axis_accumulator[name] for name in self.control_axes]
        buttons = [self.button_states[name] for name in self.control_buttons]
        
        # Call the base class publish_input which handles None, state, and ROS parameters
        self.publish_input(axes, buttons)

def main(args=None):
    """Entry point: spin a KeyboardInputInterface node until interrupted."""
    rclpy.init(args=args)
    node = KeyboardInputInterface()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.listener.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
