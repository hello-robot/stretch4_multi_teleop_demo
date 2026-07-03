#!/usr/bin/env python3
import rclpy
from multi_teleop.base import InputInterfaceNode
from pynput import mouse
import threading
import tkinter as tk

class MouseNode(InputInterfaceNode):
    def __init__(self):
        axis_names = ['x', 'y', 'scroll']
        button_names = ['left_click', 'right_click']
        super().__init__('mouse', axis_names, button_names)
        
        # Try to get screen size programmatically
        try:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            self.get_logger().info(f"Detected screen size: {screen_width}x{screen_height}")
        except Exception as e:
            self.get_logger().warn(f"Could not detect screen size via tkinter: {e}. Using defaults.")
            screen_width = 1920
            screen_height = 1080

        self.declare_parameter('screen_width', screen_width)
        self.declare_parameter('screen_height', screen_height)
        self.declare_parameter('scroll_scale', 0.05)
        
        self._current_x = 0.0
        self._current_y = 0.0
        self._scroll_value = 0.0
        self._left_pressed = 0
        self._right_pressed = 0
        
        self.lock = threading.Lock()
        
        self.listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll)
        self.listener.start()
        
        self.get_logger().info("Mouse listener started.")
        
        # Create a timer to poll the state and publish
        self.timer = self.create_timer(0.02, self.update_and_publish)

    def _on_move(self, x, y):
        width = self.get_parameter('screen_width').value
        height = self.get_parameter('screen_height').value
        
        # Normalize to -1 to 1
        with self.lock:
            self._current_x = (2.0 * x / width) - 1.0
            self._current_y = (2.0 * y / height) - 1.0
            # Clamp just in case
            self._current_x = max(-1.0, min(1.0, self._current_x))
            self._current_y = max(-1.0, min(1.0, self._current_y))

    def _on_click(self, x, y, button, pressed):
        with self.lock:
            if button == mouse.Button.left:
                self._left_pressed = 1 if pressed else 0
            elif button == mouse.Button.right:
                self._right_pressed = 1 if pressed else 0

    def _on_scroll(self, x, y, dx, dy):
        scale = self.get_parameter('scroll_scale').value
        with self.lock:
            # dy is positive for up, negative for down
            # "scrolling down decrements to -1 and scrolling up increments an axis to 1"
            self._scroll_value += dy * scale
            self._scroll_value = max(-1.0, min(1.0, self._scroll_value))

    def update_and_publish(self):
        with self.lock:
            axes = [self._current_x, self._current_y, self._scroll_value]
            buttons = [self._left_pressed, self._right_pressed]
            
            # Decay scroll value slowly back to 0 if desired? 
            # The user didn't specify, but often scroll is transient.
            # However, they said "increments an axis to 1", which suggests accumulation.
            # We'll leave it as is for now.
            
            self.publish_input(axes, buttons)

    def stop(self):
        if hasattr(self, 'listener'):
            self.listener.stop()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MouseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in MouseNode: {e}")
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
