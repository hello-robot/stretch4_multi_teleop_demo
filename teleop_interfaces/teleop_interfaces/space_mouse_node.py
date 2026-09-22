#!/usr/bin/env python3
"""ROS2 node publishing Joy messages from a 3Dconnexion SpaceMouse via pyspacemouse."""
import rclpy
from multi_teleop.base import InputInterfaceNode
import pyspacemouse

class SpaceMouseNode(InputInterfaceNode):
    """Opens a SpaceMouse device and polls it on a timer to publish 6DOF + 2 buttons."""

    def __init__(self):
        """Open the SpaceMouse device (raises on failure) and start the poll timer."""
        axis_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        button_names = ['button-1', 'button-2']
        super().__init__('space_mouse', axis_names, button_names)
        
        self.get_logger().info("Opening SpaceMouse...")
        try:
            self.device = pyspacemouse.open()
        except Exception as e:
            self.get_logger().error(f"Could not open SpaceMouse: {e}")
            raise e
            
        self.get_logger().info("SpaceMouse opened successfully.")
        
        # Create a timer to poll the device
        self.timer = self.create_timer(0.01, self.update_and_publish)

    def update_and_publish(self):
        """Timer callback: read one SpaceMouse state and publish it as Joy."""
        try:
            state = self.device.read()
            if state is not None:
                # state.x, state.y, state.z, state.roll, state.pitch, state.yaw
                raw_axes = [
                    float(state.x), 
                    float(state.y), 
                    float(state.z), 
                    float(state.roll), 
                    float(state.pitch), 
                    float(state.yaw)
                ]
                # Clamp to -1 to 1
                axes = [max(-1.0, min(1.0, a)) for a in raw_axes]
                # state.buttons is typically a list of ints, e.g., [0, 0]
                buttons = [int(b) for b in state.buttons]
                
                # InputInterfaceNode expects these to match the lengths of axis_names and button_names
                if len(buttons) >= 2:
                    buttons = buttons[:2]
                elif len(buttons) < 2:
                    buttons.extend([0] * (2 - len(buttons)))
                
                self.publish_input(axes, buttons)
        except Exception as e:
            self.get_logger().error(f"Error reading from SpaceMouse: {e}")

    def stop(self):
        """Close the SpaceMouse device handle, if one was opened."""
        if hasattr(self, 'device'):
            self.device.close()

def main(args=None):
    """Entry point: construct SpaceMouseNode and spin until interrupted."""
    rclpy.init(args=args)
    node = None
    try:
        node = SpaceMouseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in SpaceMouseNode: {e}")
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
