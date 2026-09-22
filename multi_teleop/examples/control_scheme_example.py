"""Minimal tutorial reference for writing a new ControlSchemeNode subclass."""
import rclpy
from multi_teleop.base import ControlSchemeNode

class ExampleControlScheme(ControlSchemeNode):
    """Tutorial control scheme: logs received Joy data, moves nothing."""

    def __init__(self):
        """Declare a 3-axis/1-button scheme named 'example_control_scheme'."""
        # Pass axis and button names to the super constructor
        super().__init__(
            node_name='example_control_scheme',
            axis_names=["x", "y", "z"],
            button_names=["button"]
        )
        self.get_logger().info(f"ExampleControlScheme initialized with axes: {self.control_axes} and buttons: {self.control_buttons}")

    def handle_joy(self, axes: list, buttons: list):
        """Log the routed axes/buttons (no actual robot command)."""
        self.get_logger().info(f"Received Joy data: axes={axes}, buttons={buttons}")

def main(args=None):
    """Entry point: spin an ExampleControlScheme node."""
    rclpy.init(args=args)
    node = ExampleControlScheme()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
