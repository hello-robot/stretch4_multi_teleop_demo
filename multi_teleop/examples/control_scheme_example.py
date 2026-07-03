import rclpy
from multi_teleop.base import ControlSchemeNode

class ExampleControlScheme(ControlSchemeNode):
    def __init__(self):
        # Pass axis and button names to the super constructor
        super().__init__(
            node_name='example_control_scheme',
            axis_names=["x", "y", "z"],
            button_names=["button"]
        )
        self.get_logger().info(f"ExampleControlScheme initialized with axes: {self.control_axes} and buttons: {self.control_buttons}")

    def handle_joy(self, axes: list, buttons: list):
        self.get_logger().info(f"Received Joy data: axes={axes}, buttons={buttons}")

def main(args=None):
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
