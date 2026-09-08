import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from rcl_interfaces.msg import ParameterDescriptor
from abc import ABC, abstractmethod
from std_srvs.srv import Trigger

class ControlSchemeNode(Node, ABC):
    def __init__(self, node_name, axis_names, button_names, **kwargs):
        super().__init__(node_name, **kwargs)
        self._axes = axis_names
        self._buttons = button_names
        
        self.active = True  # Active by default for standalone/backward-compatibility
        
        # Services to activate/deactivate
        self.create_service(Trigger, f'{node_name}/activate', self._activate_callback)
        self.create_service(Trigger, f'{node_name}/deactivate', self._deactivate_callback)
        
        # Wrap create_publisher to silence child class publishers when inactive
        self_create_pub = super().create_publisher
        def wrapped_create_publisher(msg_type, topic, qos_profile, **kwargs):
            pub = self_create_pub(msg_type, topic, qos_profile, **kwargs)
            original_publish = pub.publish
            def wrapped_publish(msg):
                if not self.active:
                    return  # Ignore/Silence if inactive
                return original_publish(msg)
            pub.publish = wrapped_publish
            return pub
        self.create_publisher = wrapped_create_publisher
        
        # Declare read-only parameters for names in initialization
        self.declare_parameter(
            'axis_names', 
            self._axes,
            descriptor=ParameterDescriptor(read_only=False, description="Names of the control axes")
        )
        self.declare_parameter(
            'button_names', 
            self._buttons,
            descriptor=ParameterDescriptor(read_only=False, description="Names of the control buttons")
        )
        
        # Subscriber for input Joy messages
        self.input_subscription = self.create_subscription(
            Joy,
            f'{node_name}/input',
            self._joy_callback,
            1
        )

    def _activate_callback(self, request, response):
        self.active = True
        response.success = True
        response.message = f"Control scheme {self.get_name()} activated."
        self.get_logger().info(response.message)
        return response

    def _deactivate_callback(self, request, response):
        self.active = False
        response.success = True
        response.message = f"Control scheme {self.get_name()} deactivated."
        self.get_logger().info(response.message)
        return response

    @property
    def control_axes(self):
        """Returns the list of names for control axes."""
        return self._axes

    @property
    def control_buttons(self):
        """Returns the list of names for control buttons."""
        return self._buttons

    def _joy_callback(self, msg: Joy):
        if not self.active:
            return
        # Check if length matching with the names
        axes_len = len(self.control_axes)
        buttons_len = len(self.control_buttons)

        if len(msg.axes) != axes_len:
            self.get_logger().warn(f"Received Joy message with {len(msg.axes)} axes, but expected {axes_len}")
            axes = list(msg.axes)
            if len(axes) < axes_len:
                axes.extend([0.0] * (axes_len - len(axes)))
            else:
                axes = axes[:axes_len]
        else:
            axes = list(msg.axes)

        if len(msg.buttons) != buttons_len:
            self.get_logger().warn(f"Received Joy message with {len(msg.buttons)} buttons, but expected {buttons_len}")
            buttons = list(msg.buttons)
            if len(buttons) < buttons_len:
                buttons.extend([0] * (buttons_len - len(buttons)))
            else:
                buttons = buttons[:buttons_len]
        else:
            buttons = list(msg.buttons)

        self.handle_joy(axes, buttons)

    @abstractmethod
    def handle_joy(self, axes: list, buttons: list):
        """Handle incoming Joy message components."""
        pass

class InputInterfaceNode(Node, ABC):
    def __init__(self, node_name, axis_names, button_names, **kwargs):
        super().__init__(node_name, **kwargs)
        self._axes = axis_names
        self._buttons = button_names
        
        # Internal state to track most recent values
        self._last_axes = [0.0] * len(self._axes)
        self._last_buttons = [0] * len(self._buttons)
        
        # Declare parameters in initialization
        self.declare_parameter(
            'axis_names', 
            self._axes,
            descriptor=ParameterDescriptor(description="Names of the control axes")
        )
        self.declare_parameter(
            'button_names', 
            self._buttons,
            descriptor=ParameterDescriptor(description="Names of the control buttons")
        )
        
        # Publisher for output Joy messages
        self.output_publisher = self.create_publisher(
            Joy,
            f'{node_name}/output',
            1
        )

    @property
    def control_axes(self):
        """Returns the list of names for control axes."""
        return self._axes

    @property
    def control_buttons(self):
        """Returns the list of names for control buttons."""
        return self._buttons

    def publish_input(self, axis_values: list, button_values: list):
        """Handler to publish input. Updates state and ROS parameters.
        Call on device state change."""
        if len(axis_values) != len(self.control_axes):
            raise ValueError(f"axis_values length mismatch")
        
        if len(button_values) != len(self.control_buttons):
            raise ValueError(f"button_values length mismatch")

        # Update last values
        updated_axes = False
        updated_buttons = False
        for i, val in enumerate(axis_values):
            if val is not None:
                self._last_axes[i] = float(val)
                updated_axes = True
        
        for i, val in enumerate(button_values):
            if val is not None:
                self._last_buttons[i] = int(val)
                updated_buttons = True

        # Create and publish Joy message
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = self._last_axes
        msg.buttons = self._last_buttons
        
        self.output_publisher.publish(msg)
