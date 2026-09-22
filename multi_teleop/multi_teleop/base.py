"""Shared base classes for the multi_teleop input/control-scheme pipeline.

Defines ``ControlSchemeNode`` and ``InputInterfaceNode``, the abstract
base classes that every input device node and control scheme node in
``teleop_interfaces`` and ``control_schemes`` subclasses. Both exchange
``sensor_msgs/msg/Joy`` messages over ``/{node_name}/output`` (interfaces)
and ``/{node_name}/input`` (schemes).
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from rcl_interfaces.msg import ParameterDescriptor
from abc import ABC, abstractmethod
from std_srvs.srv import Trigger

class ControlSchemeNode(Node, ABC):
    """Base class for nodes that consume routed teleop commands.

    Registers ``/{node_name}/activate`` and ``/{node_name}/deactivate``
    ``std_srvs/srv/Trigger`` services, and silences all of this node's
    publishers while inactive (see ``create_publisher`` override below).
    Subscribes to ``/{node_name}/input`` (``sensor_msgs/msg/Joy``) and
    dispatches validated axes/buttons to the abstract ``handle_joy``.
    """

    def __init__(self, node_name, axis_names, button_names, **kwargs):
        """Register activation services, input subscriber, and name params.

        Args:
            node_name: ROS node name; also used as the service/topic prefix.
            axis_names: Ordered list of expected control axis names.
            button_names: Ordered list of expected control button names.
        """
        # Set before super().__init__() so that create_publisher's
        # activation-silencing wrapper (below) has self.active available
        # even for publishers Node.__init__() itself creates internally
        # (e.g. the parameter-event publisher).
        self.active = True  # Active by default for standalone/backward-compatibility

        super().__init__(node_name, **kwargs)
        self._axes = axis_names
        self._buttons = button_names

        # Services to activate/deactivate
        self.create_service(Trigger, f'{node_name}/activate', self._activate_callback)
        self.create_service(Trigger, f'{node_name}/deactivate', self._deactivate_callback)

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

    def create_publisher(self, msg_type, topic, qos_profile, **kwargs):
        """Override ``Node.create_publisher``: silence publishing while inactive.

        A real method override (rather than an instance-level monkey-patch)
        so a base-class ``Node.create_publisher(self, ...)`` call can't
        bypass the silencing wrapper below. Wraps the created
        ``Publisher``'s ``publish`` so calls are dropped whenever
        ``self.active`` is False.
        """
        pub = super().create_publisher(msg_type, topic, qos_profile, **kwargs)
        original_publish = pub.publish
        def wrapped_publish(msg):
            if not self.active:
                return  # Ignore/Silence if inactive
            return original_publish(msg)
        pub.publish = wrapped_publish
        return pub

    def _activate_callback(self, request, response):
        """Trigger service handler: mark this scheme active."""
        self.active = True
        response.success = True
        response.message = f"Control scheme {self.get_name()} activated."
        self.get_logger().info(response.message)
        return response

    def _deactivate_callback(self, request, response):
        """Trigger service handler: mark this scheme inactive."""
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
        """Validate/pad incoming Joy axes+buttons, then call handle_joy."""
        if not self.active:
            return
        # Check if length matching with the names
        axes_len = len(self.control_axes)
        buttons_len = len(self.control_buttons)

        if len(msg.axes) != axes_len:
            self.get_logger().warn(
                f"Received Joy message with {len(msg.axes)} axes, but expected {axes_len}",
                throttle_duration_sec=5.0)
            axes = list(msg.axes)
            if len(axes) < axes_len:
                axes.extend([0.0] * (axes_len - len(axes)))
            else:
                axes = axes[:axes_len]
        else:
            axes = list(msg.axes)

        if len(msg.buttons) != buttons_len:
            self.get_logger().warn(
                f"Received Joy message with {len(msg.buttons)} buttons, but expected {buttons_len}",
                throttle_duration_sec=5.0)
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
    """Base class for nodes that expose a physical/virtual input device.

    Tracks the device's last-known axes/buttons and publishes
    ``sensor_msgs/msg/Joy`` to ``/{node_name}/output`` via
    ``publish_input``.
    """

    def __init__(self, node_name, axis_names, button_names, **kwargs):
        """Set up device state, name params, and the output publisher.

        Args:
            node_name: ROS node name; also used as the output topic prefix.
            axis_names: Ordered list of this device's axis names.
            button_names: Ordered list of this device's button names.
        """
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
