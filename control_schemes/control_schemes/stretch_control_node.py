#!/usr/bin/env python3
"""StretchControlNode: 9-axis position/velocity/mixed joystick control scheme.

Publishes to /joint_position_cmd, /joint_velocity_cmd, /cmd_vel, and pushes
matching joint modes to a discovered driver node via its set_parameters service.
"""

import sys
import argparse
import rclpy
from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from multi_teleop.base import ControlSchemeNode

class StretchControlNode(ControlSchemeNode):
    """Joystick control scheme for direct joint position/velocity/mixed control.

    One node class backs three entry-point modes (position/velocity/mixed),
    selected via the ``mode`` constructor arg or ``--mode`` CLI flag.
    """

    def __init__(self, mode="position"):
        """Builds the node in the given mode ("position", "velocity", or "mixed").

        Args:
            mode (str): Which control mode's axes/buttons/topics to set up.
        """
        self._mode = mode
        
        # Define the 9 axes as required
        self.axis_names = [
            "Lift", "Base X", "Base Y", "Base Theta", "Arm", 
            "Wrist Roll", "Wrist Pitch", "Wrist Yaw", "Gripper"
        ]
        
        # Mixed mode includes the "Hold for Position Control" button
        if self._mode == "mixed":
            self.button_names = ["Hold for Position Control"]
        else:
            self.button_names = []
            
        node_name = f"stretch_{self._mode}_control"
        
        super().__init__(node_name, self.axis_names, self.button_names)
        
        # Declare parameters for base velocity limits
        self.declare_parameter("max_linear_vel", 0.5)
        self.declare_parameter("max_angular_vel", 0.5)
        
        # Conservative joint position limits as requested:
        # Lift standard range: (0.0, 1.1) -> 15% inner safety margin: (0.165, 0.935)
        self.declare_parameter("limit.lift.lower", 0.165)
        self.declare_parameter("limit.lift.upper", 0.935)
        
        # Arm standard range: (0.0, 0.51) -> 15% inner safety margin: (0.0765, 0.4335)
        self.declare_parameter("limit.arm.lower", 0.0765)
        self.declare_parameter("limit.arm.upper", 0.4335)
        
        # Wrists: Conservative +/- Pi/3 (~1.047 rad)
        self.declare_parameter("limit.wrist_roll.lower", -1.047)
        self.declare_parameter("limit.wrist_roll.upper", 1.047)
        self.declare_parameter("limit.wrist_pitch.lower", -1.047)
        self.declare_parameter("limit.wrist_pitch.upper", 1.047)
        self.declare_parameter("limit.wrist_yaw.lower", -1.047)
        self.declare_parameter("limit.wrist_yaw.upper", 1.047)
        
        # Gripper: full aperture-open angle in radians (~0.849, from aperture_open_m/finger_length_m
        # in robot_params_SE4.py) -- the driver's "stretch_gripper" position unit, matching its own
        # /joint_states readback (gripper_finger_left_joint + gripper_finger_right_joint).
        self.declare_parameter("limit.gripper.lower", 0.0)
        self.declare_parameter("limit.gripper.upper", 0.85)
        
        # Declare parameters for joint velocity scaling limits
        self.declare_parameter("max_joint_vel.lift", 0.2)
        self.declare_parameter("max_joint_vel.arm", 0.2)
        self.declare_parameter("max_joint_vel.wrist_roll", 1.0)
        self.declare_parameter("max_joint_vel.wrist_pitch", 1.0)
        self.declare_parameter("max_joint_vel.wrist_yaw", 1.0)
        self.declare_parameter("max_joint_vel.gripper", 1.0)
        
        # Publishers
        self.pos_pub = self.create_publisher(JointState, "/joint_position_cmd", 10)
        self.vel_pub = self.create_publisher(JointState, "/joint_velocity_cmd", 10)
        self.base_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # Driver interaction state
        self.driver_node = None
        self.driver_namespace = None
        self.param_client = None
        self._last_driver_mode_is_pos = None
        
        # Try to locate driver node on startup
        self.discover_driver_node()
        self.discovery_timer = self.create_timer(1.0, self.discover_driver_node)
        
        self.get_logger().info(f"Initialized {node_name} in mode: {self._mode}")

    def discover_driver_node(self):
        """Discovers the active stretch driver node name dynamically."""
        if self.param_client is not None:
            return True
            
        try:
            node_names_and_namespaces = self.get_node_names_and_namespaces()
            for name, namespace in node_names_and_namespaces:
                if name == self.get_name():
                    continue
                services = self.get_service_names_and_types_by_node(name, namespace)
                for service_name, service_types in services:
                    if service_name.endswith("/home_the_robot") and "std_srvs/srv/Trigger" in service_types:
                        ns_prefix = namespace if namespace != '/' else ''
                        service_path = f"{ns_prefix}/{name}/set_parameters"
                        self.get_logger().info(f"Discovered driver node '{name}' in namespace '{namespace}'. Service: '{service_path}'")
                        self.driver_node = name
                        self.driver_namespace = namespace
                        self.param_client = self.create_client(SetParameters, service_path)
                        return True
        except Exception as e:
            self.get_logger().debug(f"Driver discovery error: {e}")
        return False

    def update_driver_joint_modes(self, target_is_pos):
        """Asynchronously requests the driver to update joint modes (position or velocity)."""
        if self._last_driver_mode_is_pos == target_is_pos:
            return
            
        if self.param_client is None:
            self.discover_driver_node()
            if self.param_client is None:
                return
                
        if not self.param_client.service_is_ready():
            return
            
        params = []
        for joint in ["lift", "arm", "wrist_roll", "wrist_pitch", "wrist_yaw", "stretch_gripper"]:
            p = ParamMsg()
            p.name = f"joint_mode.{joint}"
            p.value = ParameterValue(
                type=ParameterType.PARAMETER_STRING,
                string_value="position" if target_is_pos else "velocity"
            )
            params.append(p)

        req = SetParameters.Request()
        req.parameters = params
        
        future = self.param_client.call_async(req)
        def done_cb(fut):
            try:
                res = fut.result()
                if res and all(r.successful for r in res.results):
                    self.get_logger().info(f"Successfully set driver joint modes to: {'position' if target_is_pos else 'velocity'}")
                    self._last_driver_mode_is_pos = target_is_pos
                else:
                    self.get_logger().error("Failed to set joint modes on driver")
            except Exception as e:
                self.get_logger().error(f"Error calling driver set_parameters: {e}")
                
        future.add_done_callback(done_cb)

    def is_position_control_mode(self, buttons):
        """Returns whether position control is active for the current mode/buttons.

        Args:
            buttons (list): Latest button states; only used in "mixed" mode.
        """
        if self._mode == "position":
            return True
        elif self._mode == "velocity":
            return False
        elif self._mode == "mixed":
            if buttons and len(buttons) > 0:
                return bool(buttons[0])
            return False
        return False

    def map_range(self, val, joint_name):
        """Maps a joystick axis value in [-1, 1] to joint_name's declared position limits.

        Args:
            val (float): Axis value in [-1, 1].
            joint_name (str): Joint whose declared limit.<name>.lower/upper to use.
        """
        lower = self.get_parameter(f"limit.{joint_name}.lower").value
        upper = self.get_parameter(f"limit.{joint_name}.upper").value
        return (val + 1.0) / 2.0 * (upper - lower) + lower

    def handle_joy(self, axes: list, buttons: list):
        """Publishes base twist and (position- or velocity-mode) joint commands from Joy input.

        Args:
            axes (list): 9 axis values (lift, base x/y/theta, arm, wrist rpy, gripper).
            buttons (list): Mode-dependent buttons; only "Hold for Position Control" in mixed mode.
        """
        if len(axes) < 9:
            self.get_logger().warn(f"Expected at least 9 axes, got {len(axes)}")
            return
            
        is_pos = self.is_position_control_mode(buttons)
        self.update_driver_joint_modes(is_pos)
        
        max_linear_vel = self.get_parameter("max_linear_vel").value
        max_angular_vel = self.get_parameter("max_angular_vel").value
        
        # 1. Base command (always velocity via /cmd_vel)
        twist = Twist()
        twist.linear.x = float(axes[1] * max_linear_vel)
        twist.linear.y = float(axes[2] * max_linear_vel)
        twist.angular.z = float(axes[3] * max_angular_vel)
        self.base_pub.publish(twist)
        
        # 2. Joint commands depending on active mode
        if is_pos:
            # Position control: Publish joint positions
            pos_msg = JointState()
            pos_msg.header.stamp = self.get_clock().now().to_msg()
            pos_msg.name = ["lift", "arm", "wrist_roll", "wrist_pitch", "wrist_yaw", "stretch_gripper"]
            pos_msg.position = [
                self.map_range(axes[0], "lift"),
                self.map_range(axes[4], "arm"),
                self.map_range(axes[5], "wrist_roll"),
                self.map_range(axes[6], "wrist_pitch"),
                self.map_range(axes[7], "wrist_yaw"),
                self.map_range(axes[8], "gripper")
            ]
            self.pos_pub.publish(pos_msg)
            
            # Compliant with prompt: also publish to velocity command (zero velocity) to stop any velocity drift
            vel_msg = JointState()
            vel_msg.header.stamp = self.get_clock().now().to_msg()
            vel_msg.name = ["lift", "arm", "wrist_roll", "wrist_pitch", "wrist_yaw", "stretch_gripper"]
            vel_msg.velocity = [0.0] * 6
            self.vel_pub.publish(vel_msg)
        else:
            # Velocity control: Publish velocities
            vel_msg = JointState()
            vel_msg.header.stamp = self.get_clock().now().to_msg()
            vel_msg.name = ["lift", "arm", "wrist_roll", "wrist_pitch", "wrist_yaw", "stretch_gripper"]
            vel_msg.velocity = [
                float(axes[0] * self.get_parameter("max_joint_vel.lift").value),
                float(axes[4] * self.get_parameter("max_joint_vel.arm").value),
                float(-1*axes[5] * self.get_parameter("max_joint_vel.wrist_roll").value),
                float(-1*axes[6] * self.get_parameter("max_joint_vel.wrist_pitch").value),
                float(-1*axes[7] * self.get_parameter("max_joint_vel.wrist_yaw").value),
                float(axes[8] * self.get_parameter("max_joint_vel.gripper").value),
            ]
            self.vel_pub.publish(vel_msg)

def make_node(mode):
    """Constructs a StretchControlNode in the given mode."""
    return StretchControlNode(mode=mode)

def main_position(args=None):
    """Entry point: spins a StretchControlNode fixed in position-control mode."""
    rclpy.init(args=args)
    node = make_node("position")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main_velocity(args=None):
    """Entry point: spins a StretchControlNode fixed in velocity-control mode."""
    rclpy.init(args=args)
    node = make_node("velocity")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main_mixed(args=None):
    """Entry point: spins a StretchControlNode in mixed (button-toggled) mode."""
    rclpy.init(args=args)
    node = make_node("mixed")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main(args=None):
    """Entry point: parses --mode from argv and spins a StretchControlNode."""
    parser = argparse.ArgumentParser(description="Stretch control scheme node.")
    parser.add_argument("--mode", "-m", choices=["position", "velocity", "mixed"], default="position")
    parsed_args, unknown = parser.parse_known_args()
    
    # Initialize rclpy without parsed flags to avoid collisions
    rclpy.init(args=unknown)
    node = make_node(parsed_args.mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
