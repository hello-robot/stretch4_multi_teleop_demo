#!/usr/bin/env python3

import sys
import math
import rclpy
from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from multi_teleop.base import ControlSchemeNode

class StretchCombinedControlNode(ControlSchemeNode):
    def __init__(self):
        # 7 Axes as required:
        # 0: Combined Pitch Lift (controls wrist pitch and lift sequentially)
        # 1: Combined Yaw Turn (controls wrist yaw and base theta sequentially)
        # 2: Base X (linear driving forward/backward)
        # 3: Base Y (linear driving sideways/strafing)
        # 4: Arm (extension/retraction)
        # 5: Wrist Roll
        # 6: Gripper
        self.axis_names = [
            "Combined Pitch Lift",
            "Combined Yaw Turn",
            "Base X",
            "Base Y",
            "Arm",
            "Wrist Roll",
            "Gripper"
        ]
        
        # 2 Buttons for freezing specific subsystems:
        # 0: Freeze Lift Base (wrist pitch/yaw only, no limits)
        # 1: Freeze Wrist (lift/base turn only, no wrist movement)
        # 2: Hold to Go Home (slowly drives wrist to 0, arm to 0.15, lift to 0.5)
        self.button_names = [
            "Freeze Lift Base",
            "Freeze Wrist",
            "Hold to Go Home"
        ]
        
        super().__init__("stretch_combined_control", self.axis_names, self.button_names)
        
        # Declare parameters for base velocity limits
        self.declare_parameter("max_linear_vel", 0.5)
        self.declare_parameter("max_angular_vel", 0.5)
        
        # Joint velocity scaling limits
        self.declare_parameter("max_joint_vel.lift", 0.2)
        self.declare_parameter("max_joint_vel.arm", 0.2)
        self.declare_parameter("max_joint_vel.wrist_roll", 1.0)
        self.declare_parameter("max_joint_vel.wrist_pitch", 1.0)
        self.declare_parameter("max_joint_vel.wrist_yaw", 1.0)
        
        # Split limit degree thresholds (defaults: pitch = 45.0, yaw = 45.0)
        self.declare_parameter("wrist_pitch_split_limit_deg", 45.0)
        self.declare_parameter("wrist_yaw_split_limit_deg", 45.0)
        
        # Conservative joint position limits
        self.declare_parameter("limit.lift.lower", 0.165)
        self.declare_parameter("limit.lift.upper", 0.935)
        self.declare_parameter("limit.arm.lower", 0.0765)
        self.declare_parameter("limit.arm.upper", 0.4335)
        self.declare_parameter("limit.wrist_roll.lower", -1.047)
        self.declare_parameter("limit.wrist_roll.upper", 1.047)
        self.declare_parameter("limit.wrist_pitch.lower", -1.047)
        self.declare_parameter("limit.wrist_pitch.upper", 1.047)
        self.declare_parameter("limit.wrist_yaw.lower", -1.047)
        self.declare_parameter("limit.wrist_yaw.upper", 1.047)
        self.declare_parameter("limit.gripper.lower", 0.0)
        self.declare_parameter("limit.gripper.upper", 0.15)
        
        # Publishers
        self.pos_pub = self.create_publisher(JointState, "/joint_position_cmd", 10)
        self.vel_pub = self.create_publisher(JointState, "/joint_velocity_cmd", 10)
        self.base_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # Subscription to joint states for tracking wrist positions
        self.last_joint_state = None
        self.joint_states_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            10
        )
        
        # Driver interaction state
        self.driver_node = None
        self.driver_namespace = None
        self.param_client = None
        self._last_driver_mode_is_pos = None
        
        # Try to locate driver node on startup
        self.discover_driver_node()
        self.discovery_timer = self.create_timer(1.0, self.discover_driver_node)
        
        self.get_logger().info("Initialized stretch_combined_control node successfully.")

    def joint_states_callback(self, msg: JointState):
        self.last_joint_state = msg

    def get_joint_position(self, name, default=0.0):
        if self.last_joint_state is None:
            return default
        try:
            idx = self.last_joint_state.name.index(name)
            return self.last_joint_state.position[idx]
        except ValueError:
            return default

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
        for joint in ["lift", "arm", "wrist_roll", "wrist_pitch", "wrist_yaw"]:
            p = ParamMsg()
            p.name = f"joint_mode.{joint}"
            p.value = ParameterValue(
                type=ParameterType.PARAMETER_STRING,
                string_value="position" if target_is_pos else "velocity"
            )
            params.append(p)
            
        # Gripper is always position mode
        p_grip = ParamMsg()
        p_grip.name = "joint_mode.stretch_gripper"
        p_grip.value = ParameterValue(
            type=ParameterType.PARAMETER_STRING,
            string_value="position"
        )
        params.append(p_grip)
        
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

    def map_range(self, val, joint_name):
        lower = self.get_parameter(f"limit.{joint_name}.lower").value
        upper = self.get_parameter(f"limit.{joint_name}.upper").value
        return (val + 1.0) / 2.0 * (upper - lower) + lower

    def handle_joy(self, axes: list, buttons: list):
        if len(axes) < 7:
            self.get_logger().warn(f"Expected at least 7 axes, got {len(axes)}")
            return
            
        # Ensure driver joints are in velocity mode (gripper in position mode)
        self.update_driver_joint_modes(False)
        
        # Read parameters
        max_linear_vel = self.get_parameter("max_linear_vel").value
        max_angular_vel = self.get_parameter("max_angular_vel").value
        max_lift_vel = self.get_parameter("max_joint_vel.lift").value
        max_arm_vel = self.get_parameter("max_joint_vel.arm").value
        max_roll_vel = self.get_parameter("max_joint_vel.wrist_roll").value
        max_pitch_vel = self.get_parameter("max_joint_vel.wrist_pitch").value
        max_yaw_vel = self.get_parameter("max_joint_vel.wrist_yaw").value
        
        # Get threshold limits (convert degrees parameter to radians)
        pitch_split_limit_deg = self.get_parameter("wrist_pitch_split_limit_deg").value
        pitch_split_limit_rad = pitch_split_limit_deg * math.pi / 180.0
        yaw_split_limit_deg = self.get_parameter("wrist_yaw_split_limit_deg").value
        yaw_split_limit_rad = yaw_split_limit_deg * math.pi / 180.0
        
        # Decode inputs
        a_pl = axes[0]  # Combined Pitch Lift
        a_yt = axes[1]  # Combined Yaw Turn
        a_bx = axes[2]  # Base X
        a_by = axes[3]  # Base Y
        a_arm = axes[4] # Arm
        a_roll = axes[5] # Wrist Roll
        a_grip = axes[6] # Gripper
        
        # Buttons
        freeze_lift_base = bool(buttons[0]) if len(buttons) > 0 else False
        freeze_wrist = bool(buttons[1]) if len(buttons) > 1 else False
        hold_to_go_home = bool(buttons[2]) if len(buttons) > 2 else False
        
        # Velocity lists for JointState publication
        # joint_names: ["lift", "arm", "wrist_roll", "wrist_pitch", "wrist_yaw"]
        vel_cmd = [0.0] * 5
        
        # Base Command (Twist)
        twist = Twist()
        
        def clamp(val, min_val, max_val):
            return max(min_val, min(max_val, val))
            
        # Subsystem status logs/debugging or state resolution
        if hold_to_go_home:
            # Active homing mode: slowly drive joints toward home, freeze base
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            
            # 1. Get current positions
            p_lift = self.get_joint_position("lift_joint", 0.5)
            p_arm = sum(self.get_joint_position(f"arm_l{i}_joint", 0.0375) for i in [1, 2, 3, 4])
            p_roll = self.get_joint_position("wrist_roll_joint", 0.0)
            p_pitch = self.get_joint_position("wrist_pitch_joint", 0.0)
            p_yaw = self.get_joint_position("wrist_yaw_joint", 0.0)
            
            # 2. P control towards target values (Gain Kp = 1.0, saturated to safe slow speeds)
            # Safe slow speeds: 0.05 m/s for lift/arm, 0.20 rad/s for wrists
            vel_cmd[0] = float(clamp(1.0 * (0.5 - p_lift), -0.05, 0.05))   # lift target: 0.5
            vel_cmd[1] = float(clamp(1.0 * (0.15 - p_arm), -0.05, 0.05))   # arm target: 0.15
            vel_cmd[2] = float(clamp(1.0 * (0.0 - p_roll), -0.20, 0.20))   # roll target: 0.0
            vel_cmd[3] = float(clamp(1.0 * (0.0 - p_pitch), -0.20, 0.20))  # pitch target: 0.0
            vel_cmd[4] = float(clamp(1.0 * (0.0 - p_yaw), -0.20, 0.20))    # yaw target: 0.0
        elif freeze_lift_base and freeze_wrist:
            # Safe-stop state: freeze everything if both buttons are held
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            vel_cmd[0] = 0.0  # lift
            vel_cmd[3] = 0.0  # wrist_pitch
            vel_cmd[4] = 0.0  # wrist_yaw
        elif freeze_lift_base:
            # 1. Lift and base are frozen completely
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            vel_cmd[0] = 0.0  # lift
            
            # 2. Wrist pitch and wrist yaw move directly without split limits
            vel_cmd[3] = float(-1.0 * a_pl * max_pitch_vel)
            vel_cmd[4] = float(-1.0 * a_yt * max_yaw_vel)
        elif freeze_wrist:
            # 1. Wrist pitch and wrist yaw are frozen completely
            vel_cmd[3] = 0.0
            vel_cmd[4] = 0.0
            
            # 2. Base turn and lift move directly
            twist.linear.x = float(a_bx * max_linear_vel)
            twist.linear.y = float(a_by * max_linear_vel)
            twist.angular.z = float(-1.0 * a_yt * max_angular_vel)
            vel_cmd[0] = float(a_pl * max_lift_vel)
        else:
            # Normal combined/sequential mode
            # 1. Base X and Base Y move normally
            twist.linear.x = float(a_bx * max_linear_vel)
            twist.linear.y = float(a_by * max_linear_vel)
            
            # 2. Combined Pitch Lift Logic (sequential)
            curr_pitch = self.get_joint_position("wrist_pitch_joint", 0.0)
            if a_pl > 0.0:
                # Command is UP (pitch up is negative position)
                if curr_pitch > -pitch_split_limit_rad:
                    vel_cmd[3] = float(-1.0 * a_pl * max_pitch_vel)
                    vel_cmd[0] = 0.0
                else:
                    vel_cmd[3] = 0.0
                    vel_cmd[0] = float(a_pl * max_lift_vel)
            elif a_pl < 0.0:
                # Command is DOWN (pitch down is positive position)
                if curr_pitch < pitch_split_limit_rad:
                    vel_cmd[3] = float(-1.0 * a_pl * max_pitch_vel)
                    vel_cmd[0] = 0.0
                else:
                    vel_cmd[3] = 0.0
                    vel_cmd[0] = float(a_pl * max_lift_vel)
            else:
                vel_cmd[3] = 0.0
                vel_cmd[0] = 0.0
                
            # 3. Combined Yaw Turn Logic (sequential)
            curr_yaw = self.get_joint_position("wrist_yaw_joint", 0.0)
            if a_yt > 0.0:
                # Command is Left / CCW (yaw left is negative position)
                if curr_yaw > -yaw_split_limit_rad:
                    vel_cmd[4] = float(-1.0 * a_yt * max_yaw_vel)
                    twist.angular.z = 0.0
                else:
                    vel_cmd[4] = 0.0
                    twist.angular.z = float(-1.0 * a_yt * max_angular_vel)
            elif a_yt < 0.0:
                # Command is Right / CW (yaw right is positive position)
                if curr_yaw < yaw_split_limit_rad:
                    vel_cmd[4] = float(-1.0 * a_yt * max_yaw_vel)
                    twist.angular.z = 0.0
                else:
                    vel_cmd[4] = 0.0
                    twist.angular.z = float(-1.0 * a_yt * max_angular_vel)
            else:
                vel_cmd[4] = 0.0
                twist.angular.z = 0.0

        # Arm and Wrist Roll always move normally
        vel_cmd[1] = float(a_arm * max_arm_vel)
        vel_cmd[2] = float(-1.0 * a_roll * max_roll_vel)
        
        # Publish Twist (base command)
        self.base_pub.publish(twist)
        
        # Publish Joint Velocities
        vel_msg = JointState()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.name = ["lift", "arm", "wrist_roll", "wrist_pitch", "wrist_yaw"]
        vel_msg.velocity = vel_cmd
        self.vel_pub.publish(vel_msg)
        
        # Publish Gripper Position (always position controlled)
        pos_msg = JointState()
        pos_msg.header.stamp = self.get_clock().now().to_msg()
        pos_msg.name = ["stretch_gripper"]
        pos_msg.position = [self.map_range(a_grip, "gripper")]
        self.pos_pub.publish(pos_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StretchCombinedControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
