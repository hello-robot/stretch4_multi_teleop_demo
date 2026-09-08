#!/usr/bin/env python3

import sys
import numpy as np
import pinocchio as pin
import rclpy
from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from multi_teleop.base import ControlSchemeNode

from stretch4_kinematics.state import StretchJointPositions, StretchJointVelocities
from stretch4_kinematics.kinematic_models import ToolFrameKinematics

class StretchSingleSwitchControlNode(ControlSchemeNode):
    def __init__(self):
        # 0 Axes, 1 Switch Button
        self.axis_names = []
        self.button_names = ["Switch Button"]
        
        node_name = "stretch_single_switch_control"
        super().__init__(node_name, self.axis_names, self.button_names)
        
        # Declare parameters
        self.declare_parameter("exploration_speed.wrist_yaw", 0.3)      # rad/s
        self.declare_parameter("exploration_speed.wrist_pitch", 0.3)    # rad/s
        self.declare_parameter("exploration_speed.linear", 0.05)         # m/s
        
        self.declare_parameter("limit.lift.lower", 0.18)
        self.declare_parameter("limit.lift.upper", 1.18)
        self.declare_parameter("limit.arm.lower", 0.01)
        self.declare_parameter("limit.arm.upper", 0.5)
        self.declare_parameter("limit.wrist_yaw.lower", -1.047)
        self.declare_parameter("limit.wrist_yaw.upper", 1.047)
        self.declare_parameter("limit.wrist_pitch.lower", -1.047)
        self.declare_parameter("limit.wrist_pitch.upper", 1.047)
        
        self.declare_parameter("max_joint_vel.lift", 0.10)
        self.declare_parameter("max_joint_vel.arm", 0.15)
        self.declare_parameter("max_joint_vel.wrist_yaw", 0.25)
        self.declare_parameter("max_joint_vel.wrist_pitch", 0.25)
        self.declare_parameter("max_joint_vel.wrist_roll", 0.25)
        
        # State estimation and kinematics solver from library
        self.kinematics_solver = ToolFrameKinematics()
        
        # Control State
        self._state = 0  # 0: Stopped, 1: Panning, 2: Tilting, 3: Translating, 4: Moving
        self._exploration_direction = 1  # +1 or -1
        self._selected_v_x = 0.0
        self._last_button_state = False
        
        # Timing helper for State 3 oscillation
        self._state_start_time = self.get_clock().now()
        
        # ROS State and Publishers
        self.last_joint_state = None
        self.joint_states_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            10
        )
        
        self.pos_pub = self.create_publisher(JointState, "/joint_position_cmd", 10)
        self.vel_pub = self.create_publisher(JointState, "/joint_velocity_cmd", 10)
        self.base_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # Driver interaction state
        self.driver_node = None
        self.driver_namespace = None
        self.param_client = None
        self._driver_modes_configured = False
        
        # Locate driver on startup
        self.discover_driver_node()
        self.discovery_timer = self.create_timer(1.0, self.discover_driver_node)
        
        # Periodic control execution timer (20Hz)
        self.control_timer = self.create_timer(0.05, self.timer_callback)
        
        self.get_logger().info(f"Initialized {node_name} (Single Switch Assistive Controller). State: 0 (Stopped)")

    def joint_states_callback(self, msg: JointState):
        self.last_joint_state = msg

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

    def update_driver_joint_modes(self):
        """Asynchronously requests the driver to update joint modes (velocity for manipulator, position for gripper)."""
        if self._driver_modes_configured:
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
                string_value="velocity"
            )
            params.append(p)
            
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
                    self.get_logger().info("Successfully initialized driver joint modes for Kinematic Control!")
                    self._driver_modes_configured = True
                else:
                    self.get_logger().error("Failed to set joint modes on driver")
            except Exception as e:
                self.get_logger().error(f"Error calling driver set_parameters: {e}")
                
        future.add_done_callback(done_cb)

    def parse_joint_positions(self) -> StretchJointPositions:
        if self.last_joint_state is None:
            return None
            
        js = self.last_joint_state
        
        def get_pos(name, default=0.0):
            try:
                idx = js.name.index(name)
                return js.position[idx]
            except ValueError:
                return default
                
        lift_pos = get_pos("lift_joint")
        arm_pos = sum(get_pos(f"arm_l{i}_joint") for i in [1, 2, 3, 4])
        
        wrist_yaw = get_pos("wrist_yaw_joint")
        wrist_pitch = get_pos("wrist_pitch_joint")
        wrist_roll = get_pos("wrist_roll_joint")
        
        return StretchJointPositions(
            base_x=0.0,
            base_y=0.0,
            base_theta=0.0,
            lift=lift_pos,
            arm=arm_pos,
            wrist_yaw=wrist_yaw,
            wrist_pitch=wrist_pitch,
            wrist_roll=wrist_roll
        )

    def handle_joy(self, axes: list, buttons: list):
        if len(buttons) < 1:
            return
            
        # Detect rising edge of the switch button
        button_val = buttons[0]
        if button_val == 1 and not self._last_button_state:
            self._last_button_state = True
            self.transition_state()
        elif button_val == 0:
            self._last_button_state = False

    def transition_state(self):
        # Cycle through states: 0 -> 1 -> 2 -> 3 -> 4 -> 0
        prev_state = self._state
        self._state = (self._state + 1) % 5
        self._state_start_time = self.get_clock().now()
        
        self.get_logger().info(f"Assistive Switch: State {prev_state} -> State {self._state}")
        
        if self._state == 0:
            self.get_logger().info("State 0: Stopped.")
            self._selected_v_x = 0.0
            
        elif self._state == 1:
            self.get_logger().info("State 1: Panning (wrist_yaw exploration).")
            self._exploration_direction = 1
            
        elif self._state == 2:
            self.get_logger().info("State 2: Tilting (wrist_pitch exploration).")
            self._exploration_direction = 1
            
        elif self._state == 3:
            self.get_logger().info("State 3: Translating (grasp_center_link X exploration).")
            self._exploration_direction = 1
            
        elif self._state == 4:
            # Capture translation velocity of the transition moment
            max_linear_vel = self.get_parameter("exploration_speed.linear").value
            self._selected_v_x = float(self._exploration_direction * max_linear_vel)
            self.get_logger().info(f"State 4: Moving continuously along X axis with speed {self._selected_v_x:.3f} m/s")

    def timer_callback(self):
        """Continuous execution loop of the assistive state machine."""
        q_state = self.parse_joint_positions()
        if q_state is None:
            return
            
        # Ensure driver joint modes match
        self.update_driver_joint_modes()
        
        # Load speed settings and limits
        max_yaw_vel = self.get_parameter("exploration_speed.wrist_yaw").value
        max_pitch_vel = self.get_parameter("exploration_speed.wrist_pitch").value
        max_linear_vel = self.get_parameter("exploration_speed.linear").value
        
        lift_lower = self.get_parameter("limit.lift.lower").value
        lift_upper = self.get_parameter("limit.lift.upper").value
        arm_lower = self.get_parameter("limit.arm.lower").value
        arm_upper = self.get_parameter("limit.arm.upper").value
        
        twist = Twist()
        vel_msg = JointState()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.name = ["lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll"]
        vel_msg.velocity = [0.0] * 5
        
        pos_msg = JointState()
        pos_msg.header.stamp = self.get_clock().now().to_msg()
        pos_msg.name = ["stretch_gripper"]
        pos_msg.position = [0.0] # Grip centered
        
        if self._state == 0:
            # STOPPED
            self.base_pub.publish(twist)
            self.vel_pub.publish(vel_msg)
            return
            
        elif self._state == 1:
            # PANNING: Oscillate wrist_yaw
            yaw_lower = self.get_parameter("limit.wrist_yaw.lower").value
            yaw_upper = self.get_parameter("limit.wrist_yaw.upper").value
            
            if q_state.wrist_yaw >= yaw_upper - 0.05:
                self._exploration_direction = -1
            elif q_state.wrist_yaw <= yaw_lower + 0.05:
                self._exploration_direction = 1
                
            vel_msg.velocity[2] = float(self._exploration_direction * max_yaw_vel)
            
        elif self._state == 2:
            # TILTING: Oscillate wrist_pitch
            pitch_lower = self.get_parameter("limit.wrist_pitch.lower").value
            pitch_upper = self.get_parameter("limit.wrist_pitch.upper").value
            
            if q_state.wrist_pitch >= pitch_upper - 0.05:
                self._exploration_direction = -1
            elif q_state.wrist_pitch <= pitch_lower + 0.05:
                self._exploration_direction = 1
                
            vel_msg.velocity[3] = float(self._exploration_direction * max_pitch_vel)
            
        elif self._state == 3:
            # TRANSLATING: Oscillate forward/backward in local gripper X
            # Oscillate direction every 3.0 seconds, OR if lift or arm approaches safety limits
            now = self.get_clock().now()
            elapsed_sec = (now - self._state_start_time).nanoseconds / 1e9
            if elapsed_sec > 3.0:
                self._exploration_direction *= -1
                self._state_start_time = now
                self.get_logger().info(f"Timed oscillation: switching direction to {self._exploration_direction}")
            elif q_state.lift >= lift_upper - 0.03 or q_state.arm >= arm_upper - 0.03:
                if self._exploration_direction == 1:
                    self._exploration_direction = -1
                    self._state_start_time = now
                    self.get_logger().info("Limit hit: reversing to backward (-1)")
            elif q_state.lift <= lift_lower + 0.03 or q_state.arm <= arm_lower + 0.03:
                if self._exploration_direction == -1:
                    self._exploration_direction = 1
                    self._state_start_time = now
                    self.get_logger().info("Limit hit: reversing to forward (+1)")
                
            v_x = self._exploration_direction * max_linear_vel
            v_desired = np.array([v_x, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Solve using library differential_ik (fully redone at each 20Hz timestep)
            v_solved = self.kinematics_solver.differential_ik(q_state, "grasp_center_link", v_desired)
            
            twist.linear.x = float(v_solved.base_x)
            twist.linear.y = float(v_solved.base_y)
            twist.angular.z = float(v_solved.base_theta)
            vel_msg.velocity = [
                float(v_solved.lift),
                float(v_solved.arm),
                float(v_solved.wrist_yaw),
                float(v_solved.wrist_pitch),
                float(v_solved.wrist_roll)
            ]
            
        elif self._state == 4:
            # MOVING: Continuous movement in selected direction
            v_x = self._selected_v_x
            self.get_logger().info(f"Selected velocity: {v_x}")
            
            # Safe hold: Stop translation if manipulator hits safety limits
            if (v_x > 0 and (q_state.lift >= lift_upper - 0.02 or q_state.arm >= arm_upper - 0.02)) or \
               (v_x < 0 and (q_state.lift <= lift_lower + 0.02 or q_state.arm <= arm_lower + 0.02)):
                v_x = 0.0
                self.get_logger().warning("At safety limits, setting v to 0.0")
                
            v_desired = np.array([v_x, 0.0, 0.0, 0.0, 0.0, 0.0])
            v_solved = self.kinematics_solver.differential_ik(q_state, "grasp_center_link", v_desired)
            
            twist.linear.x = float(v_solved.base_x)
            twist.linear.y = float(v_solved.base_y)
            twist.angular.z = float(v_solved.base_theta)
            vel_msg.velocity = [
                float(v_solved.lift),
                float(v_solved.arm),
                float(v_solved.wrist_yaw),
                float(v_solved.wrist_pitch),
                float(v_solved.wrist_roll)
            ]
            
        # Direction-preserving joint velocity scaling for safety
        scale = 1.0
        max_lift = self.get_parameter("max_joint_vel.lift").value
        max_arm = self.get_parameter("max_joint_vel.arm").value
        max_yaw = self.get_parameter("max_joint_vel.wrist_yaw").value
        max_pitch = self.get_parameter("max_joint_vel.wrist_pitch").value
        max_roll = self.get_parameter("max_joint_vel.wrist_roll").value
        
        if abs(vel_msg.velocity[0]) > max_lift:
            scale = min(scale, max_lift / abs(vel_msg.velocity[0]))
        if abs(vel_msg.velocity[1]) > max_arm:
            scale = min(scale, max_arm / abs(vel_msg.velocity[1]))
        if abs(vel_msg.velocity[2]) > max_yaw:
            scale = min(scale, max_yaw / abs(vel_msg.velocity[2]))
        if abs(vel_msg.velocity[3]) > max_pitch:
            scale = min(scale, max_pitch / abs(vel_msg.velocity[3]))
        if abs(vel_msg.velocity[4]) > max_roll:
            scale = min(scale, max_roll / abs(vel_msg.velocity[4]))
            
        if scale < 1.0:
            twist.linear.x *= scale
            twist.linear.y *= scale
            twist.angular.z *= scale
            vel_msg.velocity = [v * scale for v in vel_msg.velocity]
            
        # Safe Clamping on joint velocity limits
        if (vel_msg.velocity[0] > 0 and q_state.lift >= lift_upper) or \
           (vel_msg.velocity[0] < 0 and q_state.lift <= lift_lower):
            vel_msg.velocity[0] = 0.0
            
        if (vel_msg.velocity[1] > 0 and q_state.arm >= arm_upper) or \
           (vel_msg.velocity[1] < 0 and q_state.arm <= arm_lower):
            vel_msg.velocity[1] = 0.0
            
        self.base_pub.publish(twist)
        self.vel_pub.publish(vel_msg)
        self.pos_pub.publish(pos_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StretchSingleSwitchControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
