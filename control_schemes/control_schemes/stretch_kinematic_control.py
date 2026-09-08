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

from stretch4_kinematics.state import StretchJointPositions
from stretch4_kinematics.kinematic_models import ToolFrameKinematics

class StretchKinematicControlNode(ControlSchemeNode):
    def __init__(self):
        # 1. Parse reference frames from sys.argv or default
        self._reference_frames = ["base_link", "grasp_center_link"]
        for arg in sys.argv:
            if "reference_frames:=" in arg:
                val = arg.split(":=")[1]
                val = val.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
                self._reference_frames = [f.strip() for f in val.split(",") if f.strip()]
        
        # Define 7 axes
        self.axis_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]
        
        # One button per frame
        self.button_names = [f"switch to {f}" for f in self._reference_frames]
        
        node_name = "stretch_kinematic_control"
        super().__init__(node_name, self.axis_names, self.button_names)
        
        # Declare parameters
        self.declare_parameter("reference_frames", self._reference_frames)
        self.declare_parameter("target_frame", "grasp_center_link")
        self.declare_parameter("max_linear_vel", 0.15)
        self.declare_parameter("max_angular_vel", 0.05)
        self.declare_parameter("limit.gripper.lower", 0.0)
        self.declare_parameter("limit.gripper.upper", 0.15)
        
        self.declare_parameter("max_joint_vel.lift", 0.10)
        self.declare_parameter("max_joint_vel.arm", 0.15)
        self.declare_parameter("max_joint_vel.wrist_yaw", 0.08)
        self.declare_parameter("max_joint_vel.wrist_pitch", 0.08)
        self.declare_parameter("max_joint_vel.wrist_roll", 0.08)
        
        # Kinematics solver from stretch4_kinematics
        self.kinematics_solver = ToolFrameKinematics()
        self._target_frame = self.get_parameter("target_frame").value
        self._current_ref_frame = self._reference_frames[0]
        
        # Weighted damped pseudo-inverse IK parameters
        # Index ordering in q: [base_x, base_y, base_theta, lift, arm, wrist_yaw, wrist_pitch, wrist_roll]
        self.jacobian_weights = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 15.0, 15.0, 15.0])
        self.damping_coefficient = 1e-3
        
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
        
        self.get_logger().info(f"Initialized {node_name} with target frame '{self._target_frame}' and active ref frame '{self._current_ref_frame}'")

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

    def map_range(self, val):
        lower = self.get_parameter("limit.gripper.lower").value
        upper = self.get_parameter("limit.gripper.upper").value
        return (val + 1.0) / 2.0 * (upper - lower) + lower

    def handle_joy(self, axes: list, buttons: list):
        if len(axes) < 7:
            self.get_logger().warn(f"Expected at least 7 axes, got {len(axes)}")
            return
            
        # 1. Switch reference frame if any button is pressed
        for i, val in enumerate(buttons):
            if val == 1 and i < len(self._reference_frames):
                new_frame = self._reference_frames[i]
                if self._current_ref_frame != new_frame:
                    self._current_ref_frame = new_frame
                    self.get_logger().info(f"Switched control reference frame to: {self._current_ref_frame}")
                    
        # Ensure driver is configured
        self.update_driver_joint_modes()
        
        # 2. Get current robot positions from /joint_states
        q_state = self.parse_joint_positions()
        if q_state is None:
            self.get_logger().warn("Waiting for joint_states...", throttle_duration_sec=2.0)
            return
            
        # 3. Compute 6-DOF Kinematic Command
        # Desired twist: [vx, vy, vz, w_roll, w_pitch, w_yaw] in current reference frame
        max_linear_vel = self.get_parameter("max_linear_vel").value
        max_angular_vel = self.get_parameter("max_angular_vel").value
        
        v_desired = np.array([
            float(axes[0] * max_linear_vel),  # X
            float(axes[1] * max_linear_vel),  # Y
            float(axes[2] * max_linear_vel),  # Z
            float(axes[3] * max_angular_vel), # Roll
            float(axes[4] * max_angular_vel), # Pitch
            float(axes[5] * max_angular_vel), # Yaw
        ])
        
        # Resolve IK relative to current reference frame utilizing Pinocchio
        q_pin = q_state.to_pinocchio_q()
        pin.computeJointJacobians(self.kinematics_solver.model, self.kinematics_solver.data, q_pin)
        pin.framesForwardKinematics(self.kinematics_solver.model, self.kinematics_solver.data, q_pin)
        
        frame_id_T = self.kinematics_solver.model.getFrameId(self._target_frame)
        frame_id_F = self.kinematics_solver.model.getFrameId(self._current_ref_frame)
        
        oMF = self.kinematics_solver.data.oMf[frame_id_F]
        R_F = oMF.rotation
        
        # Frame Jacobian in LOCAL_WORLD_ALIGNED frame (referenced at T but aligned with world axes)
        J_LWA = pin.getFrameJacobian(
            self.kinematics_solver.model,
            self.kinematics_solver.data,
            frame_id_T,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        
        # Rotate the Jacobian to align with the orientation of reference frame F
        R_block = np.zeros((6, 6))
        R_block[:3, :3] = R_F.T
        R_block[3:, 3:] = R_F.T
        J_F = R_block @ J_LWA
        
        # Solve weighted least squares with Levenberg-Marquardt damping
        W_pinv = np.diag(1.0 / self.jacobian_weights)
        J_W_JT = J_F @ W_pinv @ J_F.T
        J_W_JT_damped = J_W_JT + self.damping_coefficient * np.eye(6)
        J_pinv = W_pinv @ J_F.T @ np.linalg.inv(J_W_JT_damped)
        
        dq = J_pinv @ v_desired
        
        # Direction-preserving joint velocity scaling for safety
        scale = 1.0
        max_lift = self.get_parameter("max_joint_vel.lift").value
        max_arm = self.get_parameter("max_joint_vel.arm").value
        max_yaw = self.get_parameter("max_joint_vel.wrist_yaw").value
        max_pitch = self.get_parameter("max_joint_vel.wrist_pitch").value
        max_roll = self.get_parameter("max_joint_vel.wrist_roll").value
        
        if abs(dq[3]) > max_lift:
            scale = min(scale, max_lift / abs(dq[3]))
        if abs(dq[4]) > max_arm:
            scale = min(scale, max_arm / abs(dq[4]))
        if abs(dq[5]) > max_yaw:
            scale = min(scale, max_yaw / abs(dq[5]))
        if abs(dq[6]) > max_pitch:
            scale = min(scale, max_pitch / abs(dq[6]))
        if abs(dq[7]) > max_roll:
            scale = min(scale, max_roll / abs(dq[7]))
            
        if scale < 1.0:
            self.get_logger().info(f"Scaling joint velocities by {scale:.3f} to prevent rapid wrist movement.", throttle_duration_sec=1.0)
            dq = dq * scale
        
        # 4. Publish commands
        # Base Velocities (always velocity via /cmd_vel)
        twist = Twist()
        twist.linear.x = float(dq[0])
        twist.linear.y = float(dq[1])
        twist.angular.z = float(dq[2])
        self.base_pub.publish(twist)
        
        # Manipulator Joint Velocities
        vel_msg = JointState()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.name = ["lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll"]
        vel_msg.velocity = [
            float(dq[3]),  # Lift
            float(dq[4]),  # Arm
            float(dq[5]),  # Wrist Yaw
            float(dq[6]),  # Wrist Pitch
            float(dq[7])   # Wrist Roll
        ]
        self.vel_pub.publish(vel_msg)
        
        # Gripper Command (always position-controlled via /joint_position_cmd)
        pos_msg = JointState()
        pos_msg.header.stamp = self.get_clock().now().to_msg()
        pos_msg.name = ["stretch_gripper"]
        pos_msg.position = [float(self.map_range(axes[6]))]
        self.pos_pub.publish(pos_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StretchKinematicControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
