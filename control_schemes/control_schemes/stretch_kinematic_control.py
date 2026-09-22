#!/usr/bin/env python3
"""StretchKinematicControlNode: 6-DOF Cartesian velocity control via differential IK.

Solves for joint velocities toward a target frame using SixDofDifferentialIK,
with axes expressed relative to an operator-selectable reference frame via TF2.
"""

import sys
import numpy as np
import rclpy
from rclpy.time import Time
from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from multi_teleop.base import ControlSchemeNode

from stretch4_kinematics.state import StretchJointPositions, Stretch4IKModes
from stretch4_kinematics.kinematic_models import ToolFrameKinematics

from control_schemes.differential_ik_6dof import SixDofDifferentialIK, quaternion_to_rotation_matrix

BASE_MODES_BY_NAME = {
    "base_fixed": Stretch4IKModes.BASE_FIXED,
    "base_rotate": Stretch4IKModes.BASE_ROTATE,
    "base_planar": Stretch4IKModes.BASE_PLANAR,
}


class StretchKinematicControlNode(ControlSchemeNode):
    """6-DOF differential-IK control scheme with a switchable reference frame.

    Axes command a 6D twist of target_frame, expressed in whichever reference
    frame is currently active (switched via per-frame buttons).
    """

    def __init__(self):
        """Sets up reference frames, IK solver, TF2, driver discovery, and publishers."""
        # reference_frames must be known before super().__init__() (it determines button_names),
        # but ROS2 parameters aren't available until after Node.__init__ runs -- so bootstrap it
        # from a CLI arg here, then re-declare it as a proper parameter below for introspection.
        self._reference_frames = self._parse_reference_frames_arg()

        self.axis_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]
        self.button_names = [f"switch to {f}" for f in self._reference_frames]

        node_name = "stretch_kinematic_control"
        super().__init__(node_name, self.axis_names, self.button_names)

        self.declare_parameter("reference_frames", self._reference_frames)
        self.declare_parameter("target_frame", "grasp_center_link")
        self.declare_parameter("base_mode", "base_rotate")
        self.declare_parameter("max_linear_vel", 0.15)
        self.declare_parameter("max_angular_vel", 0.2)
        # Below this magnitude, a joint velocity is snapped to exactly 0.0 before publishing.
        # The IK's pseudoinverse solve leaves tiny nonzero residuals on "uninvolved" joints,
        # but the driver treats any nonzero value as "moving" regardless of magnitude.
        self.declare_parameter("zero_velocity_epsilon", 1e-4)
        # After this many consecutive all-zero velocity commands, stop publishing base/joint
        # velocity messages entirely (rather than streaming zeros forever) until a nonzero
        # command comes in again -- a few repeats are still sent first so a genuine "stop" is
        # reliably delivered even if a message or two is dropped.
        self.declare_parameter("zero_velocity_repeat_count", 3)

        # Roughly half of each joint's true physical max speed (lift/arm/base maxes are 0.5/0.7/0.6
        # m/s per the datasheet; wrist max is 12 rad/s per robot_params_SE4.py) -- these clamps
        # exist to bound the differential IK's redundancy-resolution output, not to be the primary
        # safety layer (guarded contact / runstop / collision avoidance already provide that).
        self.declare_parameter("max_joint_vel.base_x", 0.25)
        self.declare_parameter("max_joint_vel.base_y", 0.25)
        self.declare_parameter("max_joint_vel.base_theta", 0.4)
        self.declare_parameter("max_joint_vel.lift", 0.25)
        self.declare_parameter("max_joint_vel.arm", 0.35)
        self.declare_parameter("max_joint_vel.wrist_yaw", 1.0)
        self.declare_parameter("max_joint_vel.wrist_pitch", 1.0)
        self.declare_parameter("max_joint_vel.wrist_roll", 1.0)
        self.declare_parameter("max_joint_vel.gripper", 1.0)

        self._target_frame = self.get_parameter("target_frame").value
        # No button pressed yet: active frame defaults to target_frame itself, which resolves to
        # an identity rotation via TF2 (a frame looked up against itself), so no remap happens.
        self._active_frame = self._target_frame
        self._prev_buttons = [0] * len(self.button_names)

        self._last_joint_state = None
        self._last_base_theta = 0.0
        self._zero_vel_streak = 0

        self.kinematics_solver = ToolFrameKinematics()
        self.six_dof_ik = SixDofDifferentialIK(self.kinematics_solver)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.joint_states_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_states_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/wheel_odom", self._odom_callback, 10
        )

        self.vel_pub = self.create_publisher(JointState, "/joint_velocity_cmd", 10)
        self.base_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Driver interaction state
        self.driver_node = None
        self.driver_namespace = None
        self.param_client = None
        self._driver_modes_configured = False

        self.discover_driver_node()
        self.discovery_timer = self.create_timer(1.0, self.discover_driver_node)

        self.get_logger().info(
            f"Initialized {node_name} with target frame '{self._target_frame}', "
            f"reference frames {self._reference_frames}, base_mode "
            f"'{self.get_parameter('base_mode').value}'"
        )

    @staticmethod
    def _parse_reference_frames_arg():
        """Reads a comma-separated reference_frames:= CLI override, or a built-in default.

        Anchored to how rclpy actually passes parameters on the CLI
        (``--ros-args -p reference_frames:=...`` / ``--param reference_frames:=...``):
        only the ``--ros-args`` section of argv is scanned, and only a ``-p``/``--param``
        value token is matched, so this can't accidentally match the substring
        "reference_frames:=" appearing inside some unrelated argument's value.
        """
        default = ["grasp_center_link", "base_link", "camera_center_link"]
        argv = sys.argv
        if "--ros-args" not in argv:
            return default
        ros_args = argv[argv.index("--ros-args") + 1:]
        for i, arg in enumerate(ros_args):
            if arg in ("-p", "--param") and i + 1 < len(ros_args) \
                    and ros_args[i + 1].startswith("reference_frames:="):
                val = ros_args[i + 1].split(":=", 1)[1]
                val = val.strip("[]").replace("'", "").replace('"', "")
                return [f.strip() for f in val.split(",") if f.strip()]
        return default

    def _joint_states_callback(self, msg: JointState):
        self._last_joint_state = msg

    def _odom_callback(self, msg: Odometry):
        """Extracts and caches the base heading (theta) from wheel odometry."""
        # Only the base's heading matters for the Jacobian's rotation-dependent columns;
        # absolute base_x/base_y position doesn't affect an instantaneous velocity solve.
        q = msg.pose.pose.orientation
        R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        self._last_base_theta = float(np.arctan2(R[1, 0], R[0, 0]))

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
            string_value="velocity"
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

    def _get_joint_positions(self) -> StretchJointPositions:
        """Builds a StretchJointPositions from the last /joint_states message, or None."""
        if self._last_joint_state is None:
            return None

        js = self._last_joint_state

        def get_pos(name, default=0.0):
            try:
                return js.position[js.name.index(name)]
            except ValueError:
                return default

        arm = sum(get_pos(f"arm_l{i}_joint") for i in range(1, 5))

        return StretchJointPositions(
            base_x=0.0,
            base_y=0.0,
            base_theta=self._last_base_theta,
            lift=get_pos("lift_joint"),
            arm=arm,
            wrist_yaw=get_pos("wrist_yaw_joint"),
            wrist_pitch=get_pos("wrist_pitch_joint"),
            wrist_roll=get_pos("wrist_roll_joint"),
        )

    def _get_base_mode(self) -> Stretch4IKModes:
        """Resolves the base_mode parameter to a Stretch4IKModes, defaulting to BASE_ROTATE."""
        name = self.get_parameter("base_mode").value
        try:
            return BASE_MODES_BY_NAME[name]
        except KeyError:
            self.get_logger().error(
                f"Invalid base_mode '{name}', must be one of {list(BASE_MODES_BY_NAME)}; defaulting to base_rotate",
                throttle_duration_sec=5.0,
            )
            return Stretch4IKModes.BASE_ROTATE

    def _zero_small_velocities(self, v):
        """Snaps each joint velocity below zero_velocity_epsilon to exactly 0.0, in place."""
        epsilon = self.get_parameter("zero_velocity_epsilon").value
        for name in v.get_joint_names():
            if abs(getattr(v, name)) < epsilon:
                setattr(v, name, 0.0)
        return v

    def _scale_for_safety(self, v):
        """Uniformly scales v down (if needed) so no joint exceeds its max_joint_vel limit."""
        limits = {
            name: self.get_parameter(f"max_joint_vel.{name}").value
            for name in v.get_joint_names()
        }
        scale = 1.0
        for name, limit in limits.items():
            val = abs(getattr(v, name))
            if val > limit:
                scale = min(scale, limit / val)
        if scale < 1.0:
            self.get_logger().info(f"Scaling joint velocities by {scale:.3f} to stay within safety limits.", throttle_duration_sec=1.0)
            for name in limits:
                setattr(v, name, getattr(v, name) * scale)
        return v

    def _publish_velocity_commands(self, v, gripper_vel) -> bool:
        """Returns whether base/joint velocity commands should be published this cycle.

        After a few consecutive all-zero commands, stops publishing entirely rather than
        streaming zeros forever -- avoids spamming the driver with redundant stop commands
        while it's already idle. A nonzero command immediately resumes normal publishing.
        """
        is_zero = all(getattr(v, name) == 0.0 for name in v.get_joint_names()) and gripper_vel == 0.0
        if not is_zero:
            self._zero_vel_streak = 0
            return True

        self._zero_vel_streak += 1
        return self._zero_vel_streak <= self.get_parameter("zero_velocity_repeat_count").value

    def handle_joy(self, axes: list, buttons: list):
        """Solves differential IK for the commanded twist and publishes velocity commands.

        Args:
            axes (list): 7 values: X/Y/Z, Roll/Pitch/Yaw, Gripper (reference-frame-local).
            buttons (list): One button per reference frame; rising edge switches active frame.
        """
        if len(axes) < 7:
            self.get_logger().warn(f"Expected at least 7 axes, got {len(axes)}")
            return

        self.update_driver_joint_modes()

        # Persistent switch-on-press: a frame stays active until a different frame's button
        # is pressed, so an operator doesn't need to hold a button down during motion.
        for i, pressed in enumerate(buttons):
            if pressed and not self._prev_buttons[i] and i < len(self._reference_frames):
                self._active_frame = self._reference_frames[i]
                self.get_logger().info(f"Switched active reference frame to '{self._active_frame}'")
        self._prev_buttons = list(buttons)

        q_state = self._get_joint_positions()
        if q_state is None:
            self.get_logger().warn("Waiting for /joint_states...", throttle_duration_sec=2.0)
            return

        try:
            transform = self.tf_buffer.lookup_transform(self._target_frame, self._active_frame, Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"Could not look up transform from '{self._active_frame}' to '{self._target_frame}': {e}",
                throttle_duration_sec=2.0,
            )
            return

        # Only the rotation is used -- this is a direction remap, not a displacement.
        rot = transform.transform.rotation
        R = quaternion_to_rotation_matrix(rot.x, rot.y, rot.z, rot.w)

        max_linear_vel = self.get_parameter("max_linear_vel").value
        max_angular_vel = self.get_parameter("max_angular_vel").value

        v_lin_ref = np.array(axes[0:3], dtype=float) * max_linear_vel
        v_ang_ref = np.array(axes[3:6], dtype=float) * max_angular_vel
        v_desired = np.concatenate([R @ v_lin_ref, R @ v_ang_ref])

        v = self.six_dof_ik.differential_ik_6dof(
            q_state, self._target_frame, v_desired, self._get_base_mode()
        )
        v = self._scale_for_safety(v)
        v = self._zero_small_velocities(v)

        gripper_vel = axes[6] * self.get_parameter("max_joint_vel.gripper").value
        if abs(gripper_vel) < self.get_parameter("zero_velocity_epsilon").value:
            gripper_vel = 0.0

        if self._publish_velocity_commands(v, gripper_vel):
            twist = Twist()
            twist.linear.x = float(v.base_x)
            twist.linear.y = float(v.base_y)
            twist.angular.z = float(v.base_theta)
            self.base_pub.publish(twist)

            vel_msg = JointState()
            vel_msg.header.stamp = self.get_clock().now().to_msg()
            vel_msg.name = ["lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "stretch_gripper"]
            vel_msg.velocity = [
                float(v.lift), float(v.arm), float(v.wrist_yaw), float(v.wrist_pitch), float(v.wrist_roll),
                float(gripper_vel)
            ]
            self.vel_pub.publish(vel_msg)


def main(args=None):
    """Entry point: initializes rclpy and spins a StretchKinematicControlNode."""
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
