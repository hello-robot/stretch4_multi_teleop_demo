"""Full 6-DOF differential IK math (translation + rotation), plus a quaternion helper.

Pure numerical helpers -- no ROS, no Node subclass, no hardware/sim I/O.
"""

import numpy as np

from stretch4_kinematics.state import StretchJointPositions, StretchJointVelocities, Stretch4IKModes
from stretch4_kinematics.kinematic_models import ToolFrameKinematics


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """
    Converts a quaternion (x, y, z, w) to a 3x3 rotation matrix, without depending on Pinocchio.

    Args:
        x, y, z, w (float): Quaternion components (scalar-last convention).

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    n = np.linalg.norm([x, y, z, w])
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


class SixDofDifferentialIK:
    """
    Full 6-DOF (translation + rotation) differential IK for an arbitrary target frame,
    with a configurable base-handling mode.

    Mirrors stretch4_kinematics.kinematic_models.tool_frame_kinematics.ToolFrameKinematics.differential_ik,
    but solves all 6 twist components at once instead of translation-only (plus a hardcoded wrist-yaw
    compensation), over a base-mode-selectable actuated joint set. This lives here rather than in
    stretch4_kinematics because the weighting/masking scheme is still experimental; it's written to
    closely mirror stretch4_kinematics' conventions (StretchJointPositions/StretchJointVelocities/
    Stretch4IKModes reuse, ToolFrameKinematics.get_frame_jacobian for all Pinocchio access, no direct
    `import pinocchio` here) so it's a natural drop-in replacement if this capability gets upstreamed.
    """

    # Column indices into the 8-wide StretchJointVelocities tangent space
    # (order: base_x, base_y, base_theta, lift, arm, wrist_yaw, wrist_pitch, wrist_roll).
    _BASE_MODE_COLUMNS = {
        Stretch4IKModes.BASE_FIXED:  [3, 4, 5, 6, 7],
        Stretch4IKModes.BASE_ROTATE: [2, 3, 4, 5, 6, 7],
        Stretch4IKModes.BASE_PLANAR: [0, 1, 2, 3, 4, 5, 6, 7],
    }

    _DEFAULT_JOINT_WEIGHTS = {
        "base_x": 50.0,
        "base_y": 50.0,
        "base_theta": 5.0,
        "lift": 1.0,
        "arm": 1.0,
        "wrist_yaw": 1.0,
        "wrist_pitch": 1.0,
        "wrist_roll": 1.0,
    }

    def __init__(
        self,
        kinematics: ToolFrameKinematics,
        joint_weights: dict = None,
        damping_coefficient: float = 1e-4,
    ):
        """
        Args:
            kinematics (ToolFrameKinematics): The kinematics model to solve against.
            joint_weights (dict, optional): Overrides for the per-joint pseudoinverse weights
                (higher = more discouraged), keyed by StretchJointVelocities field name.
            damping_coefficient (float): Levenberg-Marquardt damping factor for the pseudoinverse.
        """
        self.kinematics = kinematics
        self._joint_names = StretchJointVelocities().get_joint_names()
        self._joint_weights = dict(self._DEFAULT_JOINT_WEIGHTS)
        if joint_weights:
            self._joint_weights.update(joint_weights)
        self._damping_coefficient = damping_coefficient

    def differential_ik_6dof(
        self,
        q: StretchJointPositions,
        target_frame: str,
        v_desired: np.ndarray,
        base_mode: Stretch4IKModes,
    ) -> StretchJointVelocities:
        """
        Computes the joint velocities required to achieve a desired 6D twist of target_frame,
        expressed in target_frame's own LOCAL coordinates.

        Args:
            q (StretchJointPositions): The robot's joint configuration.
            target_frame (str): The name of the frame to solve the velocity relationship for.
            v_desired (np.ndarray): The desired 6D twist of target_frame, in target_frame LOCAL
                coordinates ([vx, vy, vz, wx, wy, wz]).
            base_mode (Stretch4IKModes): Which base DOFs are allowed to participate in the solve.

        Returns:
            StretchJointVelocities: Joint velocities required to achieve the target velocity
                (zero for any joint excluded by base_mode).
        """
        col_idx = self._BASE_MODE_COLUMNS[base_mode]

        J_full = self.kinematics.get_frame_jacobian(q, target_frame)
        J = J_full[:, col_idx]

        weights = np.array([self._joint_weights[self._joint_names[i]] for i in col_idx])
        W_pinv = np.diag(1.0 / weights)
        J_W_JT = J @ W_pinv @ J.T
        J_W_JT_damped = J_W_JT + self._damping_coefficient * np.eye(6)
        J_pinv = W_pinv @ J.T @ np.linalg.inv(J_W_JT_damped)
        dq = J_pinv @ v_desired

        v_full = np.zeros(len(self._joint_names))
        v_full[col_idx] = dq
        return StretchJointVelocities.from_numpy(v_full)
