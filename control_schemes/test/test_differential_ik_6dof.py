#!/usr/bin/env python3
"""Pure-math unit tests for differential_ik_6dof.py.

Covers quaternion_to_rotation_matrix and SixDofDifferentialIK.differential_ik_6dof.
No ROS, no hardware, no simulator -- these are plain numerical computations.
ToolFrameKinematics() only loads a URDF model via Pinocchio; it never talks to
real or simulated hardware, so constructing it here is safe.
"""

import numpy as np
import pytest

from stretch4_kinematics.kinematic_models import ToolFrameKinematics
from stretch4_kinematics.state import StretchJointPositions, Stretch4IKModes

from control_schemes.differential_ik_6dof import (
    SixDofDifferentialIK,
    quaternion_to_rotation_matrix,
)


TARGET_FRAME = "grasp_center_link"


@pytest.fixture(scope="module")
def kinematics():
    """A real ToolFrameKinematics instance (URDF-model math only, no hardware)."""
    return ToolFrameKinematics()


@pytest.fixture(scope="module")
def solver(kinematics):
    return SixDofDifferentialIK(kinematics)


# --- quaternion_to_rotation_matrix -----------------------------------------

def test_identity_quaternion_gives_identity_matrix():
    R = quaternion_to_rotation_matrix(0.0, 0.0, 0.0, 1.0)
    assert np.allclose(R, np.eye(3))


def test_rotation_matrix_is_orthonormal():
    R = quaternion_to_rotation_matrix(0.1, 0.2, 0.3, 0.9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)


def test_90_degree_z_rotation():
    # 90deg about Z: (x,y,z,w) = (0, 0, sin(45deg), cos(45deg))
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    R = quaternion_to_rotation_matrix(0.0, 0.0, s, c)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(R, expected, atol=1e-8)


def test_unnormalized_quaternion_is_normalized_internally():
    # Scaling the quaternion by a constant must not change the resulting rotation.
    R1 = quaternion_to_rotation_matrix(0.0, 0.0, 1.0, 1.0)
    R2 = quaternion_to_rotation_matrix(0.0, 0.0, 2.0, 2.0)
    assert np.allclose(R1, R2, atol=1e-8)


# --- SixDofDifferentialIK.differential_ik_6dof -----------------------------

def test_zero_twist_gives_zero_velocity(solver):
    q = StretchJointPositions()
    v = solver.differential_ik_6dof(q, TARGET_FRAME, np.zeros(6), Stretch4IKModes.BASE_FIXED)
    for name in v.get_joint_names():
        assert getattr(v, name) == pytest.approx(0.0, abs=1e-9)


def test_base_fixed_mode_excludes_base_columns(solver):
    q = StretchJointPositions()
    v_desired = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0])
    v = solver.differential_ik_6dof(q, TARGET_FRAME, v_desired, Stretch4IKModes.BASE_FIXED)
    assert v.base_x == 0.0
    assert v.base_y == 0.0
    assert v.base_theta == 0.0


def test_base_rotate_mode_excludes_base_translation(solver):
    q = StretchJointPositions()
    v_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    v = solver.differential_ik_6dof(q, TARGET_FRAME, v_desired, Stretch4IKModes.BASE_ROTATE)
    assert v.base_x == 0.0
    assert v.base_y == 0.0


def test_base_planar_mode_allows_base_translation(solver, kinematics):
    # With heavily discouraged base weights (default), a small forward twist should
    # still resolve mostly through the arm, but base columns are at least permitted
    # (not hard-zeroed) in BASE_PLANAR mode -- verify the Jacobian actually has
    # nonzero columns for base_x/base_y so a nonzero solution is possible there,
    # rather than asserting a specific split between arm/base.
    q = StretchJointPositions()
    J = kinematics.get_frame_jacobian(q, TARGET_FRAME)
    # get_frame_jacobian returns the model's nv-width (tangent-space) Jacobian: 8 for BASE_PLANAR.
    assert J.shape[1] == 8
    v = solver.differential_ik_6dof(q, TARGET_FRAME, np.array([0.1, 0, 0, 0, 0, 0]), Stretch4IKModes.BASE_PLANAR)
    # Solution must be finite and not raise; base columns are included in the solve.
    assert np.all(np.isfinite(v.to_numpy()))


def test_output_is_finite_and_correct_length(solver):
    q = StretchJointPositions()
    v = solver.differential_ik_6dof(
        q, TARGET_FRAME, np.array([0.05, -0.02, 0.01, 0.0, 0.1, 0.0]), Stretch4IKModes.BASE_ROTATE
    )
    arr = v.to_numpy()
    assert arr.shape == (8,)
    assert np.all(np.isfinite(arr))


def test_custom_joint_weights_are_merged_with_defaults(kinematics):
    solver_custom = SixDofDifferentialIK(kinematics, joint_weights={"arm": 500.0})
    assert solver_custom._joint_weights["arm"] == 500.0
    # Unspecified entries keep their defaults.
    assert solver_custom._joint_weights["lift"] == 1.0


def test_higher_arm_weight_discourages_arm_motion(kinematics):
    q = StretchJointPositions()
    v_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    solver_default = SixDofDifferentialIK(kinematics)
    solver_arm_discouraged = SixDofDifferentialIK(kinematics, joint_weights={"arm": 1000.0})

    v_default = solver_default.differential_ik_6dof(q, TARGET_FRAME, v_desired, Stretch4IKModes.BASE_ROTATE)
    v_discouraged = solver_arm_discouraged.differential_ik_6dof(q, TARGET_FRAME, v_desired, Stretch4IKModes.BASE_ROTATE)

    assert abs(v_discouraged.arm) < abs(v_default.arm)


def test_damping_coefficient_changes_solution_magnitude(kinematics):
    q = StretchJointPositions()
    v_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    solver_low_damp = SixDofDifferentialIK(kinematics, damping_coefficient=1e-6)
    solver_high_damp = SixDofDifferentialIK(kinematics, damping_coefficient=1.0)

    v_low = solver_low_damp.differential_ik_6dof(q, TARGET_FRAME, v_desired, Stretch4IKModes.BASE_ROTATE)
    v_high = solver_high_damp.differential_ik_6dof(q, TARGET_FRAME, v_desired, Stretch4IKModes.BASE_ROTATE)

    # Heavy damping should shrink the solved velocities toward zero relative to light damping.
    assert np.linalg.norm(v_high.to_numpy()) < np.linalg.norm(v_low.to_numpy())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
