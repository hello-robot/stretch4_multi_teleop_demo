"""Pure-geometry tests for HeadFaceTracker.process_frame's 6DOF estimation.

Constructs HeadFaceTracker via object.__new__ to bypass the full ROS/webcam
node construction (no rclpy context needed here) and feeds it a fake
MediaPipe detection result, so only the landmark -> 6DOF math is exercised.
"""
import math

import numpy as np
import pytest

from teleop_interfaces.head_face_tracker import HeadFaceTracker


class _FakeLandmark:
    """Minimal stand-in for a mediapipe NormalizedLandmark (x, y, z only)."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _FakeDetectionResult:
    def __init__(self, face_landmarks):
        self.face_landmarks = face_landmarks


class _FakeDetector:
    def __init__(self, result):
        self._result = result

    def detect(self, mp_image):
        return self._result


def _make_tracker(landmarks_by_index):
    """Build a HeadFaceTracker with a fake detector, skipping __init__ entirely.

    Args:
        landmarks_by_index: {index: _FakeLandmark} overrides; all other
            indices (0-477) default to the origin.
    """
    tracker = object.__new__(HeadFaceTracker)
    tracker._landmarks = {}

    face_landmarks = [_FakeLandmark() for _ in range(478)]
    for idx, lm in landmarks_by_index.items():
        face_landmarks[idx] = lm

    tracker.detector = _FakeDetector(_FakeDetectionResult([face_landmarks]))
    return tracker


def _frame():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_yaw_is_zero_when_eyes_are_equidistant_from_camera():
    """No left/right depth difference between the eyes -> yaw is 0 radians."""
    tracker = _make_tracker({
        1: _FakeLandmark(x=0.0, y=0.0, z=0.0),      # nose
        33: _FakeLandmark(x=-0.1, y=0.0, z=0.0),    # left eye outer
        263: _FakeLandmark(x=0.1, y=0.0, z=0.0),    # right eye outer
        10: _FakeLandmark(x=0.0, y=-0.1, z=0.0),    # forehead
        152: _FakeLandmark(x=0.0, y=0.1, z=0.0),    # chin
    })
    result = tracker.process_frame(_frame())
    assert result is not None
    _, _, _, roll, pitch, yaw = result
    assert yaw == pytest.approx(0.0, abs=1e-9)


def test_yaw_is_a_radian_angle_from_eye_depth_difference():
    """yaw = arctan2(depth diff between eyes, horizontal eye-to-eye distance).

    Same style as pitch: an arctan2 of a z (depth) difference over an
    in-plane distance, not the old unitless nose-offset ratio.
    """
    le = _FakeLandmark(x=-0.1, y=0.0, z=0.0)
    re = _FakeLandmark(x=0.1, y=0.0, z=0.05)
    tracker = _make_tracker({
        1: _FakeLandmark(x=0.0, y=0.0, z=0.0),
        33: le,
        263: re,
        10: _FakeLandmark(x=0.0, y=-0.1, z=0.0),
        152: _FakeLandmark(x=0.0, y=0.1, z=0.0),
    })
    result = tracker.process_frame(_frame())
    _, _, _, roll, pitch, yaw = result

    expected_dx = re.x - le.x
    expected_dz = re.z - le.z
    expected_yaw = math.atan2(expected_dz, expected_dx)
    assert yaw == pytest.approx(expected_yaw)
    # It's a genuine radian angle now, not a large unitless ratio.
    assert -math.pi <= yaw <= math.pi


def test_yaw_is_independent_of_raw_nose_offset():
    """Moving the nose landmark alone (no eye-depth change) must not affect yaw.

    This is the behavior that was broken before the fix: yaw used to be
    computed straight from the nose's offset from the eye midpoint, which
    conflated in-plane nose position with head rotation.
    """
    base = _make_tracker({
        1: _FakeLandmark(x=0.0, y=0.0, z=0.0),
        33: _FakeLandmark(x=-0.1, y=0.0, z=0.0),
        263: _FakeLandmark(x=0.1, y=0.0, z=0.0),
        10: _FakeLandmark(x=0.0, y=-0.1, z=0.0),
        152: _FakeLandmark(x=0.0, y=0.1, z=0.0),
    })
    shifted_nose = _make_tracker({
        1: _FakeLandmark(x=0.08, y=0.0, z=0.0),  # nose shifted far right, same eyes
        33: _FakeLandmark(x=-0.1, y=0.0, z=0.0),
        263: _FakeLandmark(x=0.1, y=0.0, z=0.0),
        10: _FakeLandmark(x=0.0, y=-0.1, z=0.0),
        152: _FakeLandmark(x=0.0, y=0.1, z=0.0),
    })
    _, _, _, _, _, yaw_base = base.process_frame(_frame())
    _, _, _, _, _, yaw_shifted = shifted_nose.process_frame(_frame())
    assert yaw_base == pytest.approx(yaw_shifted)


def test_roll_and_pitch_are_unaffected_by_the_yaw_fix():
    """roll/pitch keep their pre-existing arctan2-based formulas."""
    le = _FakeLandmark(x=-0.1, y=0.02, z=0.0)
    re = _FakeLandmark(x=0.1, y=-0.02, z=0.0)
    top = _FakeLandmark(x=0.0, y=-0.1, z=0.03)
    chin = _FakeLandmark(x=0.0, y=0.1, z=0.0)
    tracker = _make_tracker({
        1: _FakeLandmark(x=0.0, y=0.0, z=0.0),
        33: le,
        263: re,
        10: top,
        152: chin,
    })
    result = tracker.process_frame(_frame())
    _, _, _, roll, pitch, _ = result

    expected_roll = math.atan2(re.y - le.y, re.x - le.x)
    expected_pitch = math.atan2(top.z - chin.z, chin.y - top.y)
    assert roll == pytest.approx(expected_roll)
    assert pitch == pytest.approx(expected_pitch)


def test_no_face_landmarks_returns_none():
    """An empty detection result yields None, not a crash."""
    tracker = object.__new__(HeadFaceTracker)
    tracker._landmarks = {}
    tracker.detector = _FakeDetector(_FakeDetectionResult([]))
    assert tracker.process_frame(_frame()) is None
