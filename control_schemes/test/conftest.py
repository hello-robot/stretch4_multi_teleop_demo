"""Shared pytest fixtures for control_schemes tests.

Provides a single rclpy context for the whole test session. No node in these
tests is ever spun, so timers/subscriptions created during construction never
fire and no ROS callback executes -- construction and direct handle_joy()
calls are the only things exercised.
"""

import pytest
import rclpy


@pytest.fixture(scope="session", autouse=True)
def ros_context():
    """Initializes rclpy once for the test session and shuts it down after."""
    rclpy.init(args=[])
    yield
    rclpy.shutdown()
