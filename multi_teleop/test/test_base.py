"""Unit tests for multi_teleop.base: ControlSchemeNode and InputInterfaceNode.

Pure-logic tests, no hardware or sim involved. Exercises:
- ControlSchemeNode activate/deactivate Trigger services.
- Publisher silencing while inactive (publish-interception wrapper).
- handle_joy dispatch, including axes/buttons length mismatch handling.
- InputInterfaceNode.publish_input Joy message construction.
"""
import pytest
import rclpy
from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger

from multi_teleop.base import ControlSchemeNode, InputInterfaceNode


class DummyControlScheme(ControlSchemeNode):
    """Minimal concrete ControlSchemeNode for testing: records handle_joy calls."""

    def __init__(self, node_name='dummy_control_scheme'):
        super().__init__(node_name, ['ax0', 'ax1'], ['btn0'])
        self.received = []

    def handle_joy(self, axes, buttons):
        self.received.append((list(axes), list(buttons)))


class DummyInputInterface(InputInterfaceNode):
    """Minimal concrete InputInterfaceNode for testing."""

    def __init__(self, node_name='dummy_input_interface'):
        super().__init__(node_name, ['x', 'y'], ['a', 'b'])


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def scheme_node():
    node = DummyControlScheme()
    yield node
    node.destroy_node()


@pytest.fixture
def input_node():
    node = DummyInputInterface()
    yield node
    node.destroy_node()


# --- ControlSchemeNode: activate/deactivate services ---

def test_active_by_default(scheme_node):
    assert scheme_node.active is True


def test_deactivate_callback_sets_inactive(scheme_node):
    resp = scheme_node._deactivate_callback(Trigger.Request(), Trigger.Response())
    assert scheme_node.active is False
    assert resp.success is True


def test_activate_callback_sets_active(scheme_node):
    scheme_node._deactivate_callback(Trigger.Request(), Trigger.Response())
    resp = scheme_node._activate_callback(Trigger.Request(), Trigger.Response())
    assert scheme_node.active is True
    assert resp.success is True


def test_activate_deactivate_services_registered(scheme_node):
    service_names = [name for name, _ in scheme_node.get_service_names_and_types()]
    assert '/dummy_control_scheme/activate' in service_names
    assert '/dummy_control_scheme/deactivate' in service_names


# --- ControlSchemeNode: publisher silencing while inactive ---

def test_publisher_silenced_when_inactive(scheme_node):
    pub = scheme_node.create_publisher(Joy, 'test_topic', 1)
    scheme_node._deactivate_callback(Trigger.Request(), Trigger.Response())
    # Should not raise, and should be a no-op (silenced).
    pub.publish(Joy())


def test_publisher_active_when_active(scheme_node):
    pub = scheme_node.create_publisher(Joy, 'test_topic2', 1)
    assert scheme_node.active is True
    # Should not raise; actually publishes.
    pub.publish(Joy())


# --- ControlSchemeNode: _joy_callback / handle_joy dispatch ---

def test_joy_callback_dispatches_matching_lengths(scheme_node):
    msg = Joy()
    msg.axes = [0.5, -0.5]
    msg.buttons = [1]
    scheme_node._joy_callback(msg)
    assert scheme_node.received == [([0.5, -0.5], [1])]


def test_joy_callback_ignored_when_inactive(scheme_node):
    scheme_node._deactivate_callback(Trigger.Request(), Trigger.Response())
    msg = Joy()
    msg.axes = [0.1, 0.2]
    msg.buttons = [0]
    scheme_node._joy_callback(msg)
    assert scheme_node.received == []


def test_joy_callback_pads_short_axes(scheme_node):
    msg = Joy()
    msg.axes = [0.9]  # expected 2
    msg.buttons = [1]
    scheme_node._joy_callback(msg)
    axes, buttons = scheme_node.received[0]
    assert axes == pytest.approx([0.9, 0.0])
    assert buttons == [1]


def test_joy_callback_truncates_long_axes(scheme_node):
    msg = Joy()
    msg.axes = [0.1, 0.2, 0.3]  # expected 2
    msg.buttons = [1]
    scheme_node._joy_callback(msg)
    axes, buttons = scheme_node.received[0]
    assert axes == pytest.approx([0.1, 0.2])


def test_joy_callback_pads_short_buttons(scheme_node):
    msg = Joy()
    msg.axes = [0.0, 0.0]
    msg.buttons = []  # expected 1
    scheme_node._joy_callback(msg)
    axes, buttons = scheme_node.received[0]
    assert buttons == [0]


def test_joy_callback_truncates_long_buttons(scheme_node):
    msg = Joy()
    msg.axes = [0.0, 0.0]
    msg.buttons = [1, 1]  # expected 1
    scheme_node._joy_callback(msg)
    axes, buttons = scheme_node.received[0]
    assert buttons == [1]


def test_control_axes_and_buttons_properties(scheme_node):
    assert scheme_node.control_axes == ['ax0', 'ax1']
    assert scheme_node.control_buttons == ['btn0']


# --- InputInterfaceNode: publish_input ---

def test_publish_input_updates_last_axes_and_buttons(input_node):
    input_node.publish_input([0.3, -0.7], [1, 0])
    assert input_node._last_axes == [0.3, -0.7]
    assert input_node._last_buttons == [1, 0]


def test_publish_input_publishes_joy_message(input_node):
    received = []
    sub = input_node.create_subscription(
        Joy, f'{input_node.get_name()}/output', lambda m: received.append(m), 1
    )
    input_node.publish_input([0.1, 0.2], [0, 1])
    for _ in range(20):
        rclpy.spin_once(input_node, timeout_sec=0.05)
        if received:
            break
    assert len(received) == 1
    assert list(received[0].axes) == pytest.approx([0.1, 0.2])
    assert list(received[0].buttons) == [0, 1]
    input_node.destroy_subscription(sub)


def test_publish_input_wrong_axes_length_raises(input_node):
    with pytest.raises(ValueError):
        input_node.publish_input([0.1], [0, 1])


def test_publish_input_wrong_buttons_length_raises(input_node):
    with pytest.raises(ValueError):
        input_node.publish_input([0.1, 0.2], [0])


def test_publish_input_none_values_keep_previous(input_node):
    input_node.publish_input([0.5, 0.5], [1, 1])
    input_node.publish_input([None, 0.9], [None, 0])
    assert input_node._last_axes == [0.5, 0.9]
    assert input_node._last_buttons == [1, 0]


def test_control_axes_and_buttons_properties_input(input_node):
    assert input_node.control_axes == ['x', 'y']
    assert input_node.control_buttons == ['a', 'b']
