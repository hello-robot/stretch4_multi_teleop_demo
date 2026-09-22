"""Mocked-hardware tests for voice_node.VoiceNode's command parsing (Whisper/mic/keyboard mocked).

Only _parse_commands/_word_to_num logic and construction are exercised here;
actual audio capture/transcription needs a human with a microphone (see the
findings doc's manual checklist).
"""
from unittest.mock import MagicMock, patch

import yaml
import pytest
import rclpy

from teleop_interfaces import voice_node
from teleop_interfaces.voice_node import VoiceNode


@pytest.fixture(scope='module', autouse=True)
def _rclpy_context():
    """Init/shutdown rclpy once for this test module."""
    rclpy.init()
    yield
    rclpy.shutdown()


VOICE_CONFIG = {
    'voice_commands': [
        {'axis': 'speed', 'action': 'step', 'up': 'faster', 'down': 'slower', 'step': 0.2},
        {'axis': 'zoom', 'action': 'scale', 'min': 0.0, 'max': 10.0},
        {'keyword': 'stop', 'action': 'tap'},
        {'keyword': 'lock', 'action': 'toggle'},
    ]
}


@pytest.fixture
def node(tmp_path):
    """Construct a VoiceNode with faster_whisper, sounddevice, and pynput mocked out."""
    cfg_path = tmp_path / 'voice_config.yaml'
    cfg_path.write_text(yaml.safe_dump(VOICE_CONFIG))

    fake_model_cls = MagicMock()
    with patch.object(voice_node, 'sd', MagicMock()), \
            patch.object(voice_node.keyboard, 'Listener', return_value=MagicMock()), \
            patch('faster_whisper.WhisperModel', fake_model_cls):
        n = VoiceNode(config_path=str(cfg_path))
        yield n
        n.destroy_node()


def test_constructs_and_publishes_on_expected_topic(node):
    """Node builds cleanly and its output publisher targets /voice_node/output."""
    assert node.output_publisher.topic_name == '/voice_node/output'


def test_control_axes_property_returns_configured_names(node):
    """control_axes/control_buttons return the configured names, not numeric state.

    VoiceNode's numeric state lives in self._axis_state/_button_state, kept
    separate from InputInterfaceNode's self._axes/_buttons name lists, so
    these properties (and the declared 'axis_names'/'button_names' ROS
    parameters) stay consistent.
    """
    assert node.control_axes == ['speed', 'zoom']
    assert node.control_buttons == ['stop', 'lock']
    assert node.get_parameter('axis_names').value == ['speed', 'zoom']
    assert node.get_parameter('button_names').value == ['stop', 'lock']


def test_axis_step_up_and_down(node):
    """'<name> <up-word>' / '<name> <down-word>' steps the axis by the configured step."""
    node._parse_commands('speed faster')
    assert node._axis_state[0] == pytest.approx(0.2)
    node._parse_commands('speed slower')
    assert node._axis_state[0] == pytest.approx(0.0)


def test_axis_step_clamps_to_range(node):
    """Repeated step-up commands clamp the axis at 1.0."""
    for _ in range(20):
        node._parse_commands('speed faster')
    assert node._axis_state[0] == 1.0


def test_axis_scale_maps_spoken_number_into_range(node):
    """'<name> <number>' with action=scale normalizes the value into [-1, 1] via min/max."""
    node._parse_commands('zoom five')  # midpoint of [0, 10] -> normalized 0.0
    assert node._axis_state[1] == pytest.approx(0.0, abs=1e-6)
    node._parse_commands('zoom ten')
    assert node._axis_state[1] == pytest.approx(1.0, abs=1e-6)
    node._parse_commands('zoom zero')
    assert node._axis_state[1] == pytest.approx(-1.0, abs=1e-6)


def test_axis_scale_handles_negative_words(node):
    """'<name> negative <number>' applies a -1 multiplier before normalizing."""
    node._parse_commands('zoom negative five')
    # real_val = -5, norm = 2*(-5-0)/10 - 1 = -2.0, clamped to -1.0
    assert node._axis_state[1] == -1.0


def test_keyword_tap_publishes_press_then_release(node):
    """A tap-action keyword sets the button, publishes, then clears it (current behavior)."""
    published = []
    node.publish_input = lambda axes, buttons: published.append(list(buttons))
    node._parse_commands('please stop now')
    assert node._button_state[0] == 0  # cleared again after the tap
    assert any(b[0] == 1 for b in published)  # but a press was published at some point


def test_keyword_toggle_flips_state(node):
    """A toggle-action keyword flips the button state on each match."""
    node._parse_commands('lock it')
    assert node._button_state[1] == 1
    node._parse_commands('lock it')
    assert node._button_state[1] == 0


def test_unmatched_text_is_a_no_op(node):
    """Text matching no configured command does not change axis/button state."""
    axes_before = list(node._axis_state)
    buttons_before = list(node._button_state)
    node._parse_commands('completely unrelated text')
    assert node._axis_state == axes_before
    assert node._button_state == buttons_before


@pytest.mark.parametrize('word,expected', [
    ('three', 3.0),
    ('3.5', 3.5),
    ('-2', -2.0),
    ('banana', None),
])
def test_word_to_num(node, word, expected):
    """_word_to_num parses spoken-number words and numerals, else returns None."""
    assert node._word_to_num(word) == expected


def test_update_and_publish_is_wired_to_a_timer(node):
    """update_and_publish is called periodically by a ROS timer.

    Unlike before, VoiceNode now calls create_timer() so a voice-set
    axis/button value keeps being republished between spoken commands
    (matching the pattern of the other InputInterfaceNode subclasses).
    """
    assert len(node._timers) == 1
    node.update_and_publish()  # still directly callable without raising


def test_update_and_publish_republishes_current_state(node):
    """update_and_publish publishes the current axis/button state as-is."""
    node._parse_commands('speed faster')
    published = []
    node.publish_input = lambda axes, buttons: published.append((list(axes), list(buttons)))
    node.update_and_publish()
    assert len(published) == 1
    axes, buttons = published[0]
    assert axes[0] == pytest.approx(0.2)


def test_malformed_step_command_is_skipped_not_raised(tmp_path):
    """A 'step' axis entry missing 'up'/'down' is skipped at load time, not KeyError'd."""
    cfg = {
        'voice_commands': [
            {'axis': 'speed', 'action': 'step', 'step': 0.2},  # missing up/down
            {'keyword': 'stop', 'action': 'tap'},
        ]
    }
    cfg_path = tmp_path / 'bad_voice_config.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    with patch.object(voice_node, 'sd', MagicMock()), \
            patch.object(voice_node.keyboard, 'Listener', return_value=MagicMock()), \
            patch('faster_whisper.WhisperModel', MagicMock()):
        n = VoiceNode(config_path=str(cfg_path))
        try:
            # The malformed 'speed' command was dropped entirely.
            assert 'speed' not in n.control_axes
            # Parsing text that would have matched the malformed entry must not raise.
            n._parse_commands('speed faster')
            n._parse_commands('please stop now')
        finally:
            n.destroy_node()


def test_malformed_scale_command_is_skipped_not_raised(tmp_path):
    """A 'scale' axis entry missing 'min'/'max' is skipped at load time, not KeyError'd."""
    cfg = {
        'voice_commands': [
            {'axis': 'zoom', 'action': 'scale'},  # missing min/max
        ]
    }
    cfg_path = tmp_path / 'bad_voice_config.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    with patch.object(voice_node, 'sd', MagicMock()), \
            patch.object(voice_node.keyboard, 'Listener', return_value=MagicMock()), \
            patch('faster_whisper.WhisperModel', MagicMock()):
        n = VoiceNode(config_path=str(cfg_path))
        try:
            assert 'zoom' not in n.control_axes
            n._parse_commands('zoom five')  # must not raise
        finally:
            n.destroy_node()


def test_valid_and_malformed_commands_mixed(tmp_path):
    """A malformed entry is dropped while sibling well-formed entries still work."""
    cfg = {
        'voice_commands': [
            {'axis': 'speed', 'action': 'step', 'up': 'faster', 'down': 'slower', 'step': 0.2},
            {'axis': 'zoom', 'action': 'scale'},  # malformed: missing min/max
        ]
    }
    cfg_path = tmp_path / 'mixed_voice_config.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    with patch.object(voice_node, 'sd', MagicMock()), \
            patch.object(voice_node.keyboard, 'Listener', return_value=MagicMock()), \
            patch('faster_whisper.WhisperModel', MagicMock()):
        n = VoiceNode(config_path=str(cfg_path))
        try:
            assert n.control_axes == ['speed']
            n._parse_commands('speed faster')
            assert n._axis_state[0] == pytest.approx(0.2)
        finally:
            n.destroy_node()
