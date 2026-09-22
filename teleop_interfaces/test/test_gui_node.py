"""Tests for gui_node.GuiNode's config loading and publish logic (no GTK app/display needed).

TeleopGuiApp itself is a GTK4 window and is not exercised here; see the
findings doc's manual checklist for that.
"""
import yaml
import pytest
import rclpy

from teleop_interfaces.gui_node import GuiNode


@pytest.fixture(scope='module', autouse=True)
def _rclpy_context():
    """Init/shutdown rclpy once for this test module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def test_no_config_falls_back_to_dummy_axis_and_button():
    """With no config_path, the node falls back to a single dummy axis/button."""
    n = GuiNode(config_path=None)
    try:
        assert n.control_axes == ['dummy_axis']
        assert n.control_buttons == ['dummy_button']
    finally:
        n.destroy_node()


def test_nonexistent_config_path_falls_back_to_dummy():
    """A config_path that doesn't exist on disk is silently ignored (falls back to dummy)."""
    n = GuiNode(config_path='/nonexistent/path/gui_config.yaml')
    try:
        assert n.control_axes == ['dummy_axis']
        assert n.control_buttons == ['dummy_button']
    finally:
        n.destroy_node()


def test_valid_config_loads_axis_and_button_names(tmp_path):
    """A well-formed gui_controls config produces matching axis/button names."""
    cfg = {
        'gui_controls': {
            'axes': [{'name': 'speed', 'min': -1.0, 'max': 1.0}],
            'buttons': [{'name': 'estop', 'toggle': True}],
        }
    }
    cfg_path = tmp_path / 'gui_config.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    n = GuiNode(config_path=str(cfg_path))
    try:
        assert n.control_axes == ['speed']
        assert n.control_buttons == ['estop']
    finally:
        n.destroy_node()


def test_malformed_config_falls_back_to_dummy(tmp_path):
    """Invalid YAML content is caught and treated the same as no config (current behavior)."""
    cfg_path = tmp_path / 'bad_config.yaml'
    cfg_path.write_text('not: valid: yaml: [')

    n = GuiNode(config_path=str(cfg_path))
    try:
        assert n.control_axes == ['dummy_axis']
        assert n.control_buttons == ['dummy_button']
    finally:
        n.destroy_node()


def test_update_and_publish_uses_current_widget_values(tmp_path):
    """update_and_publish publishes whatever is currently in _axes_values/_buttons_values."""
    cfg = {'gui_controls': {'axes': [{'name': 'speed', 'min': -1.0, 'max': 1.0}], 'buttons': []}}
    cfg_path = tmp_path / 'gui_config.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    n = GuiNode(config_path=str(cfg_path))
    try:
        n._axes_values[0] = 0.75
        n.update_and_publish()
        assert n._last_axes[0] == pytest.approx(0.75)
    finally:
        n.destroy_node()
