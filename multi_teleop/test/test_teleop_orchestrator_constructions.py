"""Unit tests for teleop_orchestrator.py's virtual-signal Construction classes.

Pure-math tests, no ROS/GTK spin needed (importing the module does pull
in gi/Gtk/Adw at module scope, but no window/app is created here).
Covers normal + edge-case inputs: deadband boundaries, division-shaped
edge cases, clamping/sign behavior, and missing-signal defaults.
"""
import math
import pytest

from multi_teleop.teleop_orchestrator import (
    SignalSource,
    Construction,
    AxisToButton,
    ButtonToAxis,
    TwoButtonsToAxis,
    AverageAxis,
    InvertAxis,
    SquareAxis,
    SqrtAxis,
    LogicalButton,
    AxisOverride,
    ButtonOverride,
    DeadbandAxis,
    AxisButtonControl,
)


# --- SignalSource / Construction base ---

def test_signal_source_evaluate_present():
    s = SignalSource('foo')
    assert s.evaluate({'foo': 0.42}) == 0.42


def test_signal_source_evaluate_missing_defaults_zero():
    s = SignalSource('missing')
    assert s.evaluate({}) == 0.0


def test_construction_evaluate_not_implemented():
    c = Construction('c', 'axis')
    with pytest.raises(NotImplementedError):
        c.evaluate({})


def test_construction_get_dependencies_default_empty():
    c = Construction('c', 'axis')
    assert c.get_dependencies() == []


# --- AxisToButton ---

@pytest.mark.parametrize('val,cutoff,expected', [
    (0.5, 0.3, 1),
    (0.3, 0.3, 0),   # boundary: not strictly greater than cutoff
    (0.29, 0.3, 0),
    (-0.5, 0.3, 0),
])
def test_axis_to_button(val, cutoff, expected):
    c = AxisToButton('c', 'a', cutoff)
    assert c.evaluate({'a': val}) == expected


def test_axis_to_button_missing_signal_defaults_zero():
    c = AxisToButton('c', 'a', -0.1)
    assert c.evaluate({}) == 1  # 0.0 > -0.1


def test_axis_to_button_dependencies():
    c = AxisToButton('c', 'a', 0.5)
    assert c.get_dependencies() == ['a']


# --- ButtonToAxis ---

def test_button_to_axis_pressed():
    c = ButtonToAxis('c', 'b', low=-1.0, high=1.0)
    assert c.evaluate({'b': 1}) == 1.0


def test_button_to_axis_released():
    c = ButtonToAxis('c', 'b', low=-1.0, high=1.0)
    assert c.evaluate({'b': 0}) == -1.0


def test_button_to_axis_missing_defaults_low():
    c = ButtonToAxis('c', 'b')
    assert c.evaluate({}) == 0.0


# --- TwoButtonsToAxis ---

@pytest.mark.parametrize('neg,pos,expected', [
    (0, 0, 0.0),
    (1, 0, -1.0),
    (0, 1, 1.0),
    (1, 1, 0.0),  # both held cancels out
])
def test_two_buttons_to_axis(neg, pos, expected):
    c = TwoButtonsToAxis('c', 'neg', 'pos')
    assert c.evaluate({'neg': neg, 'pos': pos}) == expected


def test_two_buttons_to_axis_missing_signals_default_zero():
    c = TwoButtonsToAxis('c', 'neg', 'pos')
    assert c.evaluate({}) == 0.0


# --- AverageAxis ---

@pytest.mark.parametrize('a,b,expected', [
    (1.0, -1.0, 0.0),
    (0.5, 0.5, 0.5),
    (0.0, 0.0, 0.0),
])
def test_average_axis(a, b, expected):
    c = AverageAxis('c', 'a', 'b')
    assert c.evaluate({'a': a, 'b': b}) == expected


def test_average_axis_missing_signal_treated_as_zero():
    c = AverageAxis('c', 'a', 'b')
    assert c.evaluate({'a': 1.0}) == 0.5


# --- InvertAxis ---

@pytest.mark.parametrize('val,expected', [
    (0.7, -0.7),
    (-0.7, 0.7),
    (0.0, 0.0),
])
def test_invert_axis(val, expected):
    c = InvertAxis('c', 'a')
    assert c.evaluate({'a': val}) == expected


# --- SquareAxis ---

@pytest.mark.parametrize('val,expected', [
    (0.5, 0.25),
    (-0.5, -0.25),
    (0.0, 0.0),
    (1.0, 1.0),
    (-1.0, -1.0),
])
def test_square_axis(val, expected):
    c = SquareAxis('c', 'a')
    assert c.evaluate({'a': val}) == pytest.approx(expected)


# --- SqrtAxis ---

@pytest.mark.parametrize('val,expected', [
    (0.25, 0.5),
    (-0.25, -0.5),
    (0.0, 0.0),
    (1.0, 1.0),
])
def test_sqrt_axis(val, expected):
    c = SqrtAxis('c', 'a')
    assert c.evaluate({'a': val}) == pytest.approx(expected)


# --- LogicalButton ---

@pytest.mark.parametrize('op,v1,v2,expected', [
    ('AND', 1, 1, 1), ('AND', 1, 0, 0), ('AND', 0, 0, 0),
    ('OR', 0, 0, 0), ('OR', 1, 0, 1), ('OR', 0, 1, 1),
    ('NAND', 1, 1, 0), ('NAND', 1, 0, 1),
    ('NOR', 0, 0, 1), ('NOR', 1, 0, 0),
])
def test_logical_button_binary_ops(op, v1, v2, expected):
    c = LogicalButton('c', 'b1', 'b2', op)
    assert c.evaluate({'b1': v1, 'b2': v2}) == expected


@pytest.mark.parametrize('v1,expected', [(0, 1), (1, 0)])
def test_logical_button_not(v1, expected):
    c = LogicalButton('c', 'b1', None, 'NOT')
    assert c.evaluate({'b1': v1}) == expected
    assert c.inputs == ['b1']


def test_logical_button_unknown_op_defaults_zero():
    c = LogicalButton('c', 'b1', 'b2', 'XOR')
    assert c.evaluate({'b1': 1, 'b2': 0}) == 0


def test_logical_button_non_not_op_without_b2_raises_value_error():
    """Fixed: b2=None with op != 'NOT' now raises ValueError in __init__.

    Previously this constructed fine and deferred to an IndexError
    inside evaluate() later.
    """
    with pytest.raises(ValueError):
        LogicalButton('c', 'b1', None, 'AND')


def test_logical_button_not_op_with_b2_raises_value_error():
    """Fixed: op == 'NOT' with a non-None b2 is rejected in __init__.

    op == 'NOT' iff b2 is None.
    """
    with pytest.raises(ValueError):
        LogicalButton('c', 'b1', 'b2', 'NOT')


# --- AxisOverride ---

def test_axis_override_primary_within_threshold_uses_secondary():
    c = AxisOverride('c', 'p', 's', threshold=0.1)
    assert c.evaluate({'p': 0.05, 's': 0.8}) == 0.8


def test_axis_override_primary_outside_threshold_uses_primary():
    c = AxisOverride('c', 'p', 's', threshold=0.1)
    assert c.evaluate({'p': 0.5, 's': 0.8}) == 0.5


def test_axis_override_boundary_equal_threshold_uses_primary():
    c = AxisOverride('c', 'p', 's', threshold=0.1)
    # abs(p) < threshold is False when equal, so primary passes through.
    assert c.evaluate({'p': 0.1, 's': 0.8}) == 0.1


# --- ButtonOverride ---

def test_button_override_priority_zero_uses_secondary():
    c = ButtonOverride('c', 'b1', 'b2')
    assert c.evaluate({'b1': 0, 'b2': 1}) == 1


def test_button_override_priority_nonzero_uses_priority():
    c = ButtonOverride('c', 'b1', 'b2')
    assert c.evaluate({'b1': 1, 'b2': 0}) == 1


# --- DeadbandAxis ---

def test_deadband_axis_within_band_zeroed():
    c = DeadbandAxis('c', 'a', low=-0.2, high=0.2)
    assert c.evaluate({'a': 0.0}) == 0.0
    assert c.evaluate({'a': 0.2}) == 0.0   # inclusive boundary
    assert c.evaluate({'a': -0.2}) == 0.0  # inclusive boundary


def test_deadband_axis_above_high_remapped():
    c = DeadbandAxis('c', 'a', low=-0.2, high=0.2)
    # val=1.0 -> (1.0-0.2)/(1.0-0.2) = 1.0
    assert c.evaluate({'a': 1.0}) == pytest.approx(1.0)
    # val=0.6 -> (0.6-0.2)/(1.0-0.2) = 0.5
    assert c.evaluate({'a': 0.6}) == pytest.approx(0.5)


def test_deadband_axis_below_low_remapped():
    c = DeadbandAxis('c', 'a', low=-0.2, high=0.2)
    assert c.evaluate({'a': -1.0}) == pytest.approx(-1.0)
    assert c.evaluate({'a': -0.6}) == pytest.approx(-0.5)


def test_deadband_axis_high_equal_one_out_of_range_input_no_longer_divides_by_zero():
    """Fixed: high=1.0 plus an out-of-range input no longer divides by zero.

    An out-of-[-1,1]-range input is possible via upstream constructions
    like AverageAxis/ButtonToAxis. The input is now clamped to [-1, 1]
    first, so val=1.5 clamps to 1.0, which falls in [low, high] and
    zeroes out.
    """
    c = DeadbandAxis('c', 'a', low=-0.2, high=1.0)
    assert c.evaluate({'a': 1.5}) == 0.0


def test_deadband_axis_low_equal_neg_one_out_of_range_input_no_longer_divides_by_zero():
    """Fixed: low=-1.0 plus an out-of-range input no longer divides by zero.

    val=-1.5 clamps to -1.0, which falls in [low, high] and zeroes out.
    """
    c = DeadbandAxis('c', 'a', low=-1.0, high=0.2)
    assert c.evaluate({'a': -1.5}) == 0.0


# --- AxisButtonControl ---

def test_axis_button_control_start_mode():
    c = AxisButtonControl('c', 'a', 'b', 'START')
    assert c.evaluate({'a': 0.5, 'b': 1}) == 0.5
    assert c.evaluate({'a': 0.5, 'b': 0}) == 0.0


def test_axis_button_control_stop_mode():
    c = AxisButtonControl('c', 'a', 'b', 'STOP')
    assert c.evaluate({'a': 0.5, 'b': 0}) == 0.5
    assert c.evaluate({'a': 0.5, 'b': 1}) == 0.0


def test_axis_button_control_toggle_mode():
    c = AxisButtonControl('c', 'a', 'b', 'TOGGLE')
    # Rising edge toggles on.
    assert c.evaluate({'a': 0.5, 'b': 1}) == 0.5
    # Held (no new rising edge): stays on.
    assert c.evaluate({'a': 0.5, 'b': 1}) == 0.5
    # Release: still on (state doesn't change on release).
    assert c.evaluate({'a': 0.5, 'b': 0}) == 0.5
    # Rising edge again toggles off.
    assert c.evaluate({'a': 0.5, 'b': 1}) == 0.0


def test_axis_button_control_unknown_mode_returns_zero():
    c = AxisButtonControl('c', 'a', 'b', 'BOGUS')
    assert c.evaluate({'a': 0.5, 'b': 1}) == 0.0
