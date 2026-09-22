"""Pure-logic tests for mediapipe_base.CvBridge.imgmsg_to_cv2 (no rclpy/hardware needed)."""
import numpy as np
import pytest
from sensor_msgs.msg import Image

from teleop_interfaces.mediapipe_base import CvBridge


def _make_image(encoding, height, width, channels, dtype, step=None, fill=None):
    """Build a sensor_msgs/Image with a deterministic data buffer for testing."""
    itemsize = np.dtype(dtype).itemsize
    row_bytes = width * channels * itemsize
    if step is None:
        step = row_bytes
    if fill is None:
        # Deterministic increasing values so we can check pixel identity.
        fill = (np.arange(height * width * channels, dtype=dtype) % 250)

    shape = (height, width, channels) if channels > 1 else (height, width)
    arr = fill.reshape(shape)
    buf = bytearray(height * step)
    for r in range(height):
        row = arr[r].tobytes()
        buf[r * step: r * step + len(row)] = row

    msg = Image()
    msg.height = height
    msg.width = width
    msg.encoding = encoding
    msg.step = step
    msg.data = bytes(buf)
    return msg, arr


def test_bgr8_no_padding_returns_matching_array():
    """bgr8 input with a tightly packed buffer round-trips to the same pixel values."""
    msg, arr = _make_image('bgr8', height=2, width=3, channels=3, dtype=np.uint8)
    bridge = CvBridge()
    out = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out, arr)


def test_bgr8_output_is_a_copy_not_a_view():
    """The bgr8 path returns a copy, so mutating it must not alter msg.data."""
    msg, _ = _make_image('bgr8', height=2, width=2, channels=3, dtype=np.uint8)
    bridge = CvBridge()
    out = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    original_data = bytes(msg.data)
    out[0, 0, 0] = 255
    assert bytes(msg.data) == original_data


def test_bgr8_with_row_padding_uses_strided_path():
    """A step larger than width*channels (row padding) is still decoded correctly."""
    width, height, channels = 2, 2, 3
    itemsize = 1
    row_bytes = width * channels * itemsize
    padded_step = row_bytes + 4  # extra padding bytes per row
    msg, arr = _make_image('bgr8', height=height, width=width, channels=channels,
                            dtype=np.uint8, step=padded_step)
    bridge = CvBridge()
    out = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    assert out.shape == (height, width, channels)
    assert np.array_equal(out, arr)


def test_rgb8_is_converted_to_bgr_order():
    """rgb8 input is channel-swapped to bgr8 output."""
    msg, arr = _make_image('rgb8', height=1, width=1, channels=3, dtype=np.uint8,
                            fill=np.array([10, 20, 30], dtype=np.uint8))
    bridge = CvBridge()
    out = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    # RGB (10, 20, 30) -> BGR (30, 20, 10)
    assert list(out[0, 0]) == [30, 20, 10]


def test_mono8_is_converted_to_3_channel_bgr():
    """mono8 grayscale input is broadcast into a 3-channel bgr8 image."""
    msg, _ = _make_image('mono8', height=2, width=2, channels=1, dtype=np.uint8)
    bridge = CvBridge()
    out = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    assert out.shape == (2, 2, 3)


def test_unsupported_desired_encoding_raises_not_implemented():
    """Requesting any desired_encoding other than bgr8 raises NotImplementedError."""
    msg, _ = _make_image('bgr8', height=1, width=1, channels=3, dtype=np.uint8)
    bridge = CvBridge()
    with pytest.raises(NotImplementedError):
        bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')


def test_unsupported_msg_encoding_raises_value_error():
    """An unrecognized msg.encoding raises ValueError."""
    msg, _ = _make_image('bgr8', height=1, width=1, channels=3, dtype=np.uint8)
    msg.encoding = 'yuv422'
    bridge = CvBridge()
    with pytest.raises(ValueError):
        bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
