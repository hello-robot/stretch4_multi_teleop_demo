"""Unit tests for OrchestratorNode's pure signal-routing/mapping logic.

Exercises topological sort, per-scheme mapping-page bookkeeping, and
process_and_publish's deadband/sensitivity/invert math and scheme
selection/publishing -- by constructing a real (unspun) OrchestratorNode
and calling its methods directly with synthetic data, per the plan's
tier-3 guidance ("dynamic topic-discovery/mapping logic testable in
isolation ... by calling its mapping methods directly with synthetic
topic lists"). The GTK UI classes are not covered here (see
findings_multi_teleop.md's Manual Checklist Candidates).

ROS service calls (activate/deactivate) triggered from background
threads by select_scheme are mocked out so tests don't block on
nonexistent services.
"""
from unittest.mock import patch

import pytest
import rclpy
from sensor_msgs.msg import Joy

from multi_teleop.teleop_orchestrator import OrchestratorNode


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    # Patch out the background-thread service call so select_scheme()
    # doesn't try to reach real activate/deactivate services.
    with patch.object(OrchestratorNode, 'call_scheme_service', lambda self, name, action: None):
        n = OrchestratorNode()
        yield n
        n.destroy_node()


# --- get_topological_sort ---

def test_topological_sort_orders_dependencies_first(node):
    from multi_teleop.teleop_orchestrator import InvertAxis, SquareAxis

    node.constructions['a'] = InvertAxis('a', 'raw_axis')
    node.constructions['b'] = SquareAxis('b', 'a')  # depends on 'a'
    order = node.get_topological_sort()
    assert order.index('a') < order.index('b')


def test_topological_sort_empty_when_no_constructions(node):
    assert node.get_topological_sort() == []


def test_topological_sort_circular_dependency_returns_empty(node):
    from multi_teleop.teleop_orchestrator import InvertAxis

    node.constructions['a'] = InvertAxis('a', 'b')
    node.constructions['b'] = InvertAxis('b', 'a')
    assert node.get_topological_sort() == []


def test_topological_sort_circular_dependency_is_logged(node):
    """Fixed: the circular-dependency error is now logged.

    Previously the log call was commented out, so it failed silently.
    """
    from multi_teleop.teleop_orchestrator import InvertAxis

    node.constructions['a'] = InvertAxis('a', 'b')
    node.constructions['b'] = InvertAxis('b', 'a')

    logged = []

    class FakeLogger:
        def error(self, msg, **kwargs):
            logged.append(msg)
        def info(self, *a, **k): pass
        def warn(self, *a, **k): pass
        def warning(self, *a, **k): pass

    node.get_logger = lambda: FakeLogger()

    assert node.get_topological_sort() == []
    assert len(logged) == 1
    assert 'a' in logged[0] and 'b' in logged[0]


def test_topological_sort_partial_cycle_still_orders_noncyclic_nodes(node):
    """Fixed: a circular dependency no longer blocks unrelated evaluation.

    Only the constructions actually involved in the cycle are excluded
    from the returned order.
    """
    from multi_teleop.teleop_orchestrator import InvertAxis, SquareAxis

    node.constructions['a'] = InvertAxis('a', 'b')
    node.constructions['b'] = InvertAxis('b', 'a')  # a <-> b cycle
    node.constructions['c'] = InvertAxis('c', 'raw_axis')  # independent
    node.constructions['d'] = SquareAxis('d', 'c')  # depends on c

    order = node.get_topological_sort()
    assert 'a' not in order
    assert 'b' not in order
    assert 'c' in order and 'd' in order
    assert order.index('c') < order.index('d')


# --- mapping-page bookkeeping ---

def test_get_mappings_for_scheme_creates_default(node):
    mappings = node.get_mappings_for_scheme('my_scheme')
    assert len(mappings) == 1
    assert mappings[0] == {'axes': {}, 'buttons': {}, 'sensitivities': {}, 'inverted': {}}


def test_get_active_mapping_index_defaults_zero(node):
    assert node.get_active_mapping_index_for_scheme('my_scheme') == 0


def test_get_active_mapping_resets_out_of_range_index(node):
    node.get_mappings_for_scheme('s')  # one mapping page exists (index 0)
    node.scheme_active_indices['s'] = 5  # out of range
    mapping = node.get_active_mapping('s')
    assert node.scheme_active_indices['s'] == 0
    assert mapping is node.scheme_mappings['s'][0]


# --- process_and_publish ---

def test_process_and_publish_no_selected_scheme_is_noop(node):
    node.selected_scheme = None
    node.process_and_publish()  # should not raise


def test_process_and_publish_applies_mapping_and_publishes(node):
    node.control_schemes['/scheme'] = {
        'axis_names': ['out_axis'],
        'button_names': ['out_btn'],
        'topic': '/scheme/input',
    }
    node.selected_scheme = '/scheme'
    node.output_publisher = node.create_publisher(Joy, '/scheme/input', 1)
    node.signal_pool = {'src_axis': 0.5, 'src_btn': 1}

    mapping = node.get_active_mapping('/scheme')
    mapping['axes']['out_axis'] = 'src_axis'
    mapping['buttons']['out_btn'] = 'src_btn'
    # Neutral sensitivity (50 => identity), no deadband, no inversion.

    received = []
    sub = node.create_subscription(Joy, '/scheme/input', lambda m: received.append(m), 1)

    node.process_and_publish()
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)
        if received:
            break

    assert len(received) == 1
    assert received[0].axes[0] == pytest.approx(0.5)
    assert received[0].buttons[0] == 1
    node.destroy_subscription(sub)


def test_process_and_publish_deadband_zeros_small_value(node):
    node.control_schemes['/scheme'] = {
        'axis_names': ['out_axis'], 'button_names': [], 'topic': '/scheme2/input',
    }
    node.selected_scheme = '/scheme'
    node.output_publisher = node.create_publisher(Joy, '/scheme2/input', 1)
    node.signal_pool = {'src_axis': 0.05}

    mapping = node.get_active_mapping('/scheme')
    mapping['axes']['out_axis'] = 'src_axis'
    mapping['deadbands'] = {'out_axis': 10}  # 10% -> 0.1 deadband

    received = []
    sub = node.create_subscription(Joy, '/scheme2/input', lambda m: received.append(m), 1)
    node.process_and_publish()
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)
        if received:
            break

    assert received[0].axes[0] == 0.0
    node.destroy_subscription(sub)


def test_process_and_publish_inversion_flips_sign(node):
    node.control_schemes['/scheme'] = {
        'axis_names': ['out_axis'], 'button_names': [], 'topic': '/scheme3/input',
    }
    node.selected_scheme = '/scheme'
    node.output_publisher = node.create_publisher(Joy, '/scheme3/input', 1)
    node.signal_pool = {'src_axis': 0.4}

    mapping = node.get_active_mapping('/scheme')
    mapping['axes']['out_axis'] = 'src_axis'
    mapping['inverted'] = {'out_axis': True}

    received = []
    sub = node.create_subscription(Joy, '/scheme3/input', lambda m: received.append(m), 1)
    node.process_and_publish()
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)
        if received:
            break

    assert received[0].axes[0] == pytest.approx(-0.4)
    node.destroy_subscription(sub)


def test_process_and_publish_unmapped_output_defaults_zero(node):
    node.control_schemes['/scheme'] = {
        'axis_names': ['unmapped_axis'], 'button_names': ['unmapped_btn'], 'topic': '/scheme4/input',
    }
    node.selected_scheme = '/scheme'
    node.output_publisher = node.create_publisher(Joy, '/scheme4/input', 1)
    node.signal_pool = {}

    received = []
    sub = node.create_subscription(Joy, '/scheme4/input', lambda m: received.append(m), 1)
    node.process_and_publish()
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)
        if received:
            break

    assert received[0].axes[0] == 0.0
    assert received[0].buttons[0] == 0
    node.destroy_subscription(sub)


def test_process_and_publish_construction_exception_is_logged(node):
    """Fixed: a Construction that raises during evaluate() is now logged.

    Previously a bare ``except: pass`` swallowed it silently. It's now
    logged (naming the failing construction), and execution continues.
    """
    class ExplodingConstruction:
        out_type = 'axis'
        def evaluate(self, pool):
            raise RuntimeError('boom')
        def get_dependencies(self):
            return []

    node.constructions['bad'] = ExplodingConstruction()
    node.selected_scheme = None

    logged = []

    class FakeLogger:
        def error(self, msg, **kwargs):
            logged.append(msg)
        def info(self, *a, **k): pass
        def warn(self, *a, **k): pass
        def warning(self, *a, **k): pass

    node.get_logger = lambda: FakeLogger()

    node.process_and_publish()  # must not raise despite the construction blowing up
    assert 'bad' not in node.signal_pool
    assert len(logged) == 1
    assert 'bad' in logged[0]


# --- select_scheme ---

def test_select_scheme_sets_selected_and_creates_publisher(node):
    node.control_schemes['/scheme_a'] = {
        'axis_names': ['x'], 'button_names': [], 'topic': '/scheme_a/input',
    }
    node.select_scheme('/scheme_a')
    assert node.selected_scheme == '/scheme_a'
    assert node.output_publisher is not None


def test_select_scheme_noop_if_already_selected(node):
    node.control_schemes['/scheme_b'] = {
        'axis_names': ['x'], 'button_names': [], 'topic': '/scheme_b/input',
    }
    node.select_scheme('/scheme_b')
    pub_before = node.output_publisher
    node.select_scheme('/scheme_b')
    assert node.output_publisher is pub_before


def test_select_scheme_unknown_scheme_resets_publisher_to_none(node):
    """Fixed: selecting an unknown scheme now resets output_publisher.

    It still sets selected_scheme and destroys the old publisher, but
    now also resets self.output_publisher to None instead of leaving it
    pointing at an already-destroyed Publisher object.
    """
    node.control_schemes['/scheme_c'] = {
        'axis_names': ['x'], 'button_names': [], 'topic': '/scheme_c/input',
    }
    node.select_scheme('/scheme_c')
    node.select_scheme('/does_not_exist')
    assert node.selected_scheme == '/does_not_exist'
    assert node.output_publisher is None


# --- process_and_publish: throttling/periodic-publish/max-rate options ---
# (OptionsDialog wiring -- previously a dead feature; see AGENT_GUIDE.md /
# KNOWN_ISSUES.md.)

def _setup_throttle_scheme(node, topic):
    node.control_schemes['/tscheme'] = {
        'axis_names': ['out_axis'], 'button_names': [], 'topic': topic,
    }
    node.selected_scheme = '/tscheme'
    # Depth >1 so multiple back-to-back publishes (before the test spins)
    # aren't dropped by a depth-1 history queue.
    node.output_publisher = node.create_publisher(Joy, topic, 10)
    mapping = node.get_active_mapping('/tscheme')
    mapping['axes']['out_axis'] = 'src_axis'
    received = []
    sub = node.create_subscription(Joy, topic, lambda m: received.append(m), 10)
    return received, sub


def _spin_until(node, received, count, tries=20):
    for _ in range(tries):
        rclpy.spin_once(node, timeout_sec=0.02)
        if len(received) >= count:
            break


def test_process_and_publish_throttling_disabled_publishes_every_tick(node):
    """With throttling_enabled defaulting False, publish every tick.

    Unthrottled, same as before the option existed.
    """
    received, sub = _setup_throttle_scheme(node, '/tscheme0/input')
    node.signal_pool = {'src_axis': 0.5}

    node.process_and_publish()
    node.process_and_publish()
    _spin_until(node, received, 2)

    assert len(received) == 2
    node.destroy_subscription(sub)


def test_process_and_publish_throttling_rate_limits_output(node):
    """A publish attempted before 1/max_rate_hz has elapsed is skipped.

    Applies whenever throttling_enabled is True.
    """
    received, sub = _setup_throttle_scheme(node, '/tscheme1/input')
    node.signal_pool = {'src_axis': 0.5}

    node.throttling_enabled = True
    node.periodic_publish = True
    node.max_rate_hz = 1.0  # min interval 1s

    with patch('multi_teleop.teleop_orchestrator.time.time', side_effect=[100.0, 100.5]):
        node.process_and_publish()  # rate ok vs. initial last_publish_time=0.0
        node.process_and_publish()  # only 0.5s later: rate not ok, skipped
    _spin_until(node, received, 1)

    assert len(received) == 1
    node.destroy_subscription(sub)


def test_process_and_publish_periodic_forces_publish_regardless_of_change(node):
    """periodic_publish=True still publishes at the bounded rate.

    Even when the computed output hasn't changed.
    """
    received, sub = _setup_throttle_scheme(node, '/tscheme2/input')
    node.signal_pool = {'src_axis': 0.5}  # never changes across calls

    node.throttling_enabled = True
    node.periodic_publish = True
    node.max_rate_hz = 1.0

    with patch('multi_teleop.teleop_orchestrator.time.time', side_effect=[100.0, 100.5, 101.0]):
        node.process_and_publish()  # publishes
        node.process_and_publish()  # rate not ok, skipped
        node.process_and_publish()  # rate ok again, publishes despite no change
    _spin_until(node, received, 2)

    assert len(received) == 2
    node.destroy_subscription(sub)


def test_process_and_publish_on_change_only_publishes_when_output_changes(node):
    """periodic_publish=False only publishes when output changed.

    Still bounded by max_rate_hz.
    """
    received, sub = _setup_throttle_scheme(node, '/tscheme3/input')

    node.throttling_enabled = True
    node.periodic_publish = False
    node.max_rate_hz = 1.0

    with patch('multi_teleop.teleop_orchestrator.time.time', side_effect=[100.0, 101.0, 102.0]):
        node.signal_pool = {'src_axis': 0.5}
        node.process_and_publish()  # first publish (changed from unset)
        node.signal_pool = {'src_axis': 0.5}
        node.process_and_publish()  # unchanged: skipped
        node.signal_pool = {'src_axis': 0.7}
        node.process_and_publish()  # changed: publishes
    _spin_until(node, received, 2)

    assert len(received) == 2
    node.destroy_subscription(sub)


# --- service client lifecycle: fetch_source_names / check_if_control_scheme ---

def test_fetch_source_names_destroys_client_when_service_unavailable(node):
    """Fixed: fetch_source_names now always destroys its service client.

    Even when the service never becomes available (previously leaked).
    """
    from unittest.mock import MagicMock

    fake_client = MagicMock()
    fake_client.wait_for_service.return_value = False
    node.create_client = MagicMock(return_value=fake_client)

    node.inputs['/leaky_topic'] = {
        'axis_names': [], 'button_names': [], 'last_msg': None,
        'last_time': None, 'node_name': '/nonexistent',
    }
    node.fetch_source_names('/leaky_topic', '/nonexistent')

    fake_client.destroy.assert_called_once()


def test_fetch_source_names_timeout_on_stuck_future_does_not_hang(node):
    """Fixed: a stuck call_async future no longer hangs the thread.

    _wait_for_future now bounds it with a timeout, and the client is
    still destroyed afterward.
    """
    from unittest.mock import MagicMock

    fake_client = MagicMock()
    fake_client.wait_for_service.return_value = True
    fake_future = MagicMock()
    fake_future.done.return_value = False  # never completes
    fake_client.call_async.return_value = fake_future
    node.create_client = MagicMock(return_value=fake_client)

    node.inputs['/stuck_topic'] = {
        'axis_names': [], 'button_names': [], 'last_msg': None,
        'last_time': None, 'node_name': '/stuck_node',
    }

    # Fast-forward time.time() past the timeout instead of really sleeping.
    times = iter([1000.0, 1000.0, 1010.0])
    with patch('multi_teleop.teleop_orchestrator.time.time', lambda: next(times, 1010.0)), \
         patch('multi_teleop.teleop_orchestrator.time.sleep', lambda s: None):
        node.fetch_source_names('/stuck_topic', '/stuck_node')

    fake_client.destroy.assert_called_once()


def test_check_if_control_scheme_destroys_client_when_service_unavailable(node):
    """Fixed: check_if_control_scheme now always destroys its client.

    Even when the service never becomes available (previously leaked).
    """
    from unittest.mock import MagicMock

    fake_client = MagicMock()
    fake_client.wait_for_service.return_value = False
    node.create_client = MagicMock(return_value=fake_client)

    node.check_if_control_scheme('/nonexistent_scheme', '/some_topic')

    fake_client.destroy.assert_called_once()


def test_discovery_lock_guards_concurrent_mutation(node):
    """Regression test for the discovery-bookkeeping race (KNOWN_ISSUES.md).

    Hammers self.inputs/self.control_schemes/self.checking_nodes from
    background threads (mimicking fetch_source_names/check_if_control_scheme)
    concurrently with spin-thread-style calls (process_and_publish,
    select_scheme), asserting nothing raises under contention.
    """
    import threading

    node.inputs['/race_topic'] = {
        'axis_names': ['a'], 'button_names': ['b'], 'last_msg': None,
        'last_time': None, 'node_name': '/race_input',
    }
    errors = []
    stop = threading.Event()

    def hammer_background():
        i = 0
        while not stop.is_set():
            name = f'/race_scheme_{i % 3}'
            try:
                with node._discovery_lock:
                    node.checking_nodes.add(name)
                    node.control_schemes[name] = {
                        'axis_names': ['x'], 'button_names': ['y'], 'topic': '/race_topic',
                    }
                    node.checking_nodes.discard(name)
            except Exception as e:
                errors.append(e)
            i += 1

    threads = [threading.Thread(target=hammer_background, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()

    try:
        for i in range(200):
            try:
                node.selected_scheme = f'/race_scheme_{i % 3}'
                node.process_and_publish()
                node.select_scheme(f'/race_scheme_{(i + 1) % 3}')
            except Exception as e:
                errors.append(e)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2.0)

    assert errors == []
