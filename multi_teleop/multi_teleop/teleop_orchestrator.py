#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from rcl_interfaces.srv import GetParameters, ListParameters
import gi
import threading
import time
import yaml
import os
import math
import numpy as np

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Gio, GLib, Adw, Pango, Gdk

# --- Construction Logic Classes ---

class SignalSource:
    def __init__(self, name):
        self.name = name
    def evaluate(self, pool):
        return pool.get(self.name, 0.0)

class Construction:
    def __init__(self, name, out_type):
        self.name = name
        self.out_type = out_type # 'axis' or 'button'
        self.inputs = [] # list of signal names
    
    def evaluate(self, pool):
        raise NotImplementedError()
    
    def get_dependencies(self):
        return self.inputs

class AxisToButton(Construction):
    def __init__(self, name, axis_name, cutoff):
        super().__init__(name, 'button')
        self.inputs = [axis_name]
        self.cutoff = cutoff
    def evaluate(self, pool):
        val = pool.get(self.inputs[0], 0.0)
        return 1 if val > self.cutoff else 0

class ButtonToAxis(Construction):
    def __init__(self, name, button_name, low=0.0, high=1.0):
        super().__init__(name, 'axis')
        self.inputs = [button_name]
        self.low = low
        self.high = high
    def evaluate(self, pool):
        return self.high if pool.get(self.inputs[0], 0) else self.low

class TwoButtonsToAxis(Construction):
    def __init__(self, name, btn_neg, btn_pos):
        super().__init__(name, 'axis')
        self.inputs = [btn_neg, btn_pos]
    def evaluate(self, pool):
        v_neg = pool.get(self.inputs[0], 0)
        v_pos = pool.get(self.inputs[1], 0)
        if v_pos and not v_neg: return 1.0
        if v_neg and not v_pos: return -1.0
        return 0.0

class AverageAxis(Construction):
    def __init__(self, name, axis1, axis2):
        super().__init__(name, 'axis')
        self.inputs = [axis1, axis2]
    def evaluate(self, pool):
        return (pool.get(self.inputs[0], 0.0) + pool.get(self.inputs[1], 0.0)) / 2.0

class InvertAxis(Construction):
    def __init__(self, name, axis_name):
        super().__init__(name, 'axis')
        self.inputs = [axis_name]
    def evaluate(self, pool):
        return -pool.get(self.inputs[0], 0.0)

class SquareAxis(Construction):
    def __init__(self, name, axis_name):
        super().__init__(name, 'axis')
        self.inputs = [axis_name]
    def evaluate(self, pool):
        val = pool.get(self.inputs[0], 0.0)
        return math.copysign(val * val, val)

class SqrtAxis(Construction):
    def __init__(self, name, axis_name):
        super().__init__(name, 'axis')
        self.inputs = [axis_name]
    def evaluate(self, pool):
        val = pool.get(self.inputs[0], 0.0)
        return math.copysign(math.sqrt(abs(val)), val)

class LogicalButton(Construction):
    def __init__(self, name, b1, b2, op):
        super().__init__(name, 'button')
        self.inputs = [b1, b2] if b2 else [b1]
        self.op = op # AND, OR, NAND, NOR, NOT
    def evaluate(self, pool):
        v1 = pool.get(self.inputs[0], 0)
        if self.op == 'NOT':
            return 0 if v1 else 1
        v2 = pool.get(self.inputs[1], 0)
        if self.op == 'AND': return 1 if v1 and v2 else 0
        if self.op == 'OR': return 1 if v1 or v2 else 0
        if self.op == 'NAND': return 0 if v1 and v2 else 1
        if self.op == 'NOR': return 0 if v1 or v2 else 1
        return 0

class AxisOverride(Construction):
    def __init__(self, name, primary, secondary, threshold):
        super().__init__(name, 'axis')
        self.inputs = [primary, secondary]
        self.threshold = threshold
    def evaluate(self, pool):
        p = pool.get(self.inputs[0], 0.0)
        s = pool.get(self.inputs[1], 0.0)
        return s if abs(p) < self.threshold else p

class ButtonOverride(Construction):
    def __init__(self, name, b1, b2):
        super().__init__(name, 'button')
        self.inputs = [b1, b2]
    def evaluate(self, pool):
        v1 = pool.get(self.inputs[0], 0)
        v2 = pool.get(self.inputs[1], 0)
        return v2 if v1 == 0 else v1

class DeadbandAxis(Construction):
    def __init__(self, name, axis_name, low, high):
        super().__init__(name, 'axis')
        self.inputs = [axis_name]
        self.low = low
        self.high = high
    def evaluate(self, pool):
        val = pool.get(self.inputs[0], 0.0)
        if self.low <= val <= self.high:
            return 0.0
        if val > self.high:
            return (val - self.high) / (1.0 - self.high)
        if val < self.low:
            return (val - self.low) / (-1.0 - self.low) * -1.0 # Linear remap to -1, 0
        return 0.0

class AxisButtonControl(Construction):
    def __init__(self, name, axis_name, button_name, mode):
        super().__init__(name, 'axis')
        self.inputs = [axis_name, button_name]
        self.mode = mode # START, STOP, TOGGLE
        self.toggle_state = False
        self.prev_button = 0
    def evaluate(self, pool):
        axis_val = pool.get(self.inputs[0], 0.0)
        btn_val = pool.get(self.inputs[1], 0)
        
        if self.mode == 'START':
            return axis_val if btn_val == 1 else 0.0
        if self.mode == 'STOP':
            return axis_val if btn_val == 0 else 0.0
        if self.mode == 'TOGGLE':
            if btn_val == 1 and self.prev_button == 0:
                self.toggle_state = not self.toggle_state
            self.prev_button = btn_val
            return axis_val if self.toggle_state else 0.0
        return 0.0

# --- Orchestrator Node ---

class OrchestratorNode(Node):
    def __init__(self):
        super().__init__('teleop_orchestrator')
        self.get_logger().info("=============================================")
        self.get_logger().info("Teleop Orchestrator Version 2.0 (Multi-Mapping, Sensitivity, Invert) Initialized!")
        self.get_logger().info("=============================================")
        self.inputs = {} # topic -> {axis_names, button_names, last_msg, node_name}
        self.constructions = {} # name -> Construction object
        self.mapping = {'axes': {}, 'buttons': {}} # output_name -> input_source_name
        self.control_schemes = {} # node_name -> {axis_names, button_names}
        self.selected_scheme = None
        self.output_publisher = None
        self.sorted_consts_cache = []
        
        # Track nodes currently being queried to avoid duplicate threads
        self.checking_nodes = set()
        
        # Options
        self.throttling_enabled = False
        self.periodic_publish = False
        self.max_rate_hz = 30.0
        self.last_publish_time = 0.0
        
        # Discovery timer
        self.create_timer(2.0, self.discover_environment)
        
        # Main computation loop trigger (30Hz)
        self.create_timer(1.0/30.0, self.process_and_publish)
        
        self.signal_pool = {}
        
        # Multi-mapping data structures
        self.scheme_mappings = {} # scheme_name -> list of mapping dicts
        self.scheme_cycle_buttons = {} # scheme_name -> cycle_button_signal_name
        self.scheme_active_indices = {} # scheme_name -> active_mapping_index
        self.prev_cycle_button_states = {} # scheme_name -> last_button_value
        self.app = None # Reference to OrchestratorUI

    def get_mappings_for_scheme(self, scheme_name):
        if scheme_name not in self.scheme_mappings:
            self.scheme_mappings[scheme_name] = [{
                'axes': {},
                'buttons': {},
                'sensitivities': {},
                'inverted': {}
            }]
        return self.scheme_mappings[scheme_name]

    def get_active_mapping_index_for_scheme(self, scheme_name):
        if scheme_name not in self.scheme_active_indices:
            self.scheme_active_indices[scheme_name] = 0
        return self.scheme_active_indices[scheme_name]

    def get_active_mapping(self, scheme_name):
        mappings = self.get_mappings_for_scheme(scheme_name)
        idx = self.get_active_mapping_index_for_scheme(scheme_name)
        if idx >= len(mappings):
            idx = 0
            self.scheme_active_indices[scheme_name] = 0
        return mappings[idx]

    def trigger_ui_cycle_update(self):
        if hasattr(self, 'app') and self.app and self.app.mapping_win:
            GLib.idle_add(self.app.mapping_win.on_mapping_cycled_externally)
        
    def discover_environment(self):
        # 1. Discover Joy Topics
        topic_info = self.get_topic_names_and_types()
        joy_topics = [t[0] for t in topic_info if 'sensor_msgs/msg/Joy' in t[1]]
        
        if len(joy_topics) > 20:
            self.get_logger().warning(f"High number of Joy topics discovered: {len(joy_topics)}")
            
        # 2. Update Input Sources (Nodes that PUBLISH Joy)
        for topic in joy_topics:
            pubs = self.get_publishers_info_by_topic(topic)
            # Filter out publishers from this node
            external_pubs = [p for p in pubs if p.node_name != self.get_name() and p.node_name != f"/{self.get_name()}"]
            if external_pubs:
                if topic not in self.inputs:
                    self.add_input_source(topic)
        
        # 3. Discover Control Schemes (Nodes that SUBSCRIBE to Joy)
        for topic in joy_topics:
            subs = self.get_subscriptions_info_by_topic(topic)
            for sub in subs:
                node_name = sub.node_name
                if not node_name.startswith('/'): node_name = '/' + node_name
                
                # Avoid self-discovery
                if node_name == f"/{self.get_name()}": 
                    continue
                if node_name not in self.control_schemes and node_name not in self.checking_nodes:
                    self.checking_nodes.add(node_name)
                    threading.Thread(target=self.check_if_control_scheme, args=(node_name, topic), daemon=True).start()

    def add_input_source(self, topic):
        pubs = self.get_publishers_info_by_topic(topic)
        external_pubs = [p for p in pubs if p.node_name != self.get_name() and p.node_name != f"/{self.get_name()}"]
        if not external_pubs: return
        
        self.get_logger().info(f"Adding input source: {topic}")
        node_name = external_pubs[0].node_name
        
        self.inputs[topic] = {
            'axis_names': [],
            'button_names': [],
            'last_msg': None,
            'last_time': None,
            'node_name': node_name,
            'sub': self.create_subscription(Joy, topic, lambda msg: self.joy_callback(msg, topic), 1)
        }
        
        # Fetch names
        if node_name != "unknown":
            threading.Thread(target=self.fetch_source_names, args=(topic, node_name), daemon=True).start()

    def fetch_source_names(self, topic, node_name):
        if not node_name.startswith('/'): node_name = '/' + node_name
        self.checking_nodes.add(node_name)
        try:
            # 1. List parameters first
            list_client = self.create_client(ListParameters, f'{node_name}/list_parameters')
            if not list_client.wait_for_service(timeout_sec=1.0): return
            
            list_req = ListParameters.Request()
            list_req.prefixes = ['axis_names', 'button_names']
            future = list_client.call_async(list_req)
            while rclpy.ok() and not future.done(): time.sleep(0.1)
            
            if future.done():
                res = future.result()
                available = res.result.names
                to_get = [p for p in ['axis_names', 'button_names'] if p in available]
                
                if to_get:
                    get_client = self.create_client(GetParameters, f'{node_name}/get_parameters')
                    if not get_client.wait_for_service(timeout_sec=1.0): return
                    get_req = GetParameters.Request()
                    get_req.names = to_get
                    future2 = get_client.call_async(get_req)
                    while rclpy.ok() and not future2.done(): time.sleep(0.1)
                    
                    if future2.done():
                        res2 = future2.result()
                        for i, name in enumerate(to_get):
                            if name == 'axis_names':
                                self.inputs[topic]['axis_names'] = res2.values[i].string_array_value
                            elif name == 'button_names':
                                self.inputs[topic]['button_names'] = res2.values[i].string_array_value
        except Exception as e:
            self.get_logger().error(f"Error fetching names from {node_name}: {e}")
        finally:
            self.checking_nodes.discard(node_name)

    def check_if_control_scheme(self, node_name, topic):
        if not node_name.startswith('/'): node_name = '/' + node_name
        self.checking_nodes.add(node_name)
        try:
            list_client = self.create_client(ListParameters, f'{node_name}/list_parameters')
            if not list_client.wait_for_service(timeout_sec=1.0): return
            
            list_req = ListParameters.Request()
            list_req.prefixes = ['axis_names', 'button_names']
            future = list_client.call_async(list_req)
            while rclpy.ok() and not future.done(): time.sleep(0.1)
            
            if future.done():
                res = future.result()
                available = res.result.names
                to_get = [p for p in ['axis_names', 'button_names'] if p in available]
                
                if len(to_get) > 0:
                    get_client = self.create_client(GetParameters, f'{node_name}/get_parameters')
                    if not get_client.wait_for_service(timeout_sec=1.0): return
                    get_req = GetParameters.Request()
                    get_req.names = to_get
                    future2 = get_client.call_async(get_req)
                    while rclpy.ok() and not future2.done(): time.sleep(0.1)
                    
                    if future2.done():
                        res2 = future2.result()
                        scheme_data = {'axis_names': [], 'button_names': []}
                        for i, name in enumerate(to_get):
                            scheme_data[name] = res2.values[i].string_array_value
                        
                        if scheme_data['axis_names'] or scheme_data['button_names']:
                            scheme_data['topic'] = topic
                            self.control_schemes[node_name] = scheme_data
                            
                            if node_name == self.selected_scheme:
                                self.get_logger().info(f"Activating discovered selected scheme: {node_name}")
                                threading.Thread(target=self.call_scheme_service, args=(node_name, 'activate'), daemon=True).start()
                            else:
                                self.get_logger().info(f"Deactivating discovered non-selected scheme: {node_name}")
                                threading.Thread(target=self.call_scheme_service, args=(node_name, 'deactivate'), daemon=True).start()
        except Exception as e:
            self.get_logger().error(f"Error checking control scheme {node_name}: {e}")
        finally:
            self.checking_nodes.discard(node_name)

    def joy_callback(self, msg, topic):
        self.inputs[topic]['last_msg'] = msg
        self.inputs[topic]['last_time'] = self.get_clock().now()
        
        # Update signal pool for this topic
        axis_names = self.inputs[topic]['axis_names']
        button_names = self.inputs[topic]['button_names']
        
        for i, val in enumerate(msg.axes):
            name = axis_names[i] if i < len(axis_names) else f"axis_{i}"
            self.signal_pool[f"{topic}/{name}"] = val
            
        for i, val in enumerate(msg.buttons):
            name = button_names[i] if i < len(button_names) else f"button_{i}"
            self.signal_pool[f"{topic}/{name}"] = val

    def process_and_publish(self):
        now = time.time()
        # 1. Compute constructions
        if not self.sorted_consts_cache:
            self.sorted_consts_cache = self.get_topological_sort()
            
        for name in self.sorted_consts_cache:
            try:
                self.signal_pool[name] = self.constructions[name].evaluate(self.signal_pool)
            except Exception as e:
                pass
            
        # 2. Map to output
        if not self.selected_scheme or self.selected_scheme not in self.control_schemes:
            return
            
        # Check cycle button first
        cycle_btn = self.scheme_cycle_buttons.get(self.selected_scheme)
        if cycle_btn:
            val = self.signal_pool.get(cycle_btn, 0)
            prev_val = self.prev_cycle_button_states.get(self.selected_scheme, 0)
            if val == 1 and prev_val == 0:
                # Rising edge detected! Cycle active mapping index
                mappings = self.get_mappings_for_scheme(self.selected_scheme)
                if len(mappings) > 1:
                    curr_idx = self.get_active_mapping_index_for_scheme(self.selected_scheme)
                    next_idx = (curr_idx + 1) % len(mappings)
                    self.scheme_active_indices[self.selected_scheme] = next_idx
                    self.trigger_ui_cycle_update()
            self.prev_cycle_button_states[self.selected_scheme] = val

        scheme = self.control_schemes[self.selected_scheme]
        active_mapping = self.get_active_mapping(self.selected_scheme)
        out_msg = Joy()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        
        for name in scheme['axis_names']:
            source = active_mapping['axes'].get(name)
            if source:
                raw_val = self.signal_pool.get(source, 0.0)
                # Apply deadband if configured
                db_pct = active_mapping.get('deadbands', {}).get(name, 0)
                d = db_pct / 100.0
                if abs(raw_val) <= d:
                    raw_val = 0.0
                else:
                    if d < 1.0:
                        raw_val = math.copysign((abs(raw_val) - d) / (1.0 - d), raw_val)
                    else:
                        raw_val = 0.0

                # Apply sensitivity/response curve
                sens = active_mapping.get('sensitivities', {}).get(name, 50)
                if sens <= 50:
                    exponent = 1.0 + (50.0 - sens) / 25.0
                    multiplier = sens / 50.0
                else:
                    exponent = 1.0 - (sens - 50.0) / 100.0
                    multiplier = 1.0 + (sens - 50.0) / 50.0
                out_val = math.copysign(abs(raw_val) ** exponent, raw_val) * multiplier
                out_val = max(-1.0, min(1.0, out_val))
                
                # Apply inversion if checked
                if active_mapping.get('inverted', {}).get(name, False):
                    out_val = -out_val
            else:
                out_val = 0.0
            out_msg.axes.append(out_val)
            
        for name in scheme['button_names']:
            source = active_mapping['buttons'].get(name)
            out_msg.buttons.append(int(self.signal_pool.get(source, 0)) if source else 0)
            
        if self.output_publisher:
            self.output_publisher.publish(out_msg)
            self.last_publish_time = now

    def get_topological_sort(self):
        # Simple Kahn's algorithm or DFS for topo sort
        nodes = list(self.constructions.keys())
        adj = {n: self.constructions[n].get_dependencies() for n in nodes}
        
        sorted_nodes = []
        visited = set()
        temp_visited = set()
        
        def visit(n):
            if n in temp_visited: raise ValueError("Circular dependency detected")
            if n not in visited:
                temp_visited.add(n)
                deps = adj.get(n, [])
                for dep in deps:
                    if dep in self.constructions:
                        visit(dep)
                temp_visited.remove(n)
                visited.add(n)
                sorted_nodes.append(n)
        
        try:
            for n in nodes:
                if n not in visited:
                    visit(n)
        except ValueError as e:
            # self.get_logger().error(str(e))
            return []
            
        return sorted_nodes

    def select_scheme(self, scheme_name):
        if scheme_name == self.selected_scheme: return
        
        # Deactivate previous scheme if any
        prev_scheme = self.selected_scheme
        if prev_scheme:
            self.get_logger().info(f"Deactivating previous scheme: {prev_scheme}")
            threading.Thread(target=self.call_scheme_service, args=(prev_scheme, 'deactivate'), daemon=True).start()
            
        self.selected_scheme = scheme_name
        if self.output_publisher:
            self.destroy_publisher(self.output_publisher)
        
        scheme = self.control_schemes.get(scheme_name)
        if not scheme: return
        
        topic = scheme['topic']
        self.output_publisher = self.create_publisher(Joy, topic, 1)
        self.get_logger().info(f"Selected control scheme: {scheme_name}, publishing to {topic}")
        
        # Activate newly selected scheme
        self.get_logger().info(f"Activating new scheme: {scheme_name}")
        threading.Thread(target=self.call_scheme_service, args=(scheme_name, 'activate'), daemon=True).start()

    def call_scheme_service(self, scheme_name, action):
        from std_srvs.srv import Trigger
        srv_name = f"{scheme_name}/{action}"
        try:
            client = self.create_client(Trigger, srv_name)
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f"Service {srv_name} not available.")
                return
            req = Trigger.Request()
            future = client.call_async(req)
            while rclpy.ok() and not future.done():
                time.sleep(0.05)
            if future.done():
                res = future.result()
                if res.success:
                    self.get_logger().info(f"Successfully called {srv_name}: {res.message}")
                else:
                    self.get_logger().error(f"Failed calling {srv_name}: {res.message}")
        except Exception as e:
            self.get_logger().error(f"Error calling service {srv_name}: {e}")

# --- UI Classes ---

class OrchestratorUI(Adw.Application):
    def __init__(self, node):
        super().__init__(application_id='com.antigravity.teleop_orchestrator',
                         flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.node = node
        self.node.app = self
        self.monitor_win = None
        self.mapping_win = None
        self.const_mgmt_win = None

    def do_startup(self):
        Adw.Application.do_startup(self)
        # CSS for monitor indicators
        css = """
            .monitor-button-off { background-color: #333; border: 1px solid #555; border-radius: 3px; }
            .monitor-button-on { background-color: #fde725; border: 1px solid #fff; border-radius: 3px; }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css.encode())
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def get_all_available_signals(self, signal_type):
        signals = ["None"]
        # Direct inputs
        # Use list() to avoid RuntimeError if inputs changes during iteration from ROS thread
        for topic, data in list(self.node.inputs.items()):
            names = data['axis_names'] if signal_type == 'axis' else data['button_names']
            if not names:
                last_msg = data['last_msg']
                if last_msg:
                    count = len(last_msg.axes if signal_type == 'axis' else last_msg.buttons)
                    names = [f"{'axis' if signal_type == 'axis' else 'button'}_{i}" for i in range(count)]
            for n in names:
                signals.append(f"{topic}/{n}")
        
        # Constructions
        for name, const in list(self.node.constructions.items()):
            if const.out_type == signal_type:
                signals.append(name)
        return signals

    def refresh_all_windows(self):
        if self.mapping_win:
            self.mapping_win.refresh_schemes()
        if self.const_mgmt_win:
            self.const_mgmt_win.refresh_list()
        if self.monitor_win:
            self.monitor_win.refresh_data()

    def do_activate(self):
        self.monitor_win = None
        self.const_mgmt_win = None
        
        # Create Mapping Window (Scheme + Mapping + Save/Load)
        self.mapping_win = MappingWindow(self, self.node)
        self.mapping_win.present()

class MonitorWindow(Gtk.Window):
    def __init__(self, app, node):
        super().__init__(application=app)
        self.node = node
        self.set_title("Orchestrator Monitor")
        self.set_default_size(400, 600)
        self.connect("close-request", self.on_close_request)
        
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.main_box.set_margin_top(10)
        self.main_box.set_margin_bottom(10)
        self.main_box.set_margin_start(10)
        self.main_box.set_margin_end(10)
        self.set_child(self.main_box)
        
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        self.main_box.append(scrolled)
        
        self.content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=20)
        scrolled.set_child(self.content)
        
        self.topic_widgets = {} # name -> {axes: [], buttons: []}
        
        GLib.timeout_add(100, self.refresh_data)
        
    def on_close_request(self, window):
        self.get_application().monitor_win = None
        return False
        
    def refresh_data(self):
        # 1. Update/Add topics
        # Use list() to avoid RuntimeError if inputs changes from ROS thread
        for topic, data in list(self.node.inputs.items()):
            if topic not in self.topic_widgets:
                self.add_topic_view(topic, data)
            self.update_topic_view(topic, data)
            
        # 2. Update/Add constructions
        if "Constructions" not in self.topic_widgets:
            self.add_constructions_view()
        self.update_constructions_view()
        
        return True

    def add_topic_view(self, topic, data):
        self.node.get_logger().info(f"Monitor: Adding view for topic {topic}")
        frame = Gtk.Frame(label=f"Source: {topic}")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        box.set_margin_top(5); box.set_margin_bottom(5); box.set_margin_start(5); box.set_margin_end(5)
        frame.set_child(box)
        self.content.append(frame)
        
        info_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.append(info_box)
        age_label = Gtk.Label(label="Age: ---")
        info_box.append(age_label)
        
        axes_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        btns_box = Gtk.FlowBox(max_children_per_line=8, selection_mode=Gtk.SelectionMode.NONE)
        box.append(axes_box)
        box.append(btns_box)
        
        self.topic_widgets[topic] = {
            'axes_box': axes_box,
            'btns_box': btns_box,
            'age_label': age_label,
            'axis_bars': [],
            'axis_labels': [],
            'btn_indicators': [],
            'btn_labels': []
        }

    def update_topic_view(self, topic, data):
        widgets = self.topic_widgets[topic]
        last_msg = data['last_msg']
        last_time = data['last_time']
        
        if last_time:
            diff = (self.node.get_clock().now() - last_time).nanoseconds / 1e9
            widgets['age_label'].set_text(f"Age: {diff:.2f}s")
        
        if not last_msg: return
        
        # Init widgets if needed
        if not widgets['axis_bars'] and last_msg.axes:
            names = data['axis_names'] or [f"A{i}" for i in range(len(last_msg.axes))]
            for i, name in enumerate(names):
                row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
                lbl = Gtk.Label(label=name)
                lbl.set_width_chars(15)
                lbl.set_ellipsize(Pango.EllipsizeMode.END)
                bar = Gtk.ProgressBar()
                bar.set_hexpand(True)
                row.append(lbl)
                row.append(bar)
                widgets['axes_box'].append(row)
                widgets['axis_bars'].append(bar)
                widgets['axis_labels'].append(lbl)
                
        if not widgets['btn_indicators'] and last_msg.buttons:
            names = data['button_names'] or [f"B{i}" for i in range(len(last_msg.buttons))]
            for i, name in enumerate(names):
                btn_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
                ind = Gtk.Box()
                ind.set_size_request(15, 15)
                ind.set_margin_top(2); ind.set_margin_bottom(2); ind.set_margin_start(2); ind.set_margin_end(2)
                ind.add_css_class('monitor-button-off')
                lbl = Gtk.Label(label=name)
                lbl.add_css_class('caption') # Small text if possible
                btn_vbox.append(ind)
                btn_vbox.append(lbl)
                widgets['btns_box'].append(btn_vbox)
                widgets['btn_indicators'].append(ind)
                widgets['btn_labels'].append(lbl)

        # Update labels if they were previously defaults but names have arrived
        if data['axis_names'] and len(data['axis_names']) == len(widgets['axis_labels']):
            for i, name in enumerate(data['axis_names']):
                if widgets['axis_labels'][i].get_text() != name:
                    widgets['axis_labels'][i].set_text(name)
        
        if data['button_names'] and len(data['button_names']) == len(widgets['btn_labels']):
            for i, name in enumerate(data['button_names']):
                if widgets['btn_labels'][i].get_text() != name:
                    widgets['btn_labels'][i].set_text(name)

        # Update
        for i, val in enumerate(last_msg.axes):
            if i < len(widgets['axis_bars']):
                widgets['axis_bars'][i].set_fraction((val + 1.0) / 2.0)
        
        for i, val in enumerate(last_msg.buttons):
            if i < len(widgets['btn_indicators']):
                if val:
                    widgets['btn_indicators'][i].add_css_class('monitor-button-on')
                    widgets['btn_indicators'][i].remove_css_class('monitor-button-off')
                else:
                    widgets['btn_indicators'][i].add_css_class('monitor-button-off')
                    widgets['btn_indicators'][i].remove_css_class('monitor-button-on')

    def add_constructions_view(self):
        frame = Gtk.Frame(label="Constructed Inputs")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        box.set_margin_top(5); box.set_margin_bottom(5); box.set_margin_start(5); box.set_margin_end(5)
        frame.set_child(box)
        self.content.append(frame)
        
        self.const_axes_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.const_btns_box = Gtk.FlowBox(max_children_per_line=8, selection_mode=Gtk.SelectionMode.NONE)
        box.append(self.const_axes_box)
        box.append(self.const_btns_box)
        
        self.topic_widgets["Constructions"] = {
            'widgets': {} # name -> widget
        }

    def update_constructions_view(self):
        widgets = self.topic_widgets["Constructions"]
        for name, const in list(self.node.constructions.items()):
            val = self.node.signal_pool.get(name, 0.0)
            if name not in widgets['widgets']:
                if const.out_type == 'axis':
                    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
                    lbl = Gtk.Label(label=name)
                    lbl.set_width_chars(15)
                    lbl.set_ellipsize(Pango.EllipsizeMode.END)
                    bar = Gtk.ProgressBar()
                    bar.set_hexpand(True)
                    row.append(lbl)
                    row.append(bar)
                    self.const_axes_box.append(row)
                    widgets['widgets'][name] = bar
                else:
                    btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
                    ind = Gtk.Box()
                    ind.set_size_request(15, 15)
                    ind.set_margin_top(2)
                    ind.set_margin_bottom(2)
                    ind.set_margin_start(2)
                    ind.set_margin_end(2)
                    lbl = Gtk.Label(label=name[:5])
                    btn_box.append(ind)
                    btn_box.append(lbl)
                    self.const_btns_box.append(btn_box)
                    widgets['widgets'][name] = ind
            
            # Update
            w = widgets['widgets'][name]
            if const.out_type == 'axis':
                w.set_fraction((val + 1.0) / 2.0)
            else:
                if val:
                    w.add_css_class('monitor-button-on')
                    w.remove_css_class('monitor-button-off')
                else:
                    w.add_css_class('monitor-button-off')
                    w.remove_css_class('monitor-button-on')
        
        # Cleanup deleted constructions
        current_names = set(self.node.constructions.keys())
        for name in list(widgets['widgets'].keys()):
            if name not in current_names:
                # For simplicity we'll just leave them for now, 
                # or a full refresh would be needed.
                pass

class ConstructionManagementWindow(Gtk.Window):
    def __init__(self, app, node):
        super().__init__(application=app)
        self.node = node
        self.set_title("Signal Construction")
        self.set_default_size(400, 500)
        self.connect("close-request", self.on_close_request)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(10); box.set_margin_bottom(10); box.set_margin_start(10); box.set_margin_end(10)
        self.set_child(box)
        
        box.append(Gtk.Label(label="Manage Constructed Inputs", xalign=0))
        
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        box.append(scrolled)
        
        self.const_list = Gtk.ListBox()
        scrolled.set_child(self.const_list)
        
        add_btn = Gtk.Button(label="Add New Construction")
        add_btn.connect('clicked', self.on_add_const_clicked)
        box.append(add_btn)
        
        self.refresh_list()
        
    def on_close_request(self, window):
        self.get_application().const_mgmt_win = None
        return False
        
    def refresh_list(self):
        for child in list(self.const_list): self.const_list.remove(child)
        for name in self.node.constructions:
            self.add_to_const_list(name)
            
    def add_to_const_list(self, name):
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        lbl = Gtk.Label(label=name, xalign=0)
        lbl.set_hexpand(True)
        row.append(lbl)
        
        edit_btn = Gtk.Button(icon_name="document-edit-symbolic")
        edit_btn.connect('clicked', lambda b, n=name: self.on_edit_const(n))
        row.append(edit_btn)
        
        del_btn = Gtk.Button(icon_name="user-trash-symbolic")
        del_btn.connect('clicked', lambda b, n=name: self.on_delete_const(n))
        row.append(del_btn)
        self.const_list.append(row)

    def on_edit_const(self, name):
        dialog = ConstructionDialog(self, self.node, self.get_application(), edit_name=name)
        dialog.present()

    def on_delete_const(self, name):
        for n, c in list(self.node.constructions.items()):
            if name in c.get_dependencies():
                self.node.get_logger().error(f"Cannot delete {name}, used by {n}")
                return
        del self.node.constructions[name]
        self.node.sorted_consts_cache = [] # Invalidate cache
        self.get_application().refresh_all_windows()

    def on_add_const_clicked(self, btn):
        dialog = ConstructionDialog(self, self.node, self.get_application())
        dialog.present()

    def get_all_available_signals(self, signal_type):
        return self.get_application().get_all_available_signals(signal_type)
class MappingWindow(Gtk.Window):
    def __init__(self, app, node):
        super().__init__(application=app)
        self.node = node
        self.app = app
        self.set_title("Orchestrator Mapping")
        self.set_default_size(1150, 650) # Extra width and height to comfortably display sliders/checkboxes
        self.connect("close-request", self.on_close_request)
        
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.set_margin_top(20)
        main_box.set_margin_bottom(20)
        main_box.set_margin_start(20)
        main_box.set_margin_end(20)
        self.set_child(main_box)
        
        # Persistence (Top)
        top_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.append(top_bar)
        
        save_btn = Gtk.Button(label="Save Config")
        save_btn.connect('clicked', self.on_save_clicked)
        top_bar.append(save_btn)
        
        load_btn = Gtk.Button(label="Load Config")
        load_btn.connect('clicked', self.on_load_clicked)
        top_bar.append(load_btn)
        
        options_btn = Gtk.Button(label="Options")
        options_btn.connect('clicked', self.on_options_clicked)
        top_bar.append(options_btn)
        
        monitor_btn = Gtk.Button(label="Monitor")
        monitor_btn.connect('clicked', self.on_monitor_clicked)
        top_bar.append(monitor_btn)
        
        const_btn = Gtk.Button(label="Signal Construction")
        const_btn.connect('clicked', self.on_const_clicked)
        top_bar.append(const_btn)
        
        # Scheme Selection & Cycle Button Configuration
        scheme_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        scheme_box.append(Gtk.Label(label="Target Control Scheme:"))
        self.scheme_combo = Gtk.DropDown()
        scheme_box.append(self.scheme_combo)
        
        scheme_box.append(Gtk.Label(label="Cycle Button:"))
        self.cycle_combo = Gtk.DropDown()
        scheme_box.append(self.cycle_combo)
        
        main_box.append(scheme_box)
        
        self.scheme_combo.connect("notify::selected-item", self.on_scheme_selected)
        self.cycle_combo.connect("notify::selected-item", self.on_cycle_button_selected)
        
        # Notebook for mapping tabs
        self.mapping_notebook = Gtk.Notebook()
        self.mapping_notebook.set_vexpand(True)
        main_box.append(self.mapping_notebook)
        
        # Connect switch-page and keep the handler ID
        self.tab_switch_handler_id = self.mapping_notebook.connect("switch-page", self.on_tab_switched)
        
        self.last_schemes = []
        self.current_scheme_name = None
        self.last_axes = []
        self.last_btns = []
        GLib.timeout_add(2000, self.refresh_schemes)
        
    def refresh_schemes(self):
        schemes = sorted(list(self.node.control_schemes.keys()))
        if schemes != self.last_schemes:
            self.last_schemes = schemes
            self.scheme_combo.set_model(Gtk.StringList.new(schemes))
            
        # Check if signals changed
        all_axes = self.get_all_available_signals('axis')
        all_btns = self.get_all_available_signals('button')
        if all_axes != self.last_axes or all_btns != self.last_btns:
            self.last_axes = all_axes
            self.last_btns = all_btns
            if self.current_scheme_name:
                self.update_cycle_button_dropdown()
                GLib.idle_add(self.rebuild_notebook_deferred)
        return True

    def on_scheme_selected(self, drop, param):
        selected = drop.get_selected_item()
        if not selected: return
        name = selected.get_string()
        if name == self.current_scheme_name: return
        
        self.current_scheme_name = name
        self.node.select_scheme(name)
        self.update_cycle_button_dropdown()
        GLib.idle_add(self.rebuild_notebook_deferred)

    def update_cycle_button_dropdown(self):
        if not self.current_scheme_name: return
        # Disconnect signal temporarily
        try:
            self.cycle_combo.disconnect_by_func(self.on_cycle_button_selected)
        except:
            pass
            
        all_btns = self.get_all_available_signals('button')
        self.cycle_combo.set_model(Gtk.StringList.new(all_btns))
        
        current = self.node.scheme_cycle_buttons.get(self.current_scheme_name)
        if current in all_btns:
            self.cycle_combo.set_selected(all_btns.index(current))
        else:
            self.cycle_combo.set_selected(0) # None
            
        self.cycle_combo.connect("notify::selected-item", self.on_cycle_button_selected)

    def on_cycle_button_selected(self, drop, param):
        if not self.current_scheme_name: return
        item = drop.get_selected_item()
        val = item.get_string() if item else "None"
        
        if val == "None":
            self.node.scheme_cycle_buttons.pop(self.current_scheme_name, None)
        else:
            self.node.scheme_cycle_buttons[self.current_scheme_name] = val
            # Clear button mapping if it was previously mapped to an output button in any mapping of this scheme
            mappings = self.node.get_mappings_for_scheme(self.current_scheme_name)
            for m in mappings:
                for out_name, mapped_btn in list(m['buttons'].items()):
                    if mapped_btn == val:
                        m['buttons'].pop(out_name, None)
                        
        GLib.idle_add(self.rebuild_notebook_deferred)

    def rebuild_notebook_deferred(self):
        self.rebuild_notebook()
        return False # Ensure only runs once

    def rebuild_notebook(self):
        # Block switch-page callback to prevent recursive triggers and state corruption
        if hasattr(self, 'tab_switch_handler_id') and self.tab_switch_handler_id:
            self.mapping_notebook.handler_block(self.tab_switch_handler_id)
            
        # Clear all pages
        while self.mapping_notebook.get_n_pages() > 0:
            self.mapping_notebook.remove_page(0)
            
        scheme_name = self.current_scheme_name
        if not scheme_name or scheme_name not in self.node.control_schemes:
            if hasattr(self, 'tab_switch_handler_id') and self.tab_switch_handler_id:
                self.mapping_notebook.handler_unblock(self.tab_switch_handler_id)
            return
            
        mappings = self.node.get_mappings_for_scheme(scheme_name)
        active_idx = self.node.get_active_mapping_index_for_scheme(scheme_name)
        
        all_axes = self.get_all_available_signals('axis')
        all_btns = self.get_all_available_signals('button')
        cycle_btn = self.node.scheme_cycle_buttons.get(scheme_name)
        filtered_btns = [b for b in all_btns if b != cycle_btn]
        
        scheme = self.node.control_schemes[scheme_name]
        
        for i, mapping in enumerate(mappings):
            page_widget = self.create_mapping_page_widget(scheme, mapping, all_axes, filtered_btns)
            
            # Custom Tab Label
            tab_label_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            lbl = Gtk.Label(label=f"Mapping {i+1}")
            tab_label_box.append(lbl)
            
            if len(mappings) > 1:
                close_btn = Gtk.Button(icon_name="window-close-symbolic")
                close_btn.set_has_frame(False)
                close_btn.connect("clicked", lambda b, idx=i: self.on_delete_tab_clicked(idx))
                tab_label_box.append(close_btn)
                
            self.mapping_notebook.append_page(page_widget, tab_label_box)
            
        # Add the "+" Tab page
        plus_page = Gtk.Box()
        plus_lbl = Gtk.Label(label="+")
        self.mapping_notebook.append_page(plus_page, plus_lbl)
        
        if active_idx < len(mappings):
            self.mapping_notebook.set_current_page(active_idx)
        else:
            self.mapping_notebook.set_current_page(0)
            self.node.scheme_active_indices[scheme_name] = 0
            
        if hasattr(self, 'tab_switch_handler_id') and self.tab_switch_handler_id:
            self.mapping_notebook.handler_unblock(self.tab_switch_handler_id)

    def on_tab_switched(self, notebook, page, page_num):
        if not self.current_scheme_name: return
        mappings = self.node.get_mappings_for_scheme(self.current_scheme_name)
        
        # If switched to the last page ("+")
        if page_num == len(mappings):
            new_mapping = {
                'axes': {},
                'buttons': {},
                'sensitivities': {},
                'inverted': {}
            }
            mappings.append(new_mapping)
            self.node.scheme_active_indices[self.current_scheme_name] = len(mappings) - 1
            GLib.idle_add(self.rebuild_notebook_deferred)
        else:
            self.node.scheme_active_indices[self.current_scheme_name] = page_num

    def create_mapping_page_widget(self, scheme, mapping, all_axes, filtered_btns):
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        
        cols_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        cols_box.set_margin_top(10)
        cols_box.set_margin_bottom(10)
        cols_box.set_margin_start(10)
        cols_box.set_margin_end(10)
        scrolled.set_child(cols_box)
        
        # Axis mapping box (Left)
        axis_mapping_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        axis_mapping_box.set_hexpand(True)
        axis_mapping_box.append(Gtk.Label(label="Axis Mapping", xalign=0))
        cols_box.append(axis_mapping_box)
        
        lbl_size_group = Gtk.SizeGroup(mode=Gtk.SizeGroupMode.HORIZONTAL)
        drop_size_group = Gtk.SizeGroup(mode=Gtk.SizeGroupMode.HORIZONTAL)
        
        for axis_name in scheme['axis_names']:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            
            lbl = Gtk.Label(label=axis_name, xalign=1.0)
            lbl_size_group.add_widget(lbl)
            row.append(lbl)
            
            drop = Gtk.DropDown.new_from_strings(all_axes)
            drop_size_group.add_widget(drop)
            current = mapping['axes'].get(axis_name)
            if current in all_axes:
                drop.set_selected(all_axes.index(current))
            drop.connect("notify::selected-item", lambda d, p, n=axis_name, m=mapping: self.on_axis_mapped_in_page(n, d, m))
            row.append(drop)
            
            invert_chk = Gtk.CheckButton(label="Invert")
            curr_invert = mapping.get('inverted', {}).get(axis_name, False)
            invert_chk.set_active(curr_invert)
            invert_chk.connect("toggled", lambda cb, n=axis_name, m=mapping: self.on_axis_inverted_toggled(n, cb, m))
            row.append(invert_chk)
            
            curr_sens = mapping.get('sensitivities', {}).get(axis_name, 50)
            sens_adj = Gtk.Adjustment.new(curr_sens, 0.0, 100.0, 1.0, 10.0, 0.0)
            sens_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=sens_adj)
            sens_scale.set_hexpand(True)
            sens_scale.set_draw_value(False)
            
            sens_lbl = Gtk.Label(label=f"Sens: {int(curr_sens)}")
            sens_lbl.set_width_chars(10)
            
            sens_scale.connect("value-changed", lambda s, n=axis_name, m=mapping, l=sens_lbl: self.on_axis_sensitivity_changed(n, s, m, l))
            row.append(sens_scale)
            row.append(sens_lbl)
            
            curr_db = mapping.get('deadbands', {}).get(axis_name, 0)
            db_adj = Gtk.Adjustment.new(curr_db, 0.0, 100.0, 1.0, 10.0, 0.0)
            db_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=db_adj)
            db_scale.set_hexpand(True)
            db_scale.set_draw_value(False)
            
            db_lbl = Gtk.Label(label=f"Dead: {curr_db / 100.0:.2f}")
            db_lbl.set_width_chars(11)
            
            db_scale.connect("value-changed", lambda s, n=axis_name, m=mapping, l=db_lbl: self.on_axis_deadband_changed(n, s, m, l))
            row.append(db_scale)
            row.append(db_lbl)
            
            axis_mapping_box.append(row)
            
        # Button mapping box (Right)
        btn_mapping_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        btn_mapping_box.set_hexpand(True)
        btn_mapping_box.append(Gtk.Label(label="Button Mapping", xalign=0))
        cols_box.append(btn_mapping_box)
        
        btn_lbl_size_group = Gtk.SizeGroup(mode=Gtk.SizeGroupMode.HORIZONTAL)
        
        for btn_name in scheme['button_names']:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            
            lbl = Gtk.Label(label=btn_name, xalign=1.0)
            btn_lbl_size_group.add_widget(lbl)
            row.append(lbl)
            
            drop = Gtk.DropDown.new_from_strings(filtered_btns)
            current = mapping['buttons'].get(btn_name)
            if current in filtered_btns:
                drop.set_selected(filtered_btns.index(current))
            drop.connect("notify::selected-item", lambda d, p, n=btn_name, m=mapping: self.on_btn_mapped_in_page(n, d, m))
            row.append(drop)
            
            btn_mapping_box.append(row)
            
        return scrolled

    def on_axis_mapped_in_page(self, out_name, drop, mapping):
        item = drop.get_selected_item()
        val = item.get_string() if item else "None"
        if val == "None":
            mapping['axes'].pop(out_name, None)
        else:
            mapping['axes'][out_name] = val

    def on_axis_inverted_toggled(self, out_name, check_btn, mapping):
        if 'inverted' not in mapping:
            mapping['inverted'] = {}
        mapping['inverted'][out_name] = check_btn.get_active()

    def on_axis_sensitivity_changed(self, out_name, scale, mapping, lbl):
        val = int(scale.get_value())
        lbl.set_text(f"Sens: {val}")
        if 'sensitivities' not in mapping:
            mapping['sensitivities'] = {}
        mapping['sensitivities'][out_name] = val

    def on_axis_deadband_changed(self, out_name, scale, mapping, lbl):
        val = int(scale.get_value())
        lbl.set_text(f"Dead: {val / 100.0:.2f}")
        if 'deadbands' not in mapping:
            mapping['deadbands'] = {}
        mapping['deadbands'][out_name] = val

    def on_btn_mapped_in_page(self, out_name, drop, mapping):
        item = drop.get_selected_item()
        val = item.get_string() if item else "None"
        if val == "None":
            mapping['buttons'].pop(out_name, None)
        else:
            mapping['buttons'][out_name] = val

    def on_delete_tab_clicked(self, idx):
        if not self.current_scheme_name: return
        mappings = self.node.get_mappings_for_scheme(self.current_scheme_name)
        if len(mappings) <= 1: return
        
        mappings.pop(idx)
        
        curr_idx = self.node.scheme_active_indices.get(self.current_scheme_name, 0)
        if curr_idx >= len(mappings):
            self.node.scheme_active_indices[self.current_scheme_name] = len(mappings) - 1
        elif curr_idx > idx:
            self.node.scheme_active_indices[self.current_scheme_name] = curr_idx - 1
            
        GLib.idle_add(self.rebuild_notebook_deferred)

    def on_mapping_cycled_externally(self):
        if not self.current_scheme_name: return False
        active_idx = self.node.get_active_mapping_index_for_scheme(self.current_scheme_name)
        if self.mapping_notebook.get_current_page() != active_idx:
            if hasattr(self, 'tab_switch_handler_id') and self.tab_switch_handler_id:
                self.mapping_notebook.handler_block(self.tab_switch_handler_id)
            self.mapping_notebook.set_current_page(active_idx)
            if hasattr(self, 'tab_switch_handler_id') and self.tab_switch_handler_id:
                self.mapping_notebook.handler_unblock(self.tab_switch_handler_id)
        return False

    def on_save_clicked(self, btn):
        data = {
            'options': {
                'throttling_enabled': self.node.throttling_enabled,
                'periodic_publish': self.node.periodic_publish,
                'max_rate_hz': self.node.max_rate_hz,
                'selected_scheme': self.node.selected_scheme
            },
            'scheme_mappings': self.node.scheme_mappings,
            'scheme_cycle_buttons': self.node.scheme_cycle_buttons,
            'constructed_inputs': {}
        }
        for name, c in list(self.node.constructions.items()):
            c_data = {'type': c.__class__.__name__, 'inputs': c.inputs}
            if isinstance(c, ButtonToAxis):
                c_data['low'] = c.low
                c_data['high'] = c.high
            if isinstance(c, AxisToButton): c_data['cutoff'] = c.cutoff
            if isinstance(c, LogicalButton): c_data['op'] = c.op
            if isinstance(c, AxisOverride): c_data['threshold'] = c.threshold
            if isinstance(c, DeadbandAxis):
                c_data['low'] = c.low
                c_data['high'] = c.high
            if isinstance(c, AxisButtonControl): c_data['mode'] = c.mode
            data['constructed_inputs'][name] = c_data
            
        file_chooser = Gtk.FileDialog(title="Save Config")
        file_chooser.save(self, None, self.on_save_file_selected, data)

    def on_save_file_selected(self, dialog, result, data):
        try:
            file = dialog.save_finish(result)
            if file:
                path = file.get_path()
                with open(path, 'w') as f:
                    yaml.dump(data, f)
        except: pass

    def on_load_clicked(self, btn):
        file_chooser = Gtk.FileDialog(title="Load Config")
        file_chooser.open(self, None, self.on_load_file_selected)

    def on_load_file_selected(self, dialog, result):
        try:
            file = dialog.open_finish(result)
            if file:
                path = file.get_path()
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.apply_config(data)
        except: pass

    def apply_config(self, data):
        # 1. Load Constructions
        for name, c_data in data.get('constructed_inputs', {}).items():
            if name in self.node.constructions: continue
            self.node.sorted_consts_cache = [] # Invalidate cache
            c_type = c_data['type']
            inputs = c_data['inputs']
            try:
                if c_type == 'AxisToButton': c = AxisToButton(name, inputs[0], c_data['cutoff'])
                elif c_type == 'ButtonToAxis': c = ButtonToAxis(name, inputs[0], c_data.get('low', 0.0), c_data.get('high', 1.0))
                elif c_type == 'TwoButtonsToAxis': c = TwoButtonsToAxis(name, inputs[0], inputs[1])
                elif c_type == 'AverageAxis': c = AverageAxis(name, inputs[0], inputs[1])
                elif c_type == 'InvertAxis': c = InvertAxis(name, inputs[0])
                elif c_type == 'SquareAxis': c = SquareAxis(name, inputs[0])
                elif c_type == 'SqrtAxis': c = SqrtAxis(name, inputs[0])
                elif c_type == 'LogicalButton': c = LogicalButton(name, inputs[0], inputs[1] if len(inputs)>1 else None, c_data['op'])
                elif c_type == 'AxisOverride': c = AxisOverride(name, inputs[0], inputs[1], c_data['threshold'])
                elif c_type == 'ButtonOverride': c = ButtonOverride(name, inputs[0], inputs[1])
                elif c_type == 'DeadbandAxis': c = DeadbandAxis(name, inputs[0], c_data['low'], c_data['high'])
                elif c_type == 'AxisButtonControl': c = AxisButtonControl(name, inputs[0], inputs[1], c_data['mode'])
                self.node.constructions[name] = c
            except Exception as e:
                self.node.get_logger().error(f"Failed to load construction {name}: {e}")
        
        # 2. Update Mappings directly
        self.node.scheme_mappings = data.get('scheme_mappings', {})
        self.node.scheme_cycle_buttons = data.get('scheme_cycle_buttons', {})
        self.node.scheme_active_indices = {} # Reset to 0 defaults on load
        
        # 3. Update Options
        opts = data.get('options', {})
        self.node.throttling_enabled = opts.get('throttling_enabled', False)
        self.node.periodic_publish = opts.get('periodic_publish', False)
        self.node.max_rate_hz = opts.get('max_rate_hz', 30.0)
        
        if opts.get('selected_scheme'):
            self.node.select_scheme(opts['selected_scheme'])
        
        # 4. Refresh All Windows
        if self.node.selected_scheme:
            self.update_cycle_button_dropdown()
            GLib.idle_add(self.rebuild_notebook_deferred)
        if self.app.const_mgmt_win:
            self.app.const_mgmt_win.refresh_list()

    def on_options_clicked(self, btn):
        dialog = OptionsDialog(self, self.node)
        dialog.present()

    def on_monitor_clicked(self, btn):
        if not self.app.monitor_win:
            self.app.monitor_win = MonitorWindow(self.app, self.node)
        self.app.monitor_win.present()

    def on_const_clicked(self, btn):
        if not self.app.const_mgmt_win:
            self.app.const_mgmt_win = ConstructionManagementWindow(self.app, self.node)
        self.app.const_mgmt_win.present()

    def on_close_request(self, window):
        if self.app.monitor_win:
            try:
                self.app.monitor_win.destroy()
            except:
                pass
        if self.app.const_mgmt_win:
            try:
                self.app.const_mgmt_win.destroy()
            except:
                pass
        self.app.quit()
        return False

    def get_all_available_signals(self, signal_type):
        return self.app.get_all_available_signals(signal_type)

class ConstructionDialog(Gtk.Window):
    def __init__(self, parent, node, app, edit_name=None):
        super().__init__(application=app, transient_for=parent, modal=True)
        self.node = node
        self.parent_win = parent
        self.edit_name = edit_name
        self.set_title("Edit Construction" if edit_name else "Add Construction")
        self.set_default_size(400, 400)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(20); box.set_margin_bottom(20); box.set_margin_start(20); box.set_margin_end(20)
        self.set_child(box)
        
        # Name
        name_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        name_box.append(Gtk.Label(label="Name:"))
        self.name_entry = Gtk.Entry(placeholder_text="Enter name")
        name_box.append(self.name_entry)
        box.append(name_box)
        
        # Type
        type_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        type_box.append(Gtk.Label(label="Type:"))
        self.types = ["AxisToButton", "ButtonToAxis", "TwoButtonsToAxis", "AverageAxis", "InvertAxis", "SquareAxis", "SqrtAxis", "LogicalButton", "AxisOverride", "ButtonOverride", "DeadbandAxis", "AxisButtonControl"]
        self.type_combo = Gtk.DropDown.new_from_strings(self.types)
        type_box.append(self.type_combo)
        box.append(type_box)
        
        if self.edit_name:
            self.name_entry.set_text(self.edit_name)
            self.name_entry.set_sensitive(False) # Don't allow renaming for now to avoid dep break
            const = self.node.constructions[self.edit_name]
            type_name = const.__class__.__name__
            if type_name in self.types:
                self.type_combo.set_selected(self.types.index(type_name))
        
        self.params_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.append(self.params_box)
        
        self.type_combo.connect("notify::selected-item", self.on_type_changed)
        
        add_btn = Gtk.Button(label="Update" if self.edit_name else "Add")
        add_btn.connect('clicked', self.on_add_clicked)
        box.append(add_btn)
        
        self.on_type_changed(self.type_combo, None)
        
        if self.edit_name:
            self.prefill_data()

    def prefill_data(self):
        const = self.node.constructions[self.edit_name]
        t = const.__class__.__name__
        
        def set_combo_by_string(combo, s):
            model = combo.get_model()
            for i in range(model.get_n_items()):
                if model.get_item(i).get_string() == s:
                    combo.set_selected(i)
                    return

        try:
            if t == "AxisToButton":
                set_combo_by_string(self.input1, const.inputs[0])
                self.cutoff.set_value(const.cutoff)
            elif t == "ButtonToAxis":
                set_combo_by_string(self.input1, const.inputs[0])
                self.low.set_value(const.low)
                self.high.set_value(const.high)
            elif t == "TwoButtonsToAxis":
                set_combo_by_string(self.input1, const.inputs[0])
                set_combo_by_string(self.input2, const.inputs[1])
            elif t == "AverageAxis":
                set_combo_by_string(self.input1, const.inputs[0])
                set_combo_by_string(self.input2, const.inputs[1])
            elif t in ["InvertAxis", "SquareAxis", "SqrtAxis"]:
                set_combo_by_string(self.input1, const.inputs[0])
            elif t == "LogicalButton":
                set_combo_by_string(self.input1, const.inputs[0])
                if len(const.inputs) > 1: set_combo_by_string(self.input2, const.inputs[1])
                else: self.input2.set_selected(0) # None
                set_combo_by_string(self.op, const.op)
            elif t == "AxisOverride":
                set_combo_by_string(self.input1, const.inputs[0])
                set_combo_by_string(self.input2, const.inputs[1])
                self.threshold.set_value(const.threshold)
            elif t == "ButtonOverride":
                set_combo_by_string(self.input1, const.inputs[0])
                set_combo_by_string(self.input2, const.inputs[1])
            elif t == "DeadbandAxis":
                set_combo_by_string(self.input1, const.inputs[0])
                self.low.set_value(const.low)
                self.high.set_value(const.high)
            elif t == "AxisButtonControl":
                set_combo_by_string(self.input1, const.inputs[0])
                set_combo_by_string(self.input2, const.inputs[1])
                set_combo_by_string(self.mode, const.mode)
        except Exception as e:
            self.node.get_logger().error(f"Error prefilling: {e}")

    def on_type_changed(self, drop, param):
        for child in list(self.params_box): self.params_box.remove(child)
        t = self.types[drop.get_selected()]
        
        all_axes = self.parent_win.get_all_available_signals('axis')
        all_btns = self.parent_win.get_all_available_signals('button')
        
        if t == "AxisToButton":
            self.input1 = Gtk.DropDown.new_from_strings(all_axes)
            self.cutoff = Gtk.SpinButton.new_with_range(-1.0, 1.0, 0.1)
            self.params_box.append(Gtk.Label(label="Axis:"))
            self.params_box.append(self.input1)
            self.params_box.append(Gtk.Label(label="Cutoff:"))
            self.params_box.append(self.cutoff)
        elif t == "ButtonToAxis":
            self.input1 = Gtk.DropDown.new_from_strings(all_btns)
            self.low = Gtk.SpinButton.new_with_range(-1.0, 1.0, 0.1)
            self.high = Gtk.SpinButton.new_with_range(-1.0, 1.0, 0.1)
            self.high.set_value(1.0)
            self.params_box.append(Gtk.Label(label="Button:"))
            self.params_box.append(self.input1)
            self.params_box.append(Gtk.Label(label="Low Value:"))
            self.params_box.append(self.low)
            self.params_box.append(Gtk.Label(label="High Value:"))
            self.params_box.append(self.high)
        elif t == "TwoButtonsToAxis":
            self.input1 = Gtk.DropDown.new_from_strings(all_btns)
            self.input2 = Gtk.DropDown.new_from_strings(all_btns)
            self.params_box.append(Gtk.Label(label="Negative Button:"))
            self.params_box.append(self.input1)
            self.params_box.append(Gtk.Label(label="Positive Button:"))
            self.params_box.append(self.input2)
        elif t == "AverageAxis":
            self.input1 = Gtk.DropDown.new_from_strings(all_axes)
            self.input2 = Gtk.DropDown.new_from_strings(all_axes)
            self.params_box.append(Gtk.Label(label="Axis 1:"))
            self.params_box.append(self.input1)
            self.params_box.append(Gtk.Label(label="Axis 2:"))
            self.params_box.append(self.input2)
        elif t in ["InvertAxis", "SquareAxis", "SqrtAxis"]:
            self.input1 = Gtk.DropDown.new_from_strings(all_axes)
            self.params_box.append(Gtk.Label(label="Axis:"))
            self.params_box.append(self.input1)
        elif t == "LogicalButton":
            self.input1 = Gtk.DropDown.new_from_strings(all_btns)
            self.input2 = Gtk.DropDown.new_from_strings(all_btns)
            self.op = Gtk.DropDown.new_from_strings(["AND", "OR", "NAND", "NOR", "NOT"])
            self.params_box.append(Gtk.Label(label="Button 1:"))
            self.params_box.append(self.input1)
            self.params_box.append(Gtk.Label(label="Button 2 (optional):"))
            self.params_box.append(self.input2)
            self.params_box.append(Gtk.Label(label="Operation:"))
            self.params_box.append(self.op)
        elif t == "AxisOverride":
            self.input1 = Gtk.DropDown.new_from_strings(all_axes)
            self.input2 = Gtk.DropDown.new_from_strings(all_axes)
            self.threshold = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.05)
            self.params_box.append(Gtk.Label(label="Primary Axis:"))
            self.params_box.append(self.input1)
            self.params_box.append(Gtk.Label(label="Secondary Axis:"))
            self.params_box.append(self.input2)
            self.params_box.append(Gtk.Label(label="Threshold:"))
            self.params_box.append(self.threshold)
        elif t == "ButtonOverride":
            self.input1 = Gtk.DropDown.new_from_strings(all_btns)
            self.input2 = Gtk.DropDown.new_from_strings(all_btns)
            self.params_box.append(Gtk.Label(label="Priority Button:"))
            self.params_box.append(self.input1)
            self.params_box.append(Gtk.Label(label="Secondary Button:"))
            self.params_box.append(self.input2)
        elif t == "DeadbandAxis":
            self.input1 = Gtk.DropDown.new_from_strings(all_axes)
            self.low = Gtk.SpinButton.new_with_range(-1.0, 0.0, 0.05)
            self.high = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.05)
            self.params_box.append(Gtk.Label(label="Axis:"))
            self.params_box.append(self.input1)
            self.params_box.append(Gtk.Label(label="Low Limit:"))
            self.params_box.append(self.low)
            self.params_box.append(Gtk.Label(label="High Limit:"))
            self.params_box.append(self.high)
        elif t == "AxisButtonControl":
            self.input1 = Gtk.DropDown.new_from_strings(all_axes)
            self.input2 = Gtk.DropDown.new_from_strings(all_btns)
            self.mode = Gtk.DropDown.new_from_strings(["START", "STOP", "TOGGLE"])
            self.params_box.append(Gtk.Label(label="Axis:"))
            self.params_box.append(self.input1)
            self.params_box.append(Gtk.Label(label="Button:"))
            self.params_box.append(self.input2)
            self.params_box.append(Gtk.Label(label="Mode:"))
            self.params_box.append(self.mode)

    def on_add_clicked(self, btn):
        name = self.name_entry.get_text()
        t = self.types[self.type_combo.get_selected()]
        if not name: name = f"{t.lower()}_{len(self.node.constructions)}"
        
        try:
            if t == "AxisToButton":
                c = AxisToButton(name, self.input1.get_selected_item().get_string(), self.cutoff.get_value())
            elif t == "ButtonToAxis":
                c = ButtonToAxis(name, self.input1.get_selected_item().get_string(), self.low.get_value(), self.high.get_value())
            elif t == "TwoButtonsToAxis":
                c = TwoButtonsToAxis(name, self.input1.get_selected_item().get_string(), self.input2.get_selected_item().get_string())
            elif t == "AverageAxis":
                c = AverageAxis(name, self.input1.get_selected_item().get_string(), self.input2.get_selected_item().get_string())
            elif t == "InvertAxis":
                c = InvertAxis(name, self.input1.get_selected_item().get_string())
            elif t == "SquareAxis":
                c = SquareAxis(name, self.input1.get_selected_item().get_string())
            elif t == "SqrtAxis":
                c = SqrtAxis(name, self.input1.get_selected_item().get_string())
            elif t == "LogicalButton":
                c = LogicalButton(name, self.input1.get_selected_item().get_string(), self.input2.get_selected_item().get_string() if self.input2.get_selected_item().get_string() != "None" else None, self.op.get_selected_item().get_string())
            elif t == "AxisOverride":
                c = AxisOverride(name, self.input1.get_selected_item().get_string(), self.input2.get_selected_item().get_string(), self.threshold.get_value())
            elif t == "ButtonOverride":
                c = ButtonOverride(name, self.input1.get_selected_item().get_string(), self.input2.get_selected_item().get_string())
            elif t == "DeadbandAxis":
                c = DeadbandAxis(name, self.input1.get_selected_item().get_string(), self.low.get_value(), self.high.get_value())
            elif t == "AxisButtonControl":
                c = AxisButtonControl(name, self.input1.get_selected_item().get_string(), self.input2.get_selected_item().get_string(), self.mode.get_selected_item().get_string())
            
            # Check for circular deps
            old_const = self.node.constructions.get(name)
            self.node.constructions[name] = c
            self.node.sorted_consts_cache = [] # Invalidate cache
            if not self.node.get_topological_sort():
                if old_const: self.node.constructions[name] = old_const
                else: del self.node.constructions[name]
                raise ValueError("Circular dependency detected!")
            
            self.get_application().refresh_all_windows()
            self.destroy()
        except Exception as e:
            self.node.get_logger().error(f"Failed to add construction: {e}")

class OptionsDialog(Gtk.Window):
    def __init__(self, parent, node):
        super().__init__(transient_for=parent, modal=True)
        self.node = node
        self.set_title("Orchestrator Options")
        self.set_default_size(300, 200)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(20); box.set_margin_bottom(20); box.set_margin_start(20); box.set_margin_end(20)
        self.set_child(box)
        
        self.throttle_sw = Gtk.Switch(active=node.throttling_enabled)
        row1 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row1.append(Gtk.Label(label="Enable Throttling:"))
        row1.append(self.throttle_sw)
        box.append(row1)
        
        self.periodic_sw = Gtk.Switch(active=node.periodic_publish)
        row2 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row2.append(Gtk.Label(label="Force Periodic Publish:"))
        row2.append(self.periodic_sw)
        box.append(row2)
        
        self.rate_spin = Gtk.SpinButton.new_with_range(1.0, 100.0, 1.0)
        self.rate_spin.set_value(node.max_rate_hz)
        row3 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row3.append(Gtk.Label(label="Max Rate (Hz):"))
        row3.append(self.rate_spin)
        box.append(row3)
        
        save_btn = Gtk.Button(label="Apply")
        save_btn.connect('clicked', self.on_apply)
        box.append(save_btn)
        
    def on_apply(self, btn):
        self.node.throttling_enabled = self.throttle_sw.get_active()
        self.node.periodic_publish = self.periodic_sw.get_active()
        self.node.max_rate_hz = self.rate_spin.get_value()
        self.destroy()

# --- Main Entry ---

def main(args=None):
    # CSS for monitor indicators
    css = """
        .monitor-button-off { background-color: #333; border: 1px solid #555; border-radius: 3px; }
        .monitor-button-on { background-color: #fde725; border: 1px solid #fff; border-radius: 3px; }
    """
    
    rclpy.init(args=args)
    node = OrchestratorNode()
    
    app = OrchestratorUI(node)
    
    # Run ROS in a separate thread
    def ros_spin():
        rclpy.spin(node)
        
    thread = threading.Thread(target=ros_spin, daemon=True)
    thread.start()
    
    try:
        app.run(None)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
