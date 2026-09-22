#!/usr/bin/env python3
"""GTK4/Adwaita debug GUI that discovers and visualizes all live sensor_msgs/Joy topics."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import gi
import threading
import time

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Gio, GLib, Adw

class InterfaceMonitor(Adw.Application):
    """Scans the ROS graph for Joy topics and renders a live axes/buttons panel each."""

    def __init__(self, node):
        """Store the plain rclpy Node used for topic discovery and subscriptions."""
        super().__init__(application_id='com.antigravity.interface_monitor',
                         flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.node = node
        self.topic_data = {}  # topic_name -> {box, sub, axes_widgets, button_widgets, axis_labels, button_labels}

    def do_activate(self):
        """GTK activation hook: build the window and trigger an initial topic scan."""
        self.win = Adw.ApplicationWindow(application=self)
        self.win.set_title("Antigravity Interface Monitor")
        self.win.set_default_size(800, 600)

        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.win.set_content(self.main_box)

        # Header bar
        header = Adw.HeaderBar()
        self.main_box.append(header)

        # Scan button
        scan_btn = Gtk.Button(label="RE-SCAN ENVIRONMENT")
        scan_btn.connect('clicked', self.on_rescan_clicked)
        header.pack_start(scan_btn)

        # Scrollable area
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        self.main_box.append(scrolled)

        self.topics_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.topics_container.set_margin_top(10)
        self.topics_container.set_margin_bottom(10)
        self.topics_container.set_margin_start(10)
        self.topics_container.set_margin_end(10)
        scrolled.set_child(self.topics_container)

        self.win.present()
        
        # Initial scan
        GLib.idle_add(self.rescan_environment)

    def on_rescan_clicked(self, button):
        """Button callback: re-run the topic scan."""
        self.rescan_environment()

    def rescan_environment(self):
        """Find all sensor_msgs/Joy topics and add/remove monitor cards to match."""
        self.node.get_logger().info("Scanning for Joy topics...")
        topic_names_and_types = self.node.get_topic_names_and_types()
        
        joy_topics = [t[0] for t in topic_names_and_types if 'sensor_msgs/msg/Joy' in t[1]]
        
        # Remove old topics
        current_topics = list(self.topic_data.keys())
        for topic in current_topics:
            if topic not in joy_topics:
                self.node.get_logger().info(f"Removing topic: {topic}")
                self.topics_container.remove(self.topic_data[topic]['card'])
                self.node.destroy_subscription(self.topic_data[topic]['sub'])
                del self.topic_data[topic]

        # Add new topics
        for topic in joy_topics:
            if topic not in self.topic_data:
                self._add_topic_monitor(topic)

    def _add_topic_monitor(self, topic):
        """Create a card + subscription for a newly discovered Joy topic."""
        self.node.get_logger().info(f"Adding monitor for: {topic}")
        
        card = Gtk.Frame()
        card.set_label(f"TOPIC: {topic}")
        card.set_margin_bottom(10)
        
        content_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        content_box.set_margin_top(10)
        content_box.set_margin_bottom(10)
        content_box.set_margin_start(10)
        content_box.set_margin_end(10)
        card.set_child(content_box)
        
        axes_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        axes_label = Gtk.Label(label="AXES")
        axes_label.set_halign(Gtk.Align.START)
        axes_box.append(axes_label)
        
        btns_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        btns_label = Gtk.Label(label="BUTTONS")
        btns_label.set_halign(Gtk.Align.START)
        btns_box.append(btns_label)
        
        content_box.append(axes_box)
        content_box.append(btns_box)
        
        self.topics_container.append(card)
        
        self.topic_data[topic] = {
            'card': card,
            'axes_box': axes_box,
            'btns_box': btns_box,
            'axis_widgets': [],
            'button_widgets': [],
            'axis_labels': [],
            'button_labels': [],
            'initialized': False
        }

        # Find the node publishing this topic
        pubs = self.node.get_publishers_info_by_topic(topic)
        if pubs:
            threading.Thread(target=self._fetch_params, args=(topic, pubs[0].node_name), daemon=True).start()

        # Subscribe
        sub = self.node.create_subscription(Joy, topic, lambda msg, t=topic: GLib.idle_add(self._joy_callback, msg, t), 10)
        self.topic_data[topic]['sub'] = sub

    def _fetch_params(self, topic, node_name):
        """Background-thread: fetch axis_names/button_names from the publishing node."""
        from rcl_interfaces.srv import GetParameters
        client = self.node.create_client(GetParameters, f'{node_name}/get_parameters')
        
        if not client.wait_for_service(timeout_sec=2.0):
            return
            
        req = GetParameters.Request()
        req.names = ['axis_names', 'button_names']
        future = client.call_async(req)
        
        # Wait for result (using a simple loop since we are in a thread)
        start_time = time.time()
        while time.time() - start_time < 5.0:
            if future.done():
                try:
                    result = future.result()
                    axis_names = result.values[0].string_array_value
                    button_names = result.values[1].string_array_value
                    GLib.idle_add(self._update_labels, topic, axis_names, button_names)
                except Exception as e:
                    self.node.get_logger().error(f"Failed to fetch params: {e}")
                return
            time.sleep(0.1)

    def _update_labels(self, topic, axis_names, button_names):
        """Apply fetched axis/button names to the card's labels, if it still exists."""
        if topic not in self.topic_data: return
        data = self.topic_data[topic]
        
        for i, name in enumerate(axis_names):
            if i < len(data['axis_labels']):
                data['axis_labels'][i].set_text(name)
        
        for i, name in enumerate(button_names):
            if i < len(data['button_labels']):
                data['button_labels'][i].set_text(name)

    def _joy_callback(self, msg, topic):
        """Subscription callback (marshaled via GLib.idle_add): update the topic's widgets."""
        if topic not in self.topic_data:
            return
            
        data = self.topic_data[topic]
        
        if not data['initialized']:
            self._init_widgets(topic, len(msg.axes), len(msg.buttons))
            data['initialized'] = True
            
        # Update Axes
        for i, val in enumerate(msg.axes):
            if i < len(data['axis_widgets']):
                data['axis_widgets'][i].set_value(val)
                
        # Update Buttons
        for i, val in enumerate(msg.buttons):
            if i < len(data['button_widgets']):
                if val > 0:
                    data['button_widgets'][i].add_css_class('suggested-action')
                else:
                    data['button_widgets'][i].remove_css_class('suggested-action')

    def _init_widgets(self, topic, num_axes, num_buttons):
        """Lazily build the axis sliders and button indicators for a topic's card."""
        data = self.topic_data[topic]
        
        # CSS for buttons
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data("""
            .joy-button {
                min-width: 20px;
                min-height: 20px;
                border-radius: 5px;
                background-color: #3b528b;
                margin: 2px;
            }
            .joy-button.suggested-action {
                background-color: #fde725;
            }
        """, -1)
        Gtk.StyleContext.add_provider_for_display(Gtk.Widget.get_display(data['card']), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        for i in range(num_axes):
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            lbl = Gtk.Label(label=f"Axis {i}")
            lbl.set_size_request(100, -1)
            lbl.set_halign(Gtk.Align.START)
            row.append(lbl)
            data['axis_labels'].append(lbl)
            
            adj = Gtk.Adjustment(value=0, lower=-1.0, upper=1.0, step_increment=0.01)
            scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj)
            scale.set_hexpand(True)
            scale.set_sensitive(False) # Read-only
            row.append(scale)
            data['axis_widgets'].append(adj)
            data['axes_box'].append(row)
            
        grid = Gtk.FlowBox()
        grid.set_max_children_per_line(4)
        grid.set_selection_mode(Gtk.SelectionMode.NONE)
        data['btns_box'].append(grid)
        
        for i in range(num_buttons):
            btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            indicator = Gtk.Box()
            indicator.add_css_class('joy-button')
            btn_box.append(indicator)
            data['button_widgets'].append(indicator)
            
            lbl = Gtk.Label(label=str(i))
            lbl.set_halign(Gtk.Align.START)
            btn_box.append(lbl)
            data['button_labels'].append(lbl)
            
            grid.append(btn_box)

def main(args=None):
    """Entry point: spin a plain Node on a thread and run the InterfaceMonitor GTK app."""
    rclpy.init(args=args)
    node = Node('interface_monitor_node')
    
    app = InterfaceMonitor(node)
    
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
