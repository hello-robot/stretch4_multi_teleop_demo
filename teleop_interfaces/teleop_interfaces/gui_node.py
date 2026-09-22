#!/usr/bin/env python3
"""GTK4/Adwaita on-screen teleop GUI: config-driven sliders/buttons published as Joy."""
import rclpy
from multi_teleop.base import InputInterfaceNode
import gi
import yaml
import os
import threading

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Gio, GLib, Adw

class TeleopGuiApp(Adw.Application):
    """GTK application that builds sliders/buttons from GuiNode's config and drives it."""

    def __init__(self, node):
        """Store the GuiNode this app's widgets will read/update."""
        super().__init__(application_id='com.antigravity.teleop_gui',
                         flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.node = node

    def do_activate(self):
        """GTK activation hook: build and present the main window from node config."""
        self.win = Adw.ApplicationWindow(application=self)
        self.win.set_title("Antigravity Teleop Dashboard")
        self.win.set_default_size(500, 400)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.win.set_content(main_box)

        header = Adw.HeaderBar()
        main_box.append(header)

        # Content area
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        main_box.append(scrolled)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=20)
        content.set_margin_top(20)
        content.set_margin_bottom(20)
        content.set_margin_start(20)
        content.set_margin_end(20)
        scrolled.set_child(content)

        # Axes Section
        if self.node._axis_configs:
            axes_group = Adw.PreferencesGroup(title="Axes")
            content.append(axes_group)
            
            for i, cfg in enumerate(self.node._axis_configs):
                row = Adw.ActionRow(title=cfg['name'])
                
                adj = Gtk.Adjustment(value=0.0, lower=cfg['min'], upper=cfg['max'], step_increment=0.01)
                scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj)
                scale.set_hexpand(True)
                scale.set_draw_value(True)
                scale.set_size_request(200, -1)
                
                scale.connect('value-changed', self._on_axis_change, i)
                
                row.add_suffix(scale)
                axes_group.add(row)

        # Buttons Section
        if self.node._button_configs:
            btns_group = Adw.PreferencesGroup(title="Buttons")
            content.append(btns_group)
            
            flowbox = Gtk.FlowBox()
            flowbox.set_valign(Gtk.Align.START)
            flowbox.set_max_children_per_line(3)
            flowbox.set_selection_mode(Gtk.SelectionMode.NONE)
            flowbox.set_column_spacing(10)
            flowbox.set_row_spacing(10)
            btns_group.add(flowbox)
            
            for i, cfg in enumerate(self.node._button_configs):
                is_toggle = cfg.get('toggle', False)
                
                if is_toggle:
                    btn = Gtk.ToggleButton(label=cfg['name'])
                    btn.connect('toggled', self._on_button_toggle, i)
                else:
                    btn = Gtk.Button(label=cfg['name'])
                    
                    # Use a click gesture to handle press/release
                    gesture = Gtk.GestureClick()
                    gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
                    gesture.connect("pressed", self._on_button_press, i)
                    gesture.connect("released", self._on_button_release, i)
                    btn.add_controller(gesture)
                
                btn.set_size_request(120, 40)
                flowbox.append(btn)

        self.win.present()

    def _on_axis_change(self, scale, idx):
        """Slider callback: write the new axis value and publish immediately."""
        self.node._axes_values[idx] = scale.get_value()
        self.node.update_and_publish()

    def _on_button_toggle(self, btn, idx):
        """Toggle-button callback: write the new button state and publish immediately."""
        self.node._buttons_values[idx] = 1 if btn.get_active() else 0
        self.node.update_and_publish()

    def _on_button_press(self, gesture, n_press, x, y, idx):
        """Momentary-button press callback: set button to 1 and publish."""
        self.node._buttons_values[idx] = 1
        self.node.update_and_publish()

    def _on_button_release(self, gesture, n_press, x, y, idx):
        """Momentary-button release callback: set button to 0 and publish."""
        self.node._buttons_values[idx] = 0
        self.node.update_and_publish()

class GuiNode(InputInterfaceNode):
    """InputInterfaceNode driven by TeleopGuiApp's widgets instead of a hardware device."""

    def __init__(self, config_path=None):
        """Load axis/button config (or fall back to one dummy axis/button) and init state."""
        # 1. Load config
        self._axis_configs = []
        self._button_configs = []
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    gui_cfg = config.get('gui_controls', {})
                    self._axis_configs = gui_cfg.get('axes', [])
                    self._button_configs = gui_cfg.get('buttons', [])
            except Exception as e:
                print(f"Failed to load GUI config: {e}")
        
        axis_names = [a['name'] for a in self._axis_configs] or ['dummy_axis']
        button_names = [b['name'] for b in self._button_configs] or ['dummy_button']
        
        super().__init__('gui_node', axis_names, button_names)
        self.declare_parameter('config_file', config_path or '')
        
        self._axes_values = [0.0] * len(axis_names)
        self._buttons_values = [0] * len(button_names)

    def update_and_publish(self):
        """Publish the current slider/button widget state as Joy."""
        self.publish_input(self._axes_values, self._buttons_values)

def main(args=None):
    """Entry point: parse --config, spin GuiNode on a thread, and run the GTK app."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to config file')
    parsed_args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = GuiNode(config_path=parsed_args.config)
    
    app = TeleopGuiApp(node)
    
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
