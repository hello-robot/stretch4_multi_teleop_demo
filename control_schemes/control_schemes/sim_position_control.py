#!/usr/bin/env python3

import rclpy
from multi_teleop.base import ControlSchemeNode
from sensor_msgs.msg import Joy
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.utils import H0_from_driving_dir
import stretch4_mujoco.config as config
import numpy as np
import threading
import time

class DirectPositionControlNode(ControlSchemeNode):
    def __init__(self):
        # Head Pan and Head Tilt are not available in this Stretch 4 MJCF model
        self.axis_names = [
            "Lift", "Arm", "Wrist Yaw", "Wrist Pitch", "Wrist Roll", "Gripper"
        ]
        self.button_names = [
            "Forward", "Backward", "Left", "Right", "rotate_cw", "rotate_ccw"
        ]
        super().__init__("direct_position_control", self.axis_names, self.button_names)
        
        # Joint ranges
        self.ranges = {
            "lift": (0.0, 1.2),
            "arm": (0.0, 0.52),
            "wrist_yaw": (-1.135, 4.276),
            "wrist_pitch": (-1.135, 4.276),
            "wrist_roll": (-4.276, 1.135),
            "gripper": (0.0, 0.5)
        }
        
        # Map axes to actuators
        self.axis_map = [
            "lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "gripper"
        ]
        
        # Desired state
        self.target_positions = {}
        self.target_v_x = 0.0
        self.target_v_y = 0.0
        self.target_omega = 0.0
        self.last_sent_v_x = 0.0
        self.last_sent_v_y = 0.0
        self.last_sent_omega = 0.0
        self.last_joy_time = 0.0
        self.prev_buttons = [0] * len(self.button_names)
        
        # Initialize simulator
        self.sim = Stretch4MujocoSimulator()
        
        # Timer for commanding the robot at a fixed rate (20Hz)
        self.cmd_timer = self.create_timer(0.05, self.command_loop)
        
        self.get_logger().info("Direct Position Control Node Initialized with 20Hz command loop.")

    def handle_joy(self, axes: list, buttons: list):
        # Safety check for message size
        if len(axes) != len(self.axis_names) or len(buttons) != len(self.button_names):
            self.get_logger().warning(
                f"Received Joy message with unexpected size. "
                f"Expected {len(self.axis_names)} axes and {len(self.button_names)} buttons, "
                f"got {len(axes)} and {len(buttons)}. Ignoring."
            )
            return

        self.last_joy_time = time.time()
        vel_scale = 1.0 # m/s (increased from 0.5)
        omega_scale = 1.5 # rad/s
        
        # Handle non-wheel joints (axes)
        for i, val in enumerate(axes):
            if i < len(self.axis_map):
                actuator_name = self.axis_map[i]
                low, high = self.ranges[actuator_name]
                # Remap -1, 1 to low, high
                pos = (val + 1.0) / 2.0 * (high - low) + low
                self.target_positions[actuator_name] = pos
                
        # Handle directional movement (buttons)
        # buttons: 0:Forward, 1:Backward, 2:Left, 3:Right, 4:rotate_cw, 5:rotate_ccw
        
        # Check for new presses to prioritize newest direction
        # X Axis (Forward/Backward)
        if buttons[0] and not self.prev_buttons[0]:
            self.target_v_x = vel_scale
        elif buttons[1] and not self.prev_buttons[1]:
            self.target_v_x = -vel_scale
        # If no new press, check if current direction was released
        elif self.target_v_x > 0 and not buttons[0]:
            self.target_v_x = -vel_scale if buttons[1] else 0.0
        elif self.target_v_x < 0 and not buttons[1]:
            self.target_v_x = vel_scale if buttons[0] else 0.0
        # Edge case: if a button is held but target_v_x is 0 (e.g. on start), set it
        elif self.target_v_x == 0:
            if buttons[0]: self.target_v_x = vel_scale
            elif buttons[1]: self.target_v_x = -vel_scale

        # Y Axis (Left/Right)
        if buttons[2] and not self.prev_buttons[2]:
            self.target_v_y = vel_scale
        elif buttons[3] and not self.prev_buttons[3]:
            self.target_v_y = -vel_scale
        elif self.target_v_y > 0 and not buttons[2]:
            self.target_v_y = -vel_scale if buttons[3] else 0.0
        elif self.target_v_y < 0 and not buttons[3]:
            self.target_v_y = vel_scale if buttons[2] else 0.0
        elif self.target_v_y == 0:
            if buttons[2]: self.target_v_y = vel_scale
            elif buttons[3]: self.target_v_y = -vel_scale

        # Rotation (CW/CCW)
        if buttons[5] and not self.prev_buttons[5]:
            self.target_omega = omega_scale
        elif buttons[4] and not self.prev_buttons[4]:
            self.target_omega = -omega_scale
        elif self.target_omega > 0 and not buttons[5]:
            self.target_omega = -omega_scale if buttons[4] else 0.0
        elif self.target_omega < 0 and not buttons[4]:
            self.target_omega = omega_scale if buttons[5] else 0.0
        elif self.target_omega == 0:
            if buttons[5]: self.target_omega = omega_scale
            elif buttons[4]: self.target_omega = -omega_scale

        self.prev_buttons = list(buttons)

    def command_loop(self):
        if not self.sim.is_running():
            return

        # Watchdog: Stop if no joy message for 0.5s
        if time.time() - self.last_joy_time > 0.5:
            self.target_v_x = 0.0
            self.target_v_y = 0.0
            self.target_omega = 0.0

        # Send joint commands
        for actuator_name, pos in self.target_positions.items():
            if actuator_name == "gripper":
                self.sim.move_to("gripper_left_finger", pos)
                self.sim.move_to("gripper_right_finger", pos)
            else:
                self.sim.move_to(actuator_name, pos)
        
        # Only send base velocity command if it has changed
        if (self.target_v_x != self.last_sent_v_x or 
            self.target_v_y != self.last_sent_v_y or 
            self.target_omega != self.last_sent_omega):
            if isinstance(self.sim, Stretch4MujocoSimulator):
                self.sim.set_base_velocity(v_x=self.target_v_x, v_y=self.target_v_y, omega=self.target_omega)
            else:
                self.sim.set_base_velocity(self.target_v_y, self.target_omega)
            self.last_sent_v_x = self.target_v_x
            self.last_sent_v_y = self.target_v_y
            self.last_sent_omega = self.target_omega

def main(args=None):
    rclpy.init(args=args)
    node = DirectPositionControlNode()
    
    # Start ROS spin in a separate thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    try:
        # Start simulator in main thread to allow signal handling
        node.sim.start()
        
        # Keep main thread alive while simulator is running
        while node.sim.is_running():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        node.sim.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
