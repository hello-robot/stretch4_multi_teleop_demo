#!/usr/bin/env python3
"""Push-to-talk voice command node: transcribes speech (faster-whisper) into Joy axes/buttons."""
import sys
import os
import threading
import time
import yaml
import numpy as np
import io
import re
from pynput import keyboard

import rclpy
from multi_teleop.base import InputInterfaceNode
import sounddevice as sd

class VoiceNode(InputInterfaceNode):
    """Records mic audio while SPACE is held, transcribes it, and maps phrases to Joy state."""

    def __init__(self, config_path=None):
        """Load voice_commands config (defines axis/button names), then load the Whisper model.

        Args:
            config_path: optional YAML path with a 'voice_commands' list.
        """
        # We need to load the config BEFORE calling super().__init__ 
        # to know our axis/button names.
        self._commands = []
        self._axis_map = {}   # axis_name -> index
        self._button_map = {} # keyword -> index
        
        # We'll use the config_path passed from argparse if available,
        # otherwise we'll try to get it from parameters later.
        # But we need it NOW for axis/button names.

        axis_names = []
        button_names = []
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    raw_commands = (config or {}).get('voice_commands', [])
                    for cmd in raw_commands:
                        if not self._is_valid_command(cmd):
                            continue
                        self._commands.append(cmd)
                        if 'axis' in cmd:
                            name = cmd['axis'].lower()
                            if name not in self._axis_map:
                                self._axis_map[name] = len(axis_names)
                                axis_names.append(name)
                        elif 'keyword' in cmd:
                            name = cmd['keyword'].lower()
                            if name not in self._button_map:
                                self._button_map[name] = len(button_names)
                                button_names.append(name)
            except Exception as e:
                print(f"Failed to load voice config: {e}")

        # Fallback if config is empty
        if not axis_names and not button_names:
            axis_names = ['dummy_axis']
            button_names = ['dummy_button']

        super().__init__('voice_node', axis_names, button_names)
        
        self.declare_parameter('config_file', config_path or '')
        self.declare_parameter('model_size', 'base.en')
        
        # Audio settings
        self.CHANNELS = 1
        self.RATE = 16000
        self._frames = []
        self._is_recording = False
        
        # Whisper model
        from faster_whisper import WhisperModel
        model_size = self.get_parameter('model_size').value
        self.get_logger().info(f"Loading Whisper model: {model_size}...")
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.get_logger().info("Whisper model loaded.")
        
        # Internal state (numeric values, NOT the InputInterfaceNode axis/button
        # name lists - keep these separate from self._axes/self._buttons so
        # control_axes/control_buttons keep returning the configured names).
        self._axis_state = [0.0] * len(axis_names)
        self._button_state = [0] * len(button_names)

        # PTT Listener
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

        # Periodic republish so a voice-set axis/button value doesn't go stale
        # for downstream consumers (e.g. control schemes with a Joy watchdog)
        # between spoken commands. 10Hz is comfortably faster than any
        # downstream staleness timeout while being appropriate for a discrete,
        # infrequent input source (unlike the continuous-device nodes, which
        # poll much faster).
        self._publish_timer = self.create_timer(0.1, self.update_and_publish)

        self.get_logger().info(f"Voice Tracker initialized with {len(axis_names)} axes and {len(button_names)} buttons. Hold SPACE to talk.")

    def _is_valid_command(self, cmd):
        """Validate one 'voice_commands' config entry; warn and skip if malformed.

        A step-action axis entry needs 'up'/'down'; a scale-action axis entry
        needs 'min'/'max'. Without this check, a malformed entry would raise
        an uncaught KeyError later inside _parse_commands.

        Args:
            cmd: one raw dict from the 'voice_commands' config list.
        Returns:
            True if the entry is well-formed and safe to use.
        """
        if not isinstance(cmd, dict):
            print(f"WARNING: skipping malformed voice command {cmd!r}: not a mapping")
            return False

        if 'axis' in cmd:
            action = cmd.get('action', 'step')
            if action == 'step' and ('up' not in cmd or 'down' not in cmd):
                print(f"WARNING: skipping malformed voice command {cmd!r}: "
                      f"'step' action requires 'up' and 'down'")
                return False
            if action == 'scale' and ('min' not in cmd or 'max' not in cmd):
                print(f"WARNING: skipping malformed voice command {cmd!r}: "
                      f"'scale' action requires 'min' and 'max'")
                return False
        elif 'keyword' not in cmd:
            print(f"WARNING: skipping malformed voice command {cmd!r}: "
                  f"missing both 'axis' and 'keyword'")
            return False

        return True

    def _on_press(self, key):
        """Pynput callback: start recording on SPACE key-down."""
        if key == keyboard.Key.space and not self._is_recording:
            self._is_recording = True
            self._frames = []
            threading.Thread(target=self._record_thread, daemon=True).start()
            self.get_logger().info("Recording...")

    def _on_release(self, key):
        """Pynput callback: stop recording on SPACE key-up."""
        if key == keyboard.Key.space and self._is_recording:
            self._is_recording = False
            self.get_logger().info("Stopped recording. Transcribing...")

    def _record_thread(self):
        """Background thread: capture mic audio via sounddevice until _is_recording clears."""
        def callback(indata, frames, time, status):
            if self._is_recording:
                self._frames.append(indata.copy())

        with sd.InputStream(samplerate=self.RATE, channels=self.CHANNELS, callback=callback):
            while self._is_recording:
                time.sleep(0.05)
        
        self._process_audio()

    def _process_audio(self):
        """Concatenate recorded frames, run Whisper transcription, and parse commands."""
        if not self._frames:
            return
        audio_np = np.concatenate(self._frames, axis=0).flatten()
        segments, info = self._model.transcribe(audio_np, beam_size=5)
        text = " ".join([s.text for s in segments]).lower().strip()
        self.get_logger().info(f"Transcribed: '{text}'")
        if text:
            self._parse_commands(text)

    def _parse_commands(self, text):
        """Match transcribed text against configured keyword/axis commands and publish.

        Args:
            text: lowercased transcription to search for keywords/step/scale phrases.
        """
        updated = False
        for cmd in self._commands:
            # Handle Buttons
            if 'keyword' in cmd:
                kw = cmd['keyword'].lower()
                if kw in text:
                    action = cmd.get('action', 'tap')
                    idx = self._button_map[kw]
                    if action == 'toggle':
                        self._button_state[idx] = 1 if self._button_state[idx] == 0 else 0
                        updated = True
                    elif action == 'tap':
                        self._button_state[idx] = 1
                        self.publish_input(self._axis_state, self._button_state)
                        time.sleep(0.1)
                        self._button_state[idx] = 0
                        updated = True

            # Handle Axes
            elif 'axis' in cmd:
                name = cmd['axis'].lower()
                idx = self._axis_map[name]
                action = cmd.get('action', 'step')

                if action == 'step':
                    up_kw = f"{name} {cmd['up'].lower()}"
                    down_kw = f"{name} {cmd['down'].lower()}"
                    step = cmd.get('step', 0.1)

                    if up_kw in text:
                        self._axis_state[idx] = min(1.0, self._axis_state[idx] + step)
                        updated = True
                    elif down_kw in text:
                        self._axis_state[idx] = max(-1.0, self._axis_state[idx] - step)
                        updated = True

                elif action == 'scale':
                    # Look for [axis name] [number]
                    match = re.search(rf"{re.escape(name)}\s+(.*)", text)
                    if match:
                        after_text = match.group(1).strip()
                        parts = after_text.split()
                        if not parts: continue

                        multiplier = 1.0
                        num_part = parts[0]
                        if parts[0] in ['negative', 'minus', '-'] and len(parts) > 1:
                            multiplier = -1.0
                            num_part = parts[1]
                        elif parts[0].startswith('-'):
                            num_part = parts[0]

                        val = self._word_to_num(num_part)
                        if val is not None:
                            v_min = cmd['min']
                            v_max = cmd['max']
                            real_val = val * multiplier
                            norm_val = 2 * (real_val - v_min) / (v_max - v_min) - 1
                            self._axis_state[idx] = max(-1.0, min(1.0, norm_val))
                            updated = True

        if updated:
            self.publish_input(self._axis_state, self._button_state)

    def update_and_publish(self):
        """Timer callback: publish the current voice-derived axis/button state as Joy.

        _parse_commands already publishes immediately on every state change;
        this periodic republish exists so a value set by a voice command
        (e.g. an axis nudged via a "step" command) keeps being republished
        between commands, since some downstream consumers treat a stale Joy
        topic as "no input" after a timeout.
        """
        self.publish_input(self._axis_state, self._button_state)

    def _word_to_num(self, word):
        """Parse a spoken number word or numeral string into a float, or None."""
        # Clean the word
        word = word.strip().strip(',.?!')
        
        # Basic mapping for spoken numbers
        num_map = {
            'zero': 0.0, 'one': 1.0, 'two': 2.0, 'three': 3.0, 'four': 4.0,
            'five': 5.0, 'six': 6.0, 'seven': 7.0, 'eight': 8.0, 'nine': 9.0, 'ten': 10.0,
            'point': 0.5, # Special case for "point five" if transcribed oddly
        }
        
        try:
            return float(word)
        except ValueError:
            return num_map.get(word.lower())

def main(args=None):
    """Entry point: parse --config, construct VoiceNode, and spin until interrupted."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to config file')
    parsed_args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = VoiceNode(config_path=parsed_args.config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
