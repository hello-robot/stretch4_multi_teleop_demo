# stretch4_multi_teleop

Multi-input teleoperation framework that routes signals from arbitrary input trackers to any ROS 2 node listening to a `sensor_msgs/msg/Joy` message.

---

## Quick Navigation

*   **[Multi-Teleoperation Agent Guide](./AGENT_GUIDE.md)**: Conceptual architecture, dynamic signal-routing pipelines, virtual signal structures, and code recipes for creating custom input interfaces and control schemes.
*   **[Stretch Drivers Agent Guide](../stretch4_ros2/DRIVERS_GUIDE.md)**: Concrete and abstract class hierarchies, safe operation manuals (homing, stowing, runstop), and runtime joint-mode parameter switching rules.

---

## Directory Structure

*   `multi_teleop/`: Contains the base base classes (`InputInterfaceNode`, `ControlSchemeNode`) and the central `OrchestratorNode`.
*   `control_schemes/`: Concrete control configurations (e.g., Kinematic IK, standard position/velocity control schemes).
*   `teleop_interfaces/`: Concrete input source nodes (such as Gamepads, SpaceMouse, Voice, and CV/MediaPipe tracking cameras).

## Quick Start (Simulation)

To launch the multi-teleop framework with the Stretch simulation, refer to the step-by-step setup in the **[AGENT_GUIDE.md](./AGENT_GUIDE.md#5-launch-and-setup)**.
