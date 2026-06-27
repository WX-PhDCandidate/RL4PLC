# Robot Choice

The baseline manipulator is **Franka Emika Panda**, loaded from Isaac Sim's built-in USD asset.

Rationale:

- Isaac Lab lists Franka lift-cube manipulation environments, including joint-position and IK variants such as `Isaac-Lift-Cube-Franka-v0`, `Isaac-Lift-Cube-Franka-IK-Abs-v0`, and `Isaac-Lift-Cube-Franka-IK-Rel-v0`.
- Isaac Sim includes an official Franka Pick and Place example for setting up the Franka robot with gripper, sequencing pick/place actions, and controlling the gripper.
- Isaac Sim policy examples include a Franka Panda policy trained in Isaac Lab.

Primary references:

- Isaac Lab environments: https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html
- Isaac Sim Franka Pick and Place: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/examples/manipulation_franka_pick_place.html
- Isaac Sim RL policy examples: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/ext_isaacsim_robot_policy_example.html

This project does not rebuild the robot model. It first tries to use Isaac Sim's official Franka manipulator class, then falls back to the packaged Franka USD as a visual reference if the class is unavailable. The next step is to replace waypoint playback with IK, then with an Isaac Lab RL policy.
