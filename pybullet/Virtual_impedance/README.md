# PyBullet Drone Swarm Simulation

This project simulates a drone swarm using PyBullet, featuring virtual impedance control for formation maintenance and obstacle avoidance. A leader drone follows a circular trajectory while followers maintain a V-shaped formation.

## Features

- Virtual impedance control to maintain formation
- Obstacle avoidance for the leader drone
- Leader circular trajectory with configurable waypoints and radius
- V-shaped formation for follower drones
- Real-time logging and rendering

## Prerequisites

- Python 3.6+
- [PyBullet](https://pypi.org/project/pybullet/)
- [NumPy](https://pypi.org/project/numpy/)
- [gym_pybullet_drones](https://pypi.org/project/gym-pybullet-drones/)

Install the dependencies:

```bash
pip install pybullet numpy gym-pybullet-drones
```

## Running the Simulation

Navigate to the project folder and run the simulation script:

```bash
cd /home/monijesu/My-Robotics/Drone-Projects/pybullet
python swarm_impedance.py
```

## Code Overview

- **swarm_impedance.py**:  
  - Computes pairwise virtual impedance forces among drones  
  - Implements obstacle avoidance for the leader drone  
  - Sets up the circular trajectory for the leader and a V-shaped formation for followers  
  - Logs simulation data and renders the simulation in real-time

## Configuration Parameters

- **Formation and Impedance Parameters**
  - `DESIRED_DISTANCE`: Nominal spacing between drones
  - `K_IMP`: Spring stiffness for the virtual impedance
  - `D_IMP`: Damping coefficient for the virtual impedance
  - `IMPEDANCE_FORCE_SCALE`: Scaling factor for the impedance force

- **Obstacle Avoidance Parameters**
  - `K_OBS`: Repulsive stiffness for obstacles
  - `THRESHOLD`: Distance threshold for obstacle interaction
  - `OBSTACLE_FORCE_SCALE`: Scaling factor for the obstacle force
