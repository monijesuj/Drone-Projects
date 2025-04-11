import os
import time
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

# Formation and Impedance Parameters
DESIRED_DISTANCE = 0.11       # Nominal inter-drone spacing
K_IMP = 0.6                  # Spring stiffness for virtual impedance
D_IMP = 0.3                  # Damping coefficient for virtual impedance
IMPEDANCE_FORCE_SCALE = 0.8  # Scaling factor for impedance force contribution

# Obstacle Avoidance Parameters (applied only to the leader’s nominal target)
K_OBS = 1.0         # Repulsive stiffness for obstacle interaction
THRESHOLD = 0.5     # Distance threshold (meters)
OBSTACLE_FORCE_SCALE = 1.0  # Scaling factor for obstacle repulsive force

def compute_virtual_forces(positions, velocities, k, d, desired_dist):
    """
    Compute pairwise virtual impedance forces among drones.
    Each pair of drones contributes a spring-damper force if they are within 2x the desired distance.
    
    Args:
        positions: Array of drone positions (num_drones, 3)
        velocities: Array of drone velocities (num_drones, 3)
        k: Spring stiffness coefficient.
        d: Damping coefficient.
        desired_dist: Desired distance between drones.
        
    Returns:
        forces: Array of total virtual impedance forces for each drone (num_drones, 3)
    """
    num_drones = len(positions)
    forces = np.zeros((num_drones, 3))
    for i in range(num_drones):
        for j in range(num_drones):
            if i != j:
                dx = positions[i] - positions[j]
                dv = velocities[i] - velocities[j]
                distance = np.linalg.norm(dx)
                # Only apply the impedance link if within a reasonable range
                if distance < 2 * desired_dist:
                    direction = dx / (distance + 1e-6)
                    f_spring = -k * (distance - desired_dist) * direction
                    f_damper = -d * dv
                    forces[i] += f_spring + f_damper
    return forces

def compute_obstacle_force(position, obstacle_positions, k_obs, threshold):
    """
    Compute the repulsive force on a single drone due to obstacles.
    
    Args:
        position: Drone position (3,)
        obstacle_positions: Array of obstacle positions (num_obstacles, 3)
        k_obs: Obstacle repulsive stiffness.
        threshold: Distance threshold for repulsion.
        
    Returns:
        force: The repulsive force (3,)
    """
    force = np.zeros(3)
    for obs_pos in obstacle_positions:
        dx = position - obs_pos
        distance = np.linalg.norm(dx)
        if distance < threshold:
            direction = dx / (distance + 1e-6)
            force += k_obs * (threshold - distance) * direction
    return force

def get_formation_offset(follower_index, spacing):
    """
    Compute the formation offset (in the leader’s body frame) for a follower drone.
    Uses a V-shaped (triangular) pattern: odd-indexed drones to the left, even-indexed to the right.
    
    Args:
        follower_index: integer (starting at 1) for follower drone.
        spacing: Nominal spacing.
        
    Returns:
        offset: numpy array [dx, dy, 0] where dx is behind the leader and dy is lateral.
    """
    row = (follower_index + 1) // 2
    if follower_index % 2 == 1:
        offset = np.array([-row * spacing, row * spacing, 0])
    else:
        offset = np.array([-row * spacing, -row * spacing, 0])
    return offset

def run(
    drone=DroneModel.CF2X,
    num_drones=10,
    physics=Physics.PYB,
    gui=True,
    record_video=False,
    plot=True,
    output_folder='results'
):
    # Leader’s circular trajectory parameters.
    R = 1.0
    CONTROL_FREQ_HZ = 48
    PERIOD = 10
    NUM_WP = CONTROL_FREQ_HZ * PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        TARGET_POS[i] = [R * np.cos((i / NUM_WP) * 2 * np.pi),
                         R * np.sin((i / NUM_WP) * 2 * np.pi),
                         1.0]

    # Set leader’s initial position and heading.
    leader_init_pos = TARGET_POS[0]
    theta = np.arctan2(leader_init_pos[1], leader_init_pos[0])
    forward = np.array([-np.sin(theta), np.cos(theta)])
    left = np.array([-np.cos(theta), -np.sin(theta)])
    
    # Initialize positions: leader at its starting waypoint, followers at formation offsets relative to leader.
    INIT_XYZS = np.zeros((num_drones, 3))
    INIT_XYZS[0] = leader_init_pos
    for j in range(1, num_drones):
        offset = get_formation_offset(j, DESIRED_DISTANCE)
        formation_offset_xy = offset[0] * forward + offset[1] * left
        INIT_XYZS[j] = leader_init_pos + np.array([formation_offset_xy[0], formation_offset_xy[1], 0])
    
    INIT_RPYS = np.zeros((num_drones, 3))
    
    # Setup simulation environment.
    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        gui=gui,
        record=record_video,
        pyb_freq=240,
        ctrl_freq=CONTROL_FREQ_HZ
    )
    
    # Add obstacles (static boxes).
    obstacle_positions = np.array([
        [0.4, 0.4, 0.8],
        [0.8, 0.8, 1.2],
        [-0.8, -0.8, 0.8]
    ])
    for pos in obstacle_positions:
        shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.13, 0.13, 0.13],
            physicsClientId=env.CLIENT
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            basePosition=pos,
            physicsClientId=env.CLIENT
        )
    
    # Initialize controllers and logger.
    ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]
    logger = Logger(logging_freq_hz=CONTROL_FREQ_HZ, 
                    num_drones=num_drones,
                    output_folder=output_folder)
    
    action = np.zeros((num_drones, 4))
    START = time.time()
    iteration = 0
    leader_wp_counter = 0

    try:
        while True:
            obs, _, _, _, _ = env.step(action)
            positions = np.array([ob[0:3] for ob in obs])
            velocities = np.array([ob[10:13] for ob in obs])
            
            # Compute the virtual impedance forces between drones.
            virtual_forces = compute_virtual_forces(positions, velocities, K_IMP, D_IMP, DESIRED_DISTANCE)
            
            # Leader’s nominal target: follow its circular trajectory and avoid obstacles.
            leader_obstacle_force = compute_obstacle_force(positions[0], obstacle_positions, K_OBS, THRESHOLD)
            leader_nominal_target = TARGET_POS[leader_wp_counter] + OBSTACLE_FORCE_SCALE * leader_obstacle_force
            
            for j in range(num_drones):
                if j == 0:
                    # Leader: its nominal target is directly used.
                    nominal_target = leader_nominal_target
                else:
                    # Followers: compute a nominal target relative to the leader’s nominal target.
                    offset = get_formation_offset(j, DESIRED_DISTANCE)
                    theta = np.arctan2(leader_nominal_target[1], leader_nominal_target[0])
                    forward = np.array([-np.sin(theta), np.cos(theta)])
                    left = np.array([-np.cos(theta), -np.sin(theta)])
                    formation_offset_xy = offset[0] * forward + offset[1] * left
                    nominal_target = leader_nominal_target + np.array([formation_offset_xy[0], formation_offset_xy[1], 0])
                
                # Add the virtual impedance force (scaled) to the nominal target.
                adjusted_target = nominal_target + IMPEDANCE_FORCE_SCALE * virtual_forces[j]
                
                action[j], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=obs[j],
                    target_pos=adjusted_target,
                    target_rpy=INIT_RPYS[j]
                )
                
                logger.log(
                    drone=j,
                    timestamp=iteration / env.CTRL_FREQ,
                    state=obs[j],
                    control=np.hstack([adjusted_target, INIT_RPYS[j], np.zeros(6)])
                )
            
            leader_wp_counter = (leader_wp_counter + 1) % NUM_WP
            env.render()
            if gui:
                sync(iteration, START, env.CTRL_TIMESTEP)
            iteration += 1

    except KeyboardInterrupt:
        print("Simulation terminated by user.")
    
    env.close()
    logger.save()
    if plot:
        logger.plot()

if __name__ == "__main__":
    run()
