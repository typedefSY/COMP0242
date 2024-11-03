import numpy as np
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, dyn_cancel
from regulator_model import RegulatorModel

def initialize_simulation(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def print_joint_info(sim, dyn_model, controlled_frame_name):
    """Print initial joint angles and limits."""
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    print(f"Initial joint angles: {init_joint_angles}")
    
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")
    
    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")
    

def getSystemMatrices(sim, num_joints):
    """
    Get the system matrices A and B according to the dimensions of the state and control input.
    
    Parameters:
    sim: Simulation object
    num_joints: Number of robot joints
    damping_coefficients: List or numpy array of damping coefficients for each joint (optional)
    
    Returns:
    A: State transition matrix
    B: Control input matrix
    """
    
    time_step = sim.GetTimeStep()

    I = np.eye(num_joints)
    zero_matrix = np.zeros((num_joints, num_joints))
    
    A = np.block([
        [I, time_step * I],
        [zero_matrix, I]
    ])

    B = np.block([
        [zero_matrix],
        [time_step * I]
    ])
    
    return A, B

def getSystemMatrice_with_damping(sim, num_joints, damping_coefficients):
    """
    Get the system matrices A and B according to the dimensions of the state and control input.
    
    Parameters:
    sim: Simulation object
    num_joints: Number of robot joints
    damping_coefficients: List or numpy array of damping coefficients for each joint (optional)
    
    Returns:
    A: State transition matrix with damping
    B: Control input matrix
    """
    
    time_step = sim.GetTimeStep()

    I = np.eye(num_joints)
    zero_matrix = np.zeros((num_joints, num_joints))
    damping_matrix = np.diag(damping_coefficients)
    
    A = np.block([
        [I, time_step * I],
        [zero_matrix, I - time_step * damping_matrix]
    ])

    B = np.block([
        [zero_matrix],
        [time_step * I]
    ])
    
    return A, B

def getCostMatrices(num_joints):
    """
    Get the cost matrices Q and R for the MPC controller.
    
    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    #Q = 0.01 * np.eye(num_states)  # State cost matrix
    Q = 10000000 * np.eye(num_states)
    Q[num_joints:, num_joints:] = 0.0
   

    R = 0.1 * np.eye(num_controls)  # Control input cost matrix
    
    return Q, R


def main():
    # Configuration
    conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"
    
    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()
    
    # Print joint information
    print_joint_info(sim, dyn_model, controlled_frame_name)
    
    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []

    # Define the matrices\\

    #Add daming into system matrix
    #damping_coefficients = np.array([0.5, 0.6, 0.2, 0.1, 0.3, 0.35, 0.8])
    #A, B = getSystemMatrice_with_damping(sim, num_joints,damping_coefficients)
    A, B = getSystemMatrices(sim, num_joints)

    Q, R = getCostMatrices(num_joints)
    
    np.set_printoptions(linewidth=np.inf)

    np.set_printoptions(linewidth=np.inf)
    print("Martix A is:")
    print(A)

    print("Martix B is:")
    print(B)

    print("Martix Q is:")
    print(Q)

    print("Martix R is:")
    print(R)

    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states)
    # Compute the matrices needed for MPC optimization
    S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
    H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
    
    # Main control loop
    episode_duration = 10
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration/time_step)
    sim.ResetPose()
    # sim.SetSpecificPose([1, 1, 1, 0.4, 0.5, 0.6, 0.7])

    
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        
        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_joints]
       
        # Control command
        tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)
        sim.Step(cmd, "torque")  # Simulation step with torque command

        #########################################################################3print(tau_cmd)
        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        # simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        # print(f"Current time: {current_time}")

    plt.figure(figsize=(10, 2 * num_joints))

    # Loop through each joint to plot position and velocity in one graph
    for i in range(num_joints):
        # Position
        plt.subplot(num_joints, 2, 2 * i + 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity
        plt.subplot(num_joints, 2, 2 * i + 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

    plt.tight_layout()

    plt.show()

#main function with new cost matrix
def main_with_new_cost_matrix():
    # Configuration
    conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"
    
    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()
    
    # Print joint information
    print_joint_info(sim, dyn_model, controlled_frame_name)
    
    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []

    #! Uncomment the following lines to add damping into system matrix
    # damping_coefficients = np.array([0.5, 0.6, 0.2, 0.1, 0.3, 0.35, 0.8])
    # A, B = getSystemMatrices(sim, num_joints,damping_coefficients)
    #! Comment the following line if you want to add damping into system matrix
    A, B = getSystemMatrices(sim, num_joints)

    Q, R = getCostMatrices(num_joints)
    
    np.set_printoptions(linewidth=np.inf)

    np.set_printoptions(linewidth=np.inf)
    print("Martix A is:")
    print(A)

    print("Martix B is:")
    print(B)

    print("Martix Q is:")
    print(Q)

    print("Martix R is:")
    print(R)

    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states)
    # Compute the matrices needed for MPC optimization
    S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
    H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
    
    # Main control loop
    episode_duration = 10
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration/time_step)
    sim.ResetPose()
    # sim.SetSpecificPose([1, 1, 1, 0.4, 0.5, 0.6, 0.7])

    
# testing loop
    for i in range(steps):
        #print(f"Current step: {i + 1} / {steps}")

        # Adjust the speed weighting in the Q matrix over time
        speed_weight = 1.0 + current_time  # Make the speed weighting grow over time

        # Modify Q matrix so that speed-related part grows with time
        Q[num_joints:, num_joints:] = speed_weight * np.eye(num_joints)

        # Recalculate the regulator model with the updated Q
        regulator = RegulatorModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states)
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)

        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        
        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_joints]

        # Control command
        tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)
        sim.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Save measurements
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        # print(f"Current time: {current_time}")


    plt.figure(figsize=(10, 2 * num_joints))

    # Loop through each joint to plot position and velocity in one graph
    for i in range(num_joints):
        # Position
        plt.subplot(num_joints, 2, 2 * i + 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity
        plt.subplot(num_joints, 2, 2 * i + 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

    plt.tight_layout()

    plt.show()
    
    
if __name__ == '__main__':
    #! Comment the line below to run the main function with the new cost matrix
    main()
    #! Uncomment the line below to run the main function with the new cost matrix
    #main_with_new_cost_matrix()
