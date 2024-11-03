import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from simulation_and_control import pb, MotorCommands, PinWrapper
from simulation_and_control import velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
landmarks = np.array([
            [5, 10],
            [15, 5],
            [10, 15],
            [0,0]
        ])


def landmark_range_observations(base_position):
    y = []
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
       
        y.append(range_meas)

    y = np.array(y)
    return y


def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized

    # Convert quaternion to rotation matrix
    rot_quat = quat.toRotationMatrix()

    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Returns [roll, pitch, yaw]

    # Extract the yaw angle
    bearing_ = base_euler[2]

    return bearing_


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints

#! Please make sure to turn off the noise flag in the configuration file before running task 2
def main_task_2():
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    sim,dyn_model,num_joints=init_simulator(conf_file_name)

    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

    #! Plz make sure you turn off the noise flag in config file before running task 2
    base_pos_all, base_bearing_all = [], []

    # initializing MPC
     # Define the matrices
    num_states = 3
    num_controls = 2

    # Horizon length
    N_mpc = 4

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    init_pos  = np.array([2.0, 3.0])
    init_quat = np.array([0,0,0.3827,0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    # Define the cost matrices
    #! Fixed A, B matrices, use Q(144, 134, 180), N=20 and R(0.4), NO P
    #! With dyn updating A, B matrices, use Q(144, 134, 181), N=20 and R(0.4), NO P
    #! With dyn updating A, B matrices, use Q(165, 363, 580), N=4, R(0.2), and P=(300, 300, 300)
    Qcoeff = np.array([165, 363, 580])
    Rcoeff = 0.2
    Pcoeff = [300, 300, 300]
    regulator.setCostMatrices(Qcoeff,Rcoeff,Pcoeff)

    u_mpc = np.zeros(num_controls)

    ##### robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
  
    ##### MPC control action #######
    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)

    predicted_pos_all = []

    while current_time <= 5.2:
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        # Compute the matrices needed for MPC optimization
        cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
        #! Comment following 2 lines if you don't want to update the A and B matrices at each time step 
        cur_u_for_linearization = u_mpc
        regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)

        # S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        #! Uncomment the line above and Comment the following line if you do not want to add terminal cost
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std_terminal_cost()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        x0_mpc = np.hstack((base_pos[:2], base_bearing_))
        x0_mpc = x0_mpc.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls]
        predicted_state = regulator.predict_next_state(np.array(cur_state_x_for_linearization), u_mpc)
        predicted_pos_all.append(predicted_state[:2])
        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)


        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        

        # Store data for plotting if necessary
        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing_)

        # Update current time
        current_time += time_step


    # Plotting 
    # add visualization of final x, y, trajectory and theta
    x_traj = [pos[0] for pos in base_pos_all]  
    y_traj = [pos[1] for pos in base_pos_all]
    x_pred = [pos[0] for pos in predicted_pos_all]
    y_pred = [pos[1] for pos in predicted_pos_all]
    theta_traj = base_bearing_all
    distance_traj = []
    for i in range(len(x_traj)):
        distance = np.sqrt(x_traj[i]**2 + y_traj[i]**2)
        distance_traj.append(distance)

    final_x = x_traj[-1]
    final_y = y_traj[-1]

    fig = plt.figure(figsize=(12, 8))

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, :], aspect='equal')
    ax1.plot(x_traj, y_traj, 'b-', label='Actual trajectory')
    ax1.plot(x_pred, y_pred, 'r--', label='Predicted trajectory')
    ax1.scatter(final_x, final_y, color='red', s=50, label="Final Position")
    ax1.scatter(0, 0, color='green', s=50, label="desired Position")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(distance_traj, label="Distance from origin", color='red')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Distance to original point")
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(theta_traj, label="Theta trajectory", color='green')
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Theta")
    ax3.legend()
    ax3.grid(True)
    
    plt.suptitle("Robot Trajectory, Distance to goal position and bearing of robot")
    
    if not os.path.exists("images/task2"):
        os.makedirs("images/task2")
    #! Uncomment the following line to save the plot
    # plt.savefig("images/task2/updated_A_B_with_P.png")

    plt.tight_layout()
    plt.show()
    
#! Plz make sure you turn on the noise flag in config file before running task3 and task4
def main_task_3_4():
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    sim,dyn_model,num_joints=init_simulator(conf_file_name)

    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

    base_pos_all, base_bearing_all = [], []
    ekf_pos_all = []

    # initializing MPC
     # Define the matrices
    num_states = 3
    num_controls = 2
   
    
    # Measuring all the state
    
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 15

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    init_pos  = np.array([2.0, 3.0])
    init_quat = np.array([0,0,0.3827,0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    # Define the cost matrices
    Qcoeff = np.array([310, 310, 600.0])
    Rcoeff = 0.5
    regulator.setCostMatrices(Qcoeff,Rcoeff)
   
    
    u_mpc = np.zeros(num_controls)

    ##### robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
  
    ##### MPC control action #######
    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)

    # Initialize Kalman filter variables
    state_estimate = np.array([init_pos[0], init_pos[1], init_base_bearing_])  # Initial state estimate
    P = np.eye(num_states) * 0.1  # Initial covariance matrix
    Q_kalman = np.eye(num_states) * 0.01  # Process noise covariance
    W = np.eye(len(landmarks)) * W_range # Measurement noise covariance


    # Initialization non-EKF error for recording
    error_noisy_no_ekf_x = []  
    error_noisy_no_ekf_y = []  
    error_noisy_no_ekf_theta = [] 

    # Initialization EKF error for recording
    error_noisy_with_ekf_x = []  
    error_noisy_with_ekf_y = []  
    error_noisy_with_ekf_theta = []

    # Initialization of the Steady time, Settling time, and Overshoot
    current_time = 0
    threshold = 0.15 # Steady state error threshold
    steady_state_window = 10 # The length of the time window for calculating the steady-state error
    steady_count_threshold = 5  # Number of consecutive steps to meet steady-state condition
    steady_count = 0
    overshoot_x, overshoot_y, overshoot_theta = 0, 0, 0
    steady_state_reached = False
    stabilization_time = None

    time_steps = [] 
    
    while current_time <= 5.2:
        # True state propagation (with process noise)
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        # Kalman filter prediction
        A, B = regulator.A, regulator.B  # State transition matrix and control matrix from the system model
        state_estimate = A @ state_estimate + B @ u_mpc  # Predicted state without measurement update
        P = A @ P @ A.T + Q_kalman  # Update estimated covariance matrix

        # measurements of the robot without noise (just for comparison purpose) #############
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
        #To record real state
        real_state = np.array([base_pos_no_noise[0], base_pos_no_noise[1], base_bearing_no_noise_])

        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        #To record noise state (without EKF)
        noisy_state = np.array([base_pos[0], base_pos[1], base_bearing_])


        # Recording errors with noise without using EKF
        error_noisy_no_ekf_x.append(real_state[0] - noisy_state[0])
        error_noisy_no_ekf_y.append(real_state[1] - noisy_state[1])
        error_noisy_no_ekf_theta.append(real_state[2] - noisy_state[2])
    
        # Get the current state estimate
        y = landmark_range_observations(base_pos)
        y_pred = []
        C = []
        x_pred = state_estimate
        for lm in landmarks:

            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            y_pred.append(range_pred)

            # Jacobian of the measurement model
            C_range = np.array([
                -(dx_pred) / range_pred,
                -(dy_pred) / range_pred,
                0
            ])
            C.append(C_range)
        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        nu = y - y_pred  # Measurement residual
        S = C @ P @ C.T + W  # Residual covariance
        K = P @ C.T @ np.linalg.inv(S)  # Kalman gain
        state_estimate = state_estimate + K @ nu  # Update the state estimate with the measurement
        P = (np.eye(num_states) - K @ C) @ P  # Updated covariance matrix

        # Normalize the angle to be within [-pi, pi]
        state_estimate[-1] = np.arctan2(np.sin(state_estimate[-1]), np.cos(state_estimate[-1]))

        ekf_pos_all.append(state_estimate.copy())
        # Update the matrices needed for MPC optimization
        cur_state_x_for_linearization = state_estimate
        cur_u_for_linearization = u_mpc
        regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)


        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        x0_mpc = state_estimate.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls] 
        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)

        # To record noise state (With EKF)
        ekf_pos_all.append(state_estimate.copy()) 

        # Recording errors with noise using EKF
        # current error
        error_x = real_state[0] - state_estimate[0]
        error_y = real_state[1] - state_estimate[1]
        error_theta = real_state[2] - state_estimate[2]

        # Cumulative error list
        error_noisy_with_ekf_x.append(real_state[0] - state_estimate[0])
        error_noisy_with_ekf_y.append(real_state[1] - state_estimate[1])
        error_noisy_with_ekf_theta.append(real_state[2] - state_estimate[2])

        # If system cannot stable, only calculated the overshoot before the system reaches steady state
        if not steady_state_reached:
            overshoot_x = max(overshoot_x, abs(error_x))
            overshoot_y = max(overshoot_y, abs(error_y))
            overshoot_theta = max(overshoot_theta, abs(error_theta))

        # Detect whether the system has reached steady state
        if not steady_state_reached:
            recent_errors_x = error_noisy_with_ekf_x[-steady_state_window:]
            recent_errors_y = error_noisy_with_ekf_y[-steady_state_window:]
            recent_errors_theta = error_noisy_with_ekf_theta[-steady_state_window:]

            avg_error_x = np.mean(np.abs(recent_errors_x))
            avg_error_y = np.mean(np.abs(recent_errors_y))
            avg_error_theta = np.mean(np.abs(recent_errors_theta))

            #Determine whether the error is below the threshold
            if avg_error_x < threshold and avg_error_y < threshold and avg_error_theta < threshold:
                steady_count += 1
                if steady_count >= steady_count_threshold:
                    steady_state_reached = True
                    stabilization_time = current_time
            else:
                steady_count = 0  # Reset count if condition not met

        # Recording time step
        time_steps.append(current_time)


        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        

        # Store data for plotting if necessary
        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing_)

        # Update current time
        current_time += time_step

    #! Task 3 Results
    plt.figure(figsize=(18, 10))

    # X range error comparison
    plt.subplot(2, 3, 1)
    plt.plot(time_steps, error_noisy_no_ekf_x, label='Error with Noise, No EKF', color='orange')
    plt.plot(time_steps, error_noisy_with_ekf_x, label='Error with Noise, with EKF', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('X Error [m]')
    plt.title('X Position Error Comparison')
    plt.legend()
    plt.grid(True)

    # Y range error comparison
    plt.subplot(2, 3, 2)
    plt.plot(time_steps, error_noisy_no_ekf_y, label='Error with Noise, No EKF', color='orange')
    plt.plot(time_steps, error_noisy_with_ekf_y, label='Error with Noise, with EKF', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Y Error [m]')
    plt.title('Y Position Error Comparison')
    plt.legend()
    plt.grid(True)

    # Theta range error comparison
    plt.subplot(2, 3, 3)
    plt.plot(time_steps, error_noisy_no_ekf_theta, label='Error with Noise, No EKF', color='orange')
    plt.plot(time_steps, error_noisy_with_ekf_theta, label='Error with Noise, with EKF', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Theta Error [rad]')
    plt.title('Theta Angle Error Comparison')
    plt.legend()
    plt.grid(True)

    # Extract trajectory data
    noisy_x = [pos[0] for pos in base_pos_all] 
    noisy_y = [pos[1] for pos in base_pos_all]  
    ekf_x = [pos[0] for pos in ekf_pos_all]  
    ekf_y = [pos[1] for pos in ekf_pos_all] 

    plt.subplot(2, 1, 2)
    plt.plot(noisy_x, noisy_y, label="Noisy without EKF", linestyle="--", color="red", marker="o", markersize=2)
    plt.plot(ekf_x, ekf_y, label="Estimated with EKF", color="blue", marker="o", markersize=2)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Trajectory Comparison: Noisy without EKF vs. Estimated with EKF")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if not os.path.exists("images/task3_4"):
        os.makedirs("images/task3_4")
    #! Uncomment the following line to save the plot
    # plt.savefig("images/task3_4/results_2.png")
    plt.show()

    #! Task 4 Results
    # Calculate steady-state error
    steady_state_error_x = np.mean(error_noisy_with_ekf_x[-steady_state_window:])
    steady_state_error_y = np.mean(error_noisy_with_ekf_y[-steady_state_window:])
    steady_state_error_theta = np.mean(error_noisy_with_ekf_theta[-steady_state_window:])
    print("\033[91m========================== Results =============================\033[0m")
    print("Steady-State Error (x, y, theta):", steady_state_error_x, steady_state_error_y, steady_state_error_theta)
    print("Stabilization Time:", stabilization_time)
    print("Overshoot (x, y, theta):", overshoot_x, overshoot_y, overshoot_theta)
    print("\033[91m================================================================\033[0m")

if __name__ == '__main__':
    #! Please make sure you open the noise flag in the configuration file before running task 3 and 4
    if len(sys.argv) > 1 and sys.argv[1] == "task_3_4":
        main_task_3_4()
    elif len(sys.argv) > 1 and sys.argv[1] == "task_2":
        main_task_2()
    else:
        main_task_2()
