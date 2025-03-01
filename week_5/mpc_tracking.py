import numpy as np
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, dyn_cancel, SinusoidalReference
from tracker_model import TrackerModel

#! Some lab switches
# Control mode
velocity_control = True  # Set to True for velocity control, False for position control only
# Trajectory type
trajectory_type = "sinusoidal"  # Set to "sinusoidal"/"linear"/"polynomial"
# Test duration
test_duration = 5  # Duration of the test in seconds

class LinearReference:
    def __init__(self, start, end, duration):
        self.start = start
        self.end = end
        self.duration = duration
        self.start_time = None

    def get_values(self, current_time):
        if self.start_time is None:
            self.start_time = current_time

        elapsed_time = current_time - self.start_time
        t = elapsed_time / self.duration
        q_d = self.start + t * (self.end - self.start)
        if elapsed_time < self.duration:
            qd_d = (self.end - self.start) / self.duration
        else:
            qd_d = np.zeros_like(self.start)
        return q_d, qd_d

class PolynomialReference:
    def __init__(self, start, end, duration, start_velocity=0, end_velocity=0):
        self.start = start
        self.end = end
        self.duration = duration
        self.start_velocity = start_velocity
        self.end_velocity = end_velocity
        self.coefficients = self.compute_coefficients()

    def compute_coefficients(self):
        a0 = self.start
        a1 = self.start_velocity
        a2 = (3 * (self.end - self.start) / (self.duration ** 2)) - (self.end_velocity / self.duration)
        a3 = (-2 * (self.end - self.start) / (self.duration ** 3)) + (self.end_velocity / (self.duration ** 2))
        return [a0, a1, a2, a3]

    def get_values(self, current_time):
        t = current_time
        a0, a1, a2, a3 = self.coefficients
        q_d = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3
        qd_d = a1 + 2 * a2 * t + 3 * a3 * t ** 2
        return q_d, qd_d

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
    

def getSystemMatricesContinuos(num_joints, damping_coefficients=None):
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
    num_states = 2 * num_joints
    num_controls = num_joints
    
    
    # Initialize A matrix
    A = np.zeros((num_states,num_states))
    
    # Upper right quadrant of A (position affected by velocity)
    A[:num_joints, num_joints:] = np.eye(num_joints) 
    
    # Lower right quadrant of A (velocity affected by damping)
    #if damping_coefficients is not None:
    #    damping_matrix = np.diag(damping_coefficients)
    #    A[num_joints:, num_joints:] = np.eye(num_joints) - time_step * damping_matrix
    
    # Initialize B matrix
    B = np.zeros((num_states, num_controls))
    
    # Lower half of B (control input affects velocity)
    B[num_joints:, :] = np.eye(num_controls) 
    
    return A, B

# Example usage:
# sim = YourSimulationObject()
# num_joints = 6  # Example: 6-DOF robot
# damping_coefficients = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]  # Example damping coefficients
# A, B = getSystemMatrices(sim, num_joints, damping_coefficients)


def getCostMatrices(num_joints):
    """
    Get the cost matrices Q and R for the MPC controller.
    
    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    # Q = 1 * np.eye(num_states)  # State cost matrix
    p_w = 10000 # Set the weight for the position
    if (velocity_control):
        v_w = 10 # Set the weight for the velocity
    else:
        v_w = 0
    Q_diag = np.array([p_w, p_w, p_w,p_w, p_w, p_w,p_w, v_w, v_w, v_w,v_w, v_w, v_w,v_w])
    Q = np.diag(Q_diag)

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

    # Define the matrices
    A, B = getSystemMatricesContinuos(num_joints)
    Q, R = getCostMatrices(num_joints)
    
    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    tracker = TrackerModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states, sim.GetTimeStep())
    # Compute the matrices needed for MPC optimization
    S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar = tracker.propagation_model_tracker_fixed_std()
    H,Ftra = tracker.tracker_std(S_bar, T_bar, Q_hat, Q_bar, R_bar)
    
    # Set trajectory reference
    if trajectory_type == "sinusoidal":
        # Sinusoidal reference
        # Specify different amplitude values for each joint
        amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
        # Specify different frequency values for each joint
        frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

        # Convert lists to NumPy arrays for easier manipulation in computations
        amplitude = np.array(amplitudes)
        frequency = np.array(frequencies)
        ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    elif trajectory_type == "linear":
        # Linear reference
        start = sim.GetInitMotorAngles()
        end = np.array([1, 1, 1, 1, 1, 1, 1])   # Example end position
        duration = test_duration  # Example duration in seconds
        ref = LinearReference(start, end, duration)  # Initialize the reference
    elif trajectory_type == "polynomial":
        # Polynomial reference
        start = sim.GetInitMotorAngles()
        end = np.array([1, 1, 1, 1, 1, 1, 1])
        duration = test_duration
        start_velocity = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        end_velocity = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        ref = PolynomialReference(start, end, duration, start_velocity, end_velocity)
    else:
        raise ValueError("Invalid trajectory type. Choose 'sinusoidal', 'linear', or 'polynomial'.")

    # Main control loop
    episode_duration = test_duration # duration in seconds
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration/time_step)
    sim.ResetPose()
    # sim.SetSpecificPose([1, 1, 1, 0.4, 0.5, 0.6, 0.7])
    # testing loop
    u_mpc = np.zeros(num_joints)
    for i in range(steps):
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        x_ref = []
        # generate the predictive trajectory for N steps
        for j in range(N_mpc):
            q_d, qd_d = ref.get_values(current_time + j*time_step)
            # here i need to stack the q_d and qd_d
            x_ref.append(np.vstack((q_d.reshape(-1, 1), qd_d.reshape(-1, 1))))
        
        x_ref = np.vstack(x_ref).flatten()
        

        # Compute the optimal control sequence
        u_star = tracker.computesolution(x_ref,x0_mpc,u_mpc, H, Ftra)
        # Return the optimal control sequence
        u_mpc += u_star[:num_joints]
       
        # Control command
        tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)
        sim.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)

        q_d, qd_d = ref.get_values(current_time)

        q_d_all.append(q_d)
        qd_d_all.append(qd_d)

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        #print(f"Time: {current_time}")
    
    
    
    # Plotting
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i+1}', linestyle='--')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i+1}', linestyle='--')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    
    main()