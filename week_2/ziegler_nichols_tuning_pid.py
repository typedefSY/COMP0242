import os 
import numpy as np
from numpy.fft import fft, fftfreq
# import time
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin


# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")

def detect_steady_oscillation(sim_, errors, min_cycles=3, time_tolerance=0.1, amplitude_tolerance=0.08):
    peaks, _ = find_peaks(np.abs(errors))
    if len(peaks) >= min_cycles:
        peak_periods = np.diff(peaks) * sim_.GetTimeStep()
        errors = np.array(errors)
        peak_values = np.abs(errors[peaks])
        print("\033[91m========================== Standard Deviation =============================\033[0m")
        print(f'amplitude std: {np.std(peak_values)}')
        print(f'time std: {np.std(peak_periods)}')
        print("\033[91m===========================================================================\033[0m")
        # if Standard Deviation of time and amplitude are both less or equal to 0.1, oscillation is true
        if np.std(peak_periods) < time_tolerance and np.std(peak_values) <= amplitude_tolerance:
            return True
    return False

# single joint tuning
#episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False, kd=0):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # updating the kp value for the joint we want to tune
    kp_vec = np.array([1000]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd_vec = np.array([0]*dyn_model.getNumberofActuatedJoints())
    kd_vec[joints_id] = kd
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement 

   
    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors


    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all, times_all, errors_all = [], [], [], [], [], []
    

    steps = int(episode_duration/time_step)
    Ku = 0
    # testing loop
    for _ in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        # qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0) # Ignore qdd for this lab
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()
        errors = q_des - q_mes
        errors_all.append(errors[joints_id])

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        times_all.append(current_time)
        current_time += time_step
        # print("current time in seconds",current_time)

    
    # TODO make the plot for the current joint
    q_mes_all = np.array(q_mes_all)
    q_d_all = np.array(q_d_all)
    times_all = np.array(times_all)

    oscillation_detected = detect_steady_oscillation(sim_, errors_all)
    if plot or oscillation_detected:
        Ku = kp
        plt.figure()
        plt.plot(times_all, q_mes_all[:, joints_id], label='Measured Angle')
        plt.plot(times_all, q_d_all[:, joints_id], label='Desired Angle', linestyle='--')
        plt.xlabel('Time [s]')
        plt.ylabel('Joint Angle [rad]')
        plt.title(f'Joint {joints_id} with Kp={kp} and Kd={kd}')
        plt.legend()
        plt.show()
    
    
    return q_mes_all, Ku
     
def perform_frequency_analysis(data, dt, joint_index=0):
    n = len(data)
    yf = fft(data[:, joint_index]) # Only consider the specified joint
    xf = fftfreq(n, dt)[:n//2]
    # print(f'shape of xf: {xf.shape}')
    power = 2.0/n * np.abs(yf[:n//2])

    # Optional: Plot the spectrum
    # plt.figure()
    # plt.plot(xf, power)
    # plt.title("FFT of the signal")
    # plt.xlabel("Frequency in Hz")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    
    # plt.show()

    return xf, power


if __name__ == '__main__':
    joint_id = 0  # Joint ID to tune
    regulation_displacement = 1.0  # Displacement from the initial joint position
    init_gain=16
    gain_step=0.01
    max_gain=10000 
    test_duration=20 # in seconds
    

    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method

    # for joint_id in range(num_joints):  # Iterate over each joint
    kp = init_gain
    while kp < max_gain:
        q_mes_all, Ku = simulate_with_given_pid_values(sim, kp, joint_id, regulation_displacement, test_duration, plot=False)
        xf, power = perform_frequency_analysis(q_mes_all, sim.GetTimeStep())
        power = power[1:]
        print("\033[92m==================== Kp and Dominant frequency ============================\033[0m")
        print(f"Kp: {kp}, Dominant frequency: {xf[np.argmax(power)]}")
        print("\033[92m===========================================================================\033[0m")
        if Ku > 0:
            Tu = 1/xf[np.argmax(power)]
            Td = 0.125 * Tu
            Kp = 0.8 * Ku
            Kd = 0.1* Tu * Ku
            print("\033[92m======================= Final Kp, Td and Kd ===============================\033[0m")
            print(f"Kp: {Kp}, Td: {Td}, Kd: {Kd}")
            print("\033[92m===========================================================================\033[0m")
            q_mes_all, Ku = simulate_with_given_pid_values(sim, kp, joint_id, regulation_displacement, test_duration, plot=True, kd=Kd)
            break
        kp += gain_step  # Increment Kp for the next iteration
    if kp >= max_gain:
        print(f"Could not find suitable Kp for joint {joint_id} within max_gain limit")
        