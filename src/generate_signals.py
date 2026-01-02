import numpy as np
import matplotlib.pyplot as plt
import os





def generate_data(fs, T):
    """
    Generates synthetic ECG data for Adaptive Filtering.
    
    Parameters:
    fs (int): Sampling Frequency [Hz]
    T  (float): Total duration of the signal, time of full signal [s]
    
    Returns:
    n (array): Discrete time index
    s (array): Clean signal (sygnał użyteczny)
    x (array): Primary input (sygnał wejściowy zaszumiony)
    r (array): Reference signal (sygnał odniesienia)
    """
    
    
    # N is the total number of samples 
    # N = T * fs = time of the signal * amount of samples per second
    N = int(T * fs)
    
    # n is array with the sample numebers
    n = np.arange(N)
    
    # t is the array with the totall time after every sample
    t = n / fs 
    
    # Generate s[n] - The Clean Signal (Target)
    # Simulating a heartbeat (QRS complex) using Gaussian pulses
    f_heart = 1.2  # 1.2 Hz = ~72 Beats Per Minute
    s = np.zeros(N)
    initial_phase= 0.5 # half of a cycle
    
    num_beats = int(T * f_heart) #total number of heart beats in the time of signal
    for i in range(num_beats):
        # Position of the beat in seconds
        pos = (i + initial_phase) / f_heart
        # Add a Gaussian pulse at that position
        s += np.exp(-1000 * (t - pos)**2) # the highest pulse in pos time, 
        #others are goint to 0
    
    # Generate v[n] - The Actual Interference
    # This is the 50Hz noise polluting the patient's body.
    # It has an unknown phase shift (phi) relative to the wall socket.
    # It interupts the ECD signall, whotch is common medical diagnostic problem
    # electricity in electrical installation is 230V while 
    # our heart is roughly 0.001V
    # Even if the wires aren't touching the patient, the "electricity in the walls 
    # jumps" through the air via electric fields. This is called Capacitive Coupling
    


    f_line = 50.0  # Power-line frequency
    A_v = 0.5      # Amplitude of the noise
    phi = np.pi/4  # Phase delay (Opóźnienie fazowe)
    
    
    v = A_v * np.sin(2 * np.pi * f_line * t + phi)
    
    # x[n] - The Primary Input
    # This is what the sensor actually records: Signal + Noise
    x = s + v
    
    # 5. Generate r[n] - The Reference Signal
    # This simulates measuring the voltage directly from the wall outlet.
    # It is correlated with v[n], but has different phase/amplitude.
    r = np.sin(2 * np.pi * f_line * t)
    # we deliberetly give it no phase to make filter do some work



    return n, s, x, r

if __name__ == "__main__":

 

# 1. Get the directory where THIS script is located (the 'src' folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go one level up to the main project directory (ECG_Adaptive_Filter)
    project_root = os.path.dirname(script_dir)

# 3. Define the path to the results/plots folder
    output_dir = os.path.join(project_root, 'results', 'plots')

# 4. Create the folder if it doesn't exist
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
      print(f"Created directory: {output_dir}")




    # Simulation Parameters
    FS = 1000  # Hz
    T_total = 2.0  # Seconds
    
    # Generate the data
    n_idx, s_clean, x_noisy, r_ref = generate_data(FS, T_total)
    
    
    
    # --- Plotting ---
# We set a larger figure size (width=10, height=8) for 3 subplots
    plt.figure(figsize=(10, 8))

# Plot 1: The Clean Heartbeat s[n]
    plt.subplot(3, 1, 1)
    plt.plot(n_idx, s_clean, 'g', label='s[n] (Clean)')
    plt.title("s[n]: Clean Signal (What we want to recover)")
    plt.legend()
    plt.grid(True)

# Plot 2: The Corrupted Input x[n]
    plt.subplot(3, 1, 2)
    plt.plot(n_idx, x_noisy, 'r', label='x[n] = s[n] + v[n]')
    plt.title("x[n]: Primary Input (What the sensor records)")
    plt.legend()
    plt.grid(True)

# Plot 3: The Reference r[n]
    plt.subplot(3, 1, 3)
    plt.plot(n_idx, r_ref, 'b', label='r[n] (Reference)')
    plt.title("r[n]: Reference Noise (Helper signal)")
    plt.xlabel("Sample Index [n]")
    plt.legend()
    plt.grid(True)

    # This prevents titles from overlapping
    plt.tight_layout()

    # --- THE SAVING PART ---
    # Use the output_dir we defined earlier
    file_name = "ecg_signal_generation.png"
    save_path = os.path.join(output_dir, file_name)

    plt.savefig(save_path)
    print(f"Success! Image saved to: {save_path}")

    # Close the figure to free up memory (good practice in DSP)
    plt.close()