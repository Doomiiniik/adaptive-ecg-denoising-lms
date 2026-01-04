"""
ECG Adaptive Noise Cancellation using NLMS
------------------------------------------
This script implements a Normalized Least Mean Squares (NLMS) filter 
to remove 50Hz powerline interference and baseline wander from ECG signals.

Features:
- Adaptive noise estimation and subtraction.
- Signal-to-Noise Ratio (SNR) validation.
- R-peak detection for Heart Rate (BPM) calculation.
- Spectral analysis using FFT.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from .generate_signals import generate_data 


class NLMSFilter:
    def __init__(self, mu, num_taps):   #num_taps = number of weights
        self.mu = mu                    #mu = step size, learning rate
        self.w = np.zeros(num_taps)     #w = the weights (filter memory)
        self.buffer = np.zeros(num_taps)#buffer, storage of the recent reference samples

    def process(self, primary, reference):
        """
        Processes the noisy signal using the NLMS algorithm.
        use array of noisy signal and array of the reference signal
        """
        N = len(primary) # number of the samples
        y = np.zeros(N)  # the filter's prediction of noise
        e = np.zeros(N)  # the error (cleaned signal)
        
        # Note: In Adaptive Noise Cancellation, we subtract the estimated noise (y)
        # from the noisy input (primary). The remaining "Error" is the clean signal.


        eps = 1e-8 # small constant to avoid division by zero
        for n in range(N):
            self.buffer = np.roll(self.buffer, 1) # shift buffer to the right
            self.buffer[0] = reference[n] 
            y[n] = np.dot(self.w, self.buffer) # dot product (convolution in one step)
            e[n] = primary[n] - y[n] 
            norm = np.dot(self.buffer, self.buffer) + eps # power of the input signal
            mu_n = self.mu / norm            # normalized step
            self.w += mu_n * e[n] * self.buffer # weight update rule

                
        return e, y

def calculate_snr(clean, processed): # signal to noise ratio
    # noise is the difference between our result and the "perfect" signal
    noise_residual = clean - processed
    
    # calculate power ratio
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise_residual**2)
    
    # result in dB
    return 10 * np.log10(signal_power / noise_power)

def calculate_bpm(signal, fs):
    # square the signal to make peaks stand out and remove negatives
    squared_signal = signal**2
    
    # set a threshold (e.g., 50% of the max peak)
    threshold = np.max(squared_signal) * 0.5
    
    # find indices where the signal crosses the threshold
    # we use a 'distance' check so we don't count the same peak twice
    peaks = []
    min_distance = int(fs * 0.4)  # 0.4s (max heart rate of ~150 BPM)
    
    last_peak = -min_distance # initialize to allow first peak detection
    for i in range(len(squared_signal)):
        if squared_signal[i] > threshold and (i - last_peak) > min_distance:
            peaks.append(i)
            last_peak = i
            
    # when less than 2 peaks are detected, BPM cannot be calculated
    if len(peaks) < 2:
        return 0
        
    intervals = np.diff(peaks) / fs  # time between peaks in seconds
    average_interval = np.mean(intervals)
    bpm = 60 / average_interval
    return bpm








def plot_spectral_analysis(x_noisy, e_out, fs, output_dir):
    plt.figure(figsize=(12, 6))
    
    for signal, label, color in zip([x_noisy, e_out], ['Noisy Input', 'LMS Filtered'], ['r', 'b']):
        
        fft_vals = np.fft.rfft(signal) # only positive frequencies
        psd = np.abs(fft_vals)**2  # Power Spectral Density
        freqs = np.fft.rfftfreq(len(signal), 1/fs) 
        
        plt.semilogy(freqs, psd, color, label=label, alpha=0.7)
    # plot in log scale for better visibility
    plt.title("Power Spectral Density (Frequency Content)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim(0, 100)  # Zoom in on the relevant 0-100Hz range
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    save_path = os.path.join(output_dir, 'fft_results.png')
    plt.savefig(save_path)
    print(f"Filter complete! Results saved to: {save_path}")



if __name__ == "__main__":
    # paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'results', 'plots')
    
    # create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")






    FS = 1000
    T_total = 12.0
    n_idx, s_clean, x_noisy, r_ref = generate_data(FS, T_total)
    
    # normalize based on the noisy signal
    # for better numerical stability
    norm_factor = np.max(np.abs(x_noisy))
    x_noisy /= norm_factor
    s_clean /= norm_factor
    r_ref /= np.max(np.abs(r_ref)) # Reference normalized to its own peak

    # remember about mu and num_taps configurations
    lms = NLMSFilter(mu=0.01, num_taps=128)
    cleaned_signal, noise_estimate = lms.process(x_noisy, r_ref)
    
    # SNR Calculation (Fair Comparison)
    # We use the same 'start_index' for BEFORE and AFTER 
    # so we arent count early phase where filter is still converging
    start_index = int(len(s_clean) * 0.3)
    
    snr_before = calculate_snr(s_clean[start_index:], x_noisy[start_index:])
    snr_after = calculate_snr(s_clean[start_index:], cleaned_signal[start_index:])
   
    print(f"SNR before filtering: {snr_before:.2f} dB")
    print(f"SNR after filtering (Stabilized): {snr_after:.2f} dB")

    # BPM Calculation
    bpm_before = calculate_bpm(x_noisy[start_index:], FS)
    bpm_after = calculate_bpm(cleaned_signal[start_index:], FS)
    print(f"Estimated BPM before filtering: {bpm_before:.1f} BPM")
    print(f"Estimated BPM after filtering: {bpm_after:.1f} BPM")
    

    # Spectral Analysis Plots
    plot_spectral_analysis(x_noisy, cleaned_signal, FS, output_dir)



    # saving plots into file
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(n_idx, x_noisy, 'r', alpha=0.6, label='Noisy ECG (Input)')
    plt.title("Before: Corrupted Signal")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(n_idx, cleaned_signal, 'b', label='Cleaned ECG (Output)')
    plt.title("After: LMS Adaptive Filter Result(Detected BPM: {bpm_after:.1f})")
    plt.legend()
    
    
    plt.subplot(3, 1, 3)
    plt.plot(n_idx, (s_clean - cleaned_signal)**2, 'k', label='Squared Error')
    plt.title("Errors: (s_clean - cleaned_signal)^2")
    plt.xlabel("Sample Index")
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'lms_filter_results.png')
    plt.savefig(save_path)
    print(f"Filter complete! Results saved to: {save_path}")