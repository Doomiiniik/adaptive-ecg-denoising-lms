import numpy as np
import matplotlib.pyplot as plt
import os
from generate_signals import generate_data 


class LMSFilter:
    def __init__(self, mu, num_taps):   #num_taps = number of weights
        self.mu = mu                    #mu = step size, sould be between 0 and lambda
        self.w = np.zeros(num_taps)     #w = the weights (filter memory)
        self.buffer = np.zeros(num_taps)#buffer = storage of the recent reference samples

    def process(self, primary, reference):
        """
        Processes the noisy signal using the LMS algorithm.
        use array of noisy signal and array of the reference signal
        """
        N = len(primary) #number of the samples
        y = np.zeros(N) # the filter's prediction of noise
        e = np.zeros(N) # the error (cleaned signal, only in this specific appliacation)
        
        eps = 1e-8 # small constant to avoid division by zero
        for n in range(N):
            self.buffer = np.roll(self.buffer, 1) #shift buffer to the right by one position
            self.buffer[0] = reference[n] 
            y[n] = np.dot(self.w, self.buffer) # convolution in one point
            e[n] = primary[n] - y[n] 
            norm = np.dot(self.buffer, self.buffer) + eps
            mu_n = self.mu / norm            # normalized step
            self.w += mu_n * e[n] * self.buffer

                
        return e, y

def calculate_snr(clean, processed):
    # noise is the difference between our result and the "perfect" signal
    noise_residual = clean - processed
    
    # calculate power ratio
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise_residual**2)
    
    # result in dB
    return 10 * np.log10(signal_power / noise_power)


def plot_spectral_analysis(x_noisy, e_out, fs, output_dir):
    plt.figure(figsize=(12, 6))
    
    for signal, label, color in zip([x_noisy, e_out], ['Noisy Input', 'LMS Filtered'], ['r', 'b']):
        # Compute Power Spectral Density
        fft_vals = np.fft.rfft(signal)
        psd = np.abs(fft_vals)**2
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        
        plt.semilogy(freqs, psd, color, label=label, alpha=0.7)

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
    
    FS = 1000
    T_total = 12.0
    n_idx, s_clean, x_noisy, r_ref = generate_data(FS, T_total)
    
    # 1. Normalize based on the Noisy signal
    norm_factor = np.max(np.abs(x_noisy))
    x_noisy /= norm_factor
    s_clean /= norm_factor
    r_ref /= np.max(np.abs(r_ref)) # Reference normalized to its own peak

    # 2. Run the filter with slightly higher mu for faster learning
    lms = LMSFilter(mu=0.01, num_taps=128) # Increased taps for drift
    cleaned_signal, noise_estimate = lms.process(x_noisy, r_ref)
    
    # 3. SNR Calculation (Fair Comparison)
    # We use the same 'start_index' for BEFORE and AFTER 
    # so we aren't comparing a whole signal to a small piece.
    start_index = int(len(s_clean) * 0.3)
    
    snr_before = calculate_snr(s_clean[start_index:], x_noisy[start_index:])
    snr_after = calculate_snr(s_clean[start_index:], cleaned_signal[start_index:])
   
    print(f"SNR before filtering: {snr_before:.2f} dB")
    print(f"SNR after filtering (Stabilized): {snr_after:.2f} dB")














    plot_spectral_analysis(x_noisy, cleaned_signal, FS, output_dir)



    # saving plots into file
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(n_idx, x_noisy, 'r', alpha=0.6, label='Noisy ECG (Input)')
    plt.title("Before: Corrupted Signal")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(n_idx, cleaned_signal, 'b', label='Cleaned ECG (Output)')
    plt.title("After: LMS Adaptive Filter Result")
    plt.legend()
    
    # Let's plot the "Learning Curve" - Squared Error
    plt.subplot(3, 1, 3)
    plt.plot(n_idx, (s_clean - cleaned_signal)**2, 'k', label='Squared Error')
    plt.title("Errors: (s_clean - cleaned_signal)^2")
    plt.xlabel("Sample Index")
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'lms_filter_results.png')
    plt.savefig(save_path)
    print(f"Filter complete! Results saved to: {save_path}")