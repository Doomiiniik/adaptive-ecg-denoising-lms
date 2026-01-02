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
        
        for n in range(N):
            # update the buffer (shift old samples, add new one)
            self.buffer = np.roll(self.buffer, 1) # np.roll -> move all buffer one position to the right
            self.buffer[0] = reference[n] #empty place repace with the reference signal sample
            
            # predict the noise (y = weights * buffer)
            y[n] = np.dot(self.w, self.buffer) #convolution, but only in one point, sum w*buffer[0..buf_size]
            
            # calculate the Error/Clean Signal (e = x - y)
            e[n] = primary[n] - y[n]
            
            # update weights (the learning step)
            # w = w + 2 * mu * error * reference_sample
            self.w = self.w + 2 * self.mu * e[n] * self.buffer
            
        return e, y

if __name__ == "__main__":
    # paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'results', 'plots')
    
    # generating data
    FS = 1000
    T_total = 2.0
    n_idx, s_clean, x_noisy, r_ref = generate_data(FS, T_total)
    
    # --- 3. FILTERING ---
    # mu: Learning rate (Try 0.01)
    # num_taps: Filter memory (Try 32)
    lms = LMSFilter(mu=0.02, num_taps=32)
    cleaned_signal, noise_estimate = lms.process(x_noisy, r_ref)
    
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