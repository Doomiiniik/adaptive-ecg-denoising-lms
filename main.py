"""
ecg adaptive noise cancellation - main execution logic
------------------------------------------------------
this script coordinates the signal generation and the nlms filter.
it handles directory management, data flow, and performance reporting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# add the current directory to the path so python sees the src package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.generate_signals import generate_data
from src.lms_algorithm import NLMSFilter, calculate_snr, calculate_bpm
def run_diagnostic_pipeline():
    # define pathing for results
    # finds the base project folder to ensure portability
    root_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(root_dir, 'results', 'plots')
    
    # create the output folder if it is missing
    os.makedirs(plot_dir, exist_ok=True)

    # configuration parameters
    # sampling rate and duration for the simulation
    fs_val = 1000
    seconds = 12.0
    
    # generate the synthetic dataset
    # returns clean signal, noisy input, and reference noise
    indices, clean, noisy, ref = generate_data(fs_val, seconds)

    # signal normalization
    # scaling ensures numerical stability during weight updates
    peak_val = np.max(np.abs(noisy))
    noisy /= peak_val
    clean /= peak_val
    ref /= np.max(np.abs(ref))

    # filter initialization and processing
    # applying the normalized least mean squares algorithm
    mu_val = 0.01
    taps_count = 128
    adaptive_filter = NLMSFilter(mu=mu_val, num_taps=taps_count)
    output_signal, noise_est = adaptive_filter.process(noisy, ref)

    # performance evaluation
    # ignoring the initial convergence phase for accuracy
    trim_point = int(len(clean) * 0.3)
    
    snr_initial = calculate_snr(clean[trim_point:], noisy[trim_point:])
    snr_final = calculate_snr(clean[trim_point:], output_signal[trim_point:])
    heart_rate = calculate_bpm(output_signal[trim_point:], fs_val)

    # terminal reporting
    print("-" * 40)
    print("adaptive filter performance report")
    print("-" * 40)
    print(f"initial snr: {snr_initial:.2f} db")
    print(f"final snr:   {snr_final:.2f} db")
    print(f"snr gain:    {snr_final - snr_initial:.2f} db")
    print(f"detected bpm: {heart_rate:.1f}")
    print("-" * 40)

    # data visualization
    plt.figure(figsize=(12, 10))

    # input signal plot
    plt.subplot(3, 1, 1)
    plt.plot(indices, noisy, 'r', alpha=0.7, label='noisy input')
    plt.title("corrupted signal with interference and drift")
    plt.legend()

    # filtered signal plot
    plt.subplot(3, 1, 2)
    plt.plot(indices, output_signal, 'b', label='filtered output')
    plt.title(f"recovered ecg signal - heart rate: {heart_rate:.1f} bpm")
    plt.legend()

    # convergence error plot
    plt.subplot(3, 1, 3)
    plt.plot(indices, (clean - output_signal)**2, 'k', alpha=0.5, label='squared error')
    plt.title("filter error convergence")
    plt.xlabel("sample index")
    plt.legend()

    plt.tight_layout()
    
    # exporting the visualization
    save_name = os.path.join(plot_dir, 'final_report.png')
    plt.savefig(save_name)
    print(f"report saved successfully to: {save_name}")

if __name__ == "__main__":
    run_diagnostic_pipeline()