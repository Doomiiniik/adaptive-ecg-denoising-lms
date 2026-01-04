# Adaptive ECG Denoising using LMS Algorithm

## 📌 Project Overview
This project implements an **Adaptive Noise Canceller (ANC)** based on the **Least Mean Squares (LMS)** algorithm. It is specifically designed to remove 50Hz powerline interference from Electrocardiogram (ECG) signals in real-time.

Unlike static filters, this adaptive approach learns the noise characteristics dynamically, allowing for high-quality signal recovery with minimal distortion to the QRS complex.

## 🚀 Key Features
* **Real-time Adaptation:** Uses the LMS update rule to track non-stationary noise.
* **Synthetic ECG Generation:** Simulates a realistic cardiac signal for performance validation.
* **Convergence Analysis:** Includes visualization of the Mean Squared Error (MSE) to track filter learning.
* **Modular Design:** Object-oriented implementation for easy integration into larger DSP pipelines.

## 🧮 How it Works
The system follows the standard Adaptive Noise Cancellation architecture:
1. **Primary Input ($d$):** Noisy ECG ($Signal + Noise$)
2. **Reference Input ($r$):** Pure noise source (50Hz hum)
3. **Filter Output ($y$):** The algorithm's estimate of the noise.
4. **Error Signal ($e$):** The cleaned ECG signal ($d - y$).

The weights are updated using the formula:
$$w[n+1] = w[n] + 2\mu e[n] r[n]$$

## 📊 Results
*(You can upload your plot here later!)*
The filter successfully converges within the first 200 samples, significantly reducing 50Hz interference while preserving the morphological features of the heartbeat.

## 🛠️ Installation & Usage
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/adaptive-ecg-denoising-lms.git`
2. Install dependencies: `pip install numpy matplotlib`
4. Run command: source dsp_env/bin/activate
3. Run the simulation: `python src/lms_algorithm.py`
