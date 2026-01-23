"""
Agent 1: Signal Processing Agent
Responsible for ECG signal cleaning and artifact removal
"""

import numpy as np
from scipy import signal

class SignalFilterAgent:
    """Agent 1: Responsible for ECG signal cleaning"""

    def __init__(self):
        self.name = "Signal Filter Agent"
        self.status = "Standby"
        self.processing_log = []

    def filter_ecg(self, raw_signal, sampling_rate=250):
        """
        Clean ECG signal by removing noise

        Processing steps:
        1. Highpass filter: Remove baseline wander
        2. Lowpass filter: Remove high-frequency noise
        3. Notch filter: Remove power line interference (50/60 Hz)

        Args:
            raw_signal: Raw ECG data (numpy array)
            sampling_rate: Sampling rate (Hz)

        Returns:
            cleaned_signal: Cleaned signal
        """
        self.status = "Processing..."
        self.processing_log = []

        try:
            # 1. Remove baseline wander (highpass filter, cutoff 0.5 Hz)
            sos_high = signal.butter(4, 0.5, 'highpass', fs=sampling_rate, output='sos')
            signal_high = signal.sosfilt(sos_high, raw_signal)
            self.processing_log.append("[OK] Baseline wander removed (highpass 0.5 Hz)")

            # 2. Remove high-frequency noise (lowpass filter, cutoff 40 Hz)
            sos_low = signal.butter(4, 40, 'lowpass', fs=sampling_rate, output='sos')
            signal_filtered = signal.sosfilt(sos_low, signal_high)
            self.processing_log.append("[OK] High-frequency noise removed (lowpass 40 Hz)")

            # 3. Remove power line interference (notch filter, 50 Hz)
            # Note: Change to 60 Hz for US and other regions
            b_notch, a_notch = signal.iirnotch(50, 30, sampling_rate)
            cleaned_signal = signal.filtfilt(b_notch, a_notch, signal_filtered)
            self.processing_log.append("[OK] Power line interference removed (notch 50 Hz)")

            # 4. Normalization
            cleaned_signal = (cleaned_signal - np.mean(cleaned_signal)) / np.std(cleaned_signal)
            self.processing_log.append("[OK] Signal normalized")

            self.status = "Done"

            return cleaned_signal

        except Exception as e:
            self.status = f"Error: {str(e)}"
            self.processing_log.append(f"[FAIL] Processing failed: {str(e)}")
            raise

    def detect_artifacts(self, signal_data, threshold_multiplier=3.0):
        """
        Detect motion artifacts

        Uses simple statistical method:
        - If a segment's std exceeds N times the overall std, it's considered an artifact

        Args:
            signal_data: Signal data
            threshold_multiplier: Threshold multiplier

        Returns:
            list: Detected artifact descriptions
        """
        artifacts = []

        # Calculate overall statistics
        overall_std = np.std(signal_data)
        overall_mean = np.mean(signal_data)

        # Check segments (1 second each)
        window_size = 250  # Assuming 250 Hz sampling rate
        n_windows = len(signal_data) // window_size

        artifact_count = 0
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window = signal_data[start:end]

            window_std = np.std(window)

            # If this segment's std is abnormally high
            if window_std > overall_std * threshold_multiplier:
                artifact_count += 1

        # Describe based on artifact count
        if artifact_count > n_windows * 0.3:
            artifacts.append(f"[WARN] Detected heavy motion artifacts ({artifact_count}/{n_windows} segments)")
            artifacts.append("   Possible cause: Continuous head movement or speaking")
        elif artifact_count > n_windows * 0.1:
            artifacts.append(f"[WARN] Detected motion artifacts ({artifact_count}/{n_windows} segments)")
            artifacts.append("   Possible cause: Occasional body movement")

        # Check overall signal quality
        if overall_std > 2.0:
            artifacts.append("[WARN] High overall signal variance, may contain significant noise")

        if not artifacts:
            artifacts.append("[OK] Signal quality good, no obvious artifacts detected")

        return artifacts

    def get_quality_metrics(self, raw_signal, cleaned_signal):
        """
        Calculate signal quality metrics

        Args:
            raw_signal: Raw signal
            cleaned_signal: Cleaned signal

        Returns:
            dict: Quality metrics
        """
        # Calculate signal energy reduction percentage (represents noise removed)
        raw_energy = np.sum(raw_signal ** 2)
        cleaned_energy = np.sum(cleaned_signal ** 2)
        noise_reduction = (1 - cleaned_energy / raw_energy) * 100

        # Calculate Signal-to-Noise Ratio (SNR)
        signal_power = np.mean(cleaned_signal ** 2)
        noise = raw_signal - cleaned_signal
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        return {
            "noise_reduction_percent": round(noise_reduction, 2),
            "signal_to_noise_ratio_db": round(snr, 2),
            "raw_std": round(np.std(raw_signal), 4),
            "cleaned_std": round(np.std(cleaned_signal), 4)
        }

    def get_log(self):
        """Get processing log"""
        return self.processing_log

# Test program
if __name__ == "__main__":
    print("=" * 50)
    print("Agent 1 Test - Signal Processing Agent")
    print("=" * 50)
    print()

    # Create test signal
    t = np.linspace(0, 10, 2500)  # 10 seconds, 250 Hz
    # Simulate heartbeat
    heart_signal = np.sin(2 * np.pi * 1.2 * t)
    # Add baseline wander
    baseline = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    # Add high-frequency noise
    noise = 0.2 * np.random.randn(len(t))
    # Combine
    test_signal = heart_signal + baseline + noise

    # Test Agent
    agent1 = SignalFilterAgent()
    print(f"Agent status: {agent1.status}")
    print()

    print("Starting signal processing...")
    cleaned = agent1.filter_ecg(test_signal, sampling_rate=250)

    print("\nProcessing log:")
    for log in agent1.get_log():
        print(f"  {log}")

    print(f"\nFinal status: {agent1.status}")

    # Detect artifacts
    print("\nArtifact detection:")
    artifacts = agent1.detect_artifacts(test_signal)
    for artifact in artifacts:
        print(f"  {artifact}")

    # Quality metrics
    print("\nSignal quality metrics:")
    metrics = agent1.get_quality_metrics(test_signal, cleaned)
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("[OK] Agent 1 test complete!")
    print("=" * 50)
