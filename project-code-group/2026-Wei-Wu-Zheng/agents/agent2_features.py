"""
Agent 2: Feature Extraction Agent
Responsible for calculating heart rate and HRV metrics from cleaned ECG signal
"""

import numpy as np
from scipy import signal as scipy_signal

class FeatureExtractionAgent:
    """Agent 2: Responsible for calculating physiological features"""

    def __init__(self):
        self.name = "Feature Extraction Agent"
        self.status = "Standby"
        self.extraction_log = []

    def extract_features(self, cleaned_signal, sampling_rate=250):
        """
        Extract features from cleaned ECG signal

        Steps:
        1. Detect R-peaks (QRS complex)
        2. Calculate RR intervals
        3. Calculate Heart Rate (HR)
        4. Calculate HRV metrics (SDNN, RMSSD)

        Args:
            cleaned_signal: Cleaned ECG signal
            sampling_rate: Sampling rate (Hz)

        Returns:
            features: dict containing HR, HRV metrics
        """
        self.status = "Analyzing..."
        self.extraction_log = []

        try:
            # 1. Detect R-peaks
            r_peaks = self._detect_r_peaks(cleaned_signal, sampling_rate)
            self.extraction_log.append(f"[OK] Detected {len(r_peaks)} R-peaks")

            if len(r_peaks) < 5:
                self.status = "Warning: Too few R-peaks"
                self.extraction_log.append("[FAIL] Insufficient R-peaks for HRV calculation")
                return self._get_default_features()

            # 2. Calculate RR intervals (milliseconds)
            rr_intervals = np.diff(r_peaks) / sampling_rate * 1000
            self.extraction_log.append(f"[OK] Calculated {len(rr_intervals)} RR intervals")

            # 3. Filter abnormal RR intervals (possible artifacts)
            rr_intervals = self._filter_rr_intervals(rr_intervals)
            self.extraction_log.append(f"[OK] {len(rr_intervals)} valid RR intervals after filtering")

            if len(rr_intervals) < 5:
                self.status = "Warning: Too few valid RR intervals"
                return self._get_default_features()

            # 4. Calculate features
            features = {
                "heart_rate": self._calculate_hr(rr_intervals),
                "hrv_sdnn": self._calculate_sdnn(rr_intervals),
                "hrv_rmssd": self._calculate_rmssd(rr_intervals),
                "rr_intervals": rr_intervals.tolist(),
                "num_beats": len(r_peaks),
                "mean_rr": round(np.mean(rr_intervals), 2),
                "min_hr": round(60000 / np.max(rr_intervals), 2),
                "max_hr": round(60000 / np.min(rr_intervals), 2)
            }

            self.extraction_log.append("[OK] All features calculated")
            self.status = "Done"

            return features

        except Exception as e:
            self.status = f"Error: {str(e)}"
            self.extraction_log.append(f"[FAIL] Feature extraction failed: {str(e)}")
            return self._get_default_features()

    def _detect_r_peaks(self, signal_data, sampling_rate):
        """
        Detect R-peaks using simple peak detection

        Args:
            signal_data: ECG signal
            sampling_rate: Sampling rate

        Returns:
            r_peaks: R-peak position indices (numpy array)
        """
        # Set minimum distance (assuming HR won't exceed 180 bpm)
        min_distance = int(sampling_rate * 60 / 180)  # ~0.33 seconds

        # Set threshold (using multiple of signal std)
        threshold = np.std(signal_data) * 0.5

        # Use scipy's find_peaks
        peaks, properties = scipy_signal.find_peaks(
            signal_data,
            height=threshold,
            distance=min_distance
        )

        return peaks

    def _filter_rr_intervals(self, rr_intervals, lower_limit=300, upper_limit=2000):
        """
        Filter abnormal RR intervals

        Args:
            rr_intervals: RR interval array
            lower_limit: Lower limit (ms), corresponds to ~200 bpm
            upper_limit: Upper limit (ms), corresponds to ~30 bpm

        Returns:
            filtered_rr: Filtered RR intervals
        """
        # Remove values outside physiological range
        mask = (rr_intervals >= lower_limit) & (rr_intervals <= upper_limit)
        filtered_rr = rr_intervals[mask]

        # If too many filtered out, relax constraints
        if len(filtered_rr) < len(rr_intervals) * 0.5:
            # Use looser range
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            mask = np.abs(rr_intervals - mean_rr) < 3 * std_rr
            filtered_rr = rr_intervals[mask]

        return filtered_rr

    def _calculate_hr(self, rr_intervals):
        """Calculate average heart rate (bpm)"""
        if len(rr_intervals) == 0:
            return 0
        mean_rr = np.mean(rr_intervals)  # milliseconds
        hr = 60000 / mean_rr  # convert to bpm
        return round(hr, 2)

    def _calculate_sdnn(self, rr_intervals):
        """
        Calculate SDNN (Standard Deviation of NN intervals)
        NN intervals = Normal-to-Normal intervals
        """
        if len(rr_intervals) == 0:
            return 0
        return round(np.std(rr_intervals, ddof=1), 2)

    def _calculate_rmssd(self, rr_intervals):
        """
        Calculate RMSSD (Root Mean Square of Successive Differences)
        Square root of mean of squared successive RR interval differences
        """
        if len(rr_intervals) < 2:
            return 0
        diff_rr = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        return round(rmssd, 2)

    def _get_default_features(self):
        """Return default features (when extraction fails)"""
        return {
            "heart_rate": 0,
            "hrv_sdnn": 0,
            "hrv_rmssd": 0,
            "rr_intervals": [],
            "num_beats": 0,
            "mean_rr": 0,
            "min_hr": 0,
            "max_hr": 0
        }

    def get_log(self):
        """Get extraction log"""
        return self.extraction_log

    def interpret_features(self, features):
        """
        Interpret feature data

        Args:
            features: Feature dictionary

        Returns:
            interpretations: List of interpretation results
        """
        interpretations = []

        hr = features['heart_rate']
        sdnn = features['hrv_sdnn']

        # Heart rate interpretation
        if hr == 0:
            interpretations.append("[WARN] Unable to calculate heart rate")
        elif hr < 60:
            interpretations.append(f"[LOW HR] Heart rate low ({hr} bpm), may indicate relaxation or fatigue")
        elif 60 <= hr <= 100:
            interpretations.append(f"[NORMAL] Heart rate normal ({hr} bpm)")
        else:
            interpretations.append(f"[HIGH HR] Heart rate elevated ({hr} bpm), may indicate alertness or stress")

        # HRV interpretation
        if sdnn == 0:
            interpretations.append("[WARN] Unable to calculate HRV")
        elif sdnn > 80:
            interpretations.append(f"[HIGH HRV] HRV elevated (SDNN={sdnn} ms), parasympathetic active, possible fatigue")
        elif sdnn >= 50:
            interpretations.append(f"[NORMAL] HRV normal (SDNN={sdnn} ms)")
        else:
            interpretations.append(f"[LOW HRV] HRV low (SDNN={sdnn} ms), possible stress or alertness")

        return interpretations

# Test program
if __name__ == "__main__":
    print("=" * 50)
    print("Agent 2 Test - Feature Extraction Agent")
    print("=" * 50)
    print()

    # Create test signal (simulating cleaned ECG)
    # Generate simple periodic signal to simulate heartbeat
    sampling_rate = 250
    duration = 30  # seconds
    t = np.linspace(0, duration, duration * sampling_rate)

    # Simulate 70 bpm heart rate signal
    heart_rate = 70
    beat_times = np.arange(0, duration, 60/heart_rate)

    # Create signal
    test_signal = np.zeros(len(t))
    for beat_time in beat_times:
        beat_idx = int(beat_time * sampling_rate)
        if beat_idx < len(test_signal):
            # Simulate QRS complex
            test_signal[beat_idx] = 1.0
            if beat_idx + 1 < len(test_signal):
                test_signal[beat_idx + 1] = 0.3

    # Add some variation (HRV)
    test_signal += np.random.randn(len(test_signal)) * 0.05

    # Test Agent
    agent2 = FeatureExtractionAgent()
    print(f"Agent status: {agent2.status}")
    print()

    print("Starting feature extraction...")
    features = agent2.extract_features(test_signal, sampling_rate)

    print("\nExtraction log:")
    for log in agent2.get_log():
        print(f"  {log}")

    print(f"\nFinal status: {agent2.status}")

    print("\nExtracted features:")
    for key, value in features.items():
        if key != "rr_intervals":  # Don't display full RR interval list
            print(f"  {key}: {value}")

    print("\nFeature interpretation:")
    interpretations = agent2.interpret_features(features)
    for interp in interpretations:
        print(f"  {interp}")

    print("\n" + "=" * 50)
    print("[OK] Agent 2 test complete!")
    print("=" * 50)
