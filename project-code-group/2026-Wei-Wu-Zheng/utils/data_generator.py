"""
測試資料生成器
生成模擬的 ECG 訊號用於測試系統
"""

import numpy as np
import pandas as pd
from pathlib import Path

class ECGDataGenerator:
    """生成模擬 ECG 訊號"""
    
    def __init__(self, sampling_rate=250):
        """
        初始化生成器
        
        參數:
            sampling_rate: 採樣率 (Hz)
        """
        self.sampling_rate = sampling_rate
    
    def generate_ecg(self, duration=30, heart_rate=70, noise_level=0.05, drowsy=False):
        """
        生成模擬 ECG 訊號
        
        參數:
            duration: 持續時間（秒）
            heart_rate: 平均心率 (bpm)
            noise_level: 噪音程度
            drowsy: 是否為疲勞狀態
            
        返回:
            numpy array: ECG 訊號
        """
        n_samples = duration * self.sampling_rate
        t = np.linspace(0, duration, n_samples)
        
        # 如果是疲勞狀態，調整參數
        if drowsy:
            heart_rate = 58  # 疲勞時心率較低
            hrv_variation = 0.15  # 較大的變異（HRV 較高）
            noise_level = noise_level * 1.5  # 稍微多一點噪音
        else:
            hrv_variation = 0.05  # 較小的變異（HRV 較低）
        
        # 初始化訊號
        ecg = np.zeros(n_samples)
        
        # 計算心跳間隔（以樣本點為單位）
        beat_interval = self.sampling_rate * (60 / heart_rate)
        
        # 生成每個心跳
        current_pos = 0
        beat_count = 0
        
        while current_pos < n_samples:
            # 加入 HRV：每次心跳間隔有變化
            interval_variation = np.random.randn() * hrv_variation * beat_interval
            current_interval = beat_interval + interval_variation
            
            # 確保間隔合理
            current_interval = max(beat_interval * 0.7, 
                                 min(current_interval, beat_interval * 1.3))
            
            # 生成 QRS 波群
            peak_pos = int(current_pos)
            
            if peak_pos < n_samples:
                # R 波峰（主波峰）
                ecg[peak_pos] = 1.0
                
                # Q 波（R 波前的小下降）
                if peak_pos > 2:
                    ecg[peak_pos - 2:peak_pos] = [-0.1, -0.05]
                
                # S 波（R 波後的小下降）
                if peak_pos + 3 < n_samples:
                    ecg[peak_pos + 1:peak_pos + 4] = [-0.15, -0.1, -0.05]
                
                # T 波（較慢的正波）
                if peak_pos + 15 < n_samples:
                    t_wave_len = 15
                    t_wave = 0.3 * np.exp(-((np.arange(t_wave_len) - 7) ** 2) / 20)
                    ecg[peak_pos + 5:peak_pos + 5 + t_wave_len] += t_wave
            
            # 移動到下一個心跳
            current_pos += current_interval
            beat_count += 1
        
        # 加入基線漂移（模擬呼吸）
        baseline_freq = 0.25  # Hz (呼吸頻率)
        baseline = 0.1 * np.sin(2 * np.pi * baseline_freq * t)
        ecg += baseline
        
        # 加入隨機噪音
        noise = np.random.randn(n_samples) * noise_level
        ecg += noise
        
        # 加入偶爾的運動偽影（模擬頭部移動、說話）
        if np.random.rand() > 0.7:  # 30% 機率有偽影
            artifact_start = int(np.random.rand() * n_samples * 0.5)
            artifact_len = int(self.sampling_rate * 2)  # 2 秒的偽影
            if artifact_start + artifact_len < n_samples:
                artifact = np.random.randn(artifact_len) * noise_level * 3
                ecg[artifact_start:artifact_start + artifact_len] += artifact
        
        return ecg
    
    def save_to_csv(self, ecg, filename):
        """
        將 ECG 訊號存成 CSV 檔案
        
        參數:
            ecg: ECG 訊號 array
            filename: 檔案名稱
        """
        df = pd.DataFrame({'ECG': ecg})
        df.to_csv(filename, index=False)
        print(f"[OK] Saved: {filename}")
        print(f"     Samples: {len(ecg)}")
        print(f"     Duration: {len(ecg) / self.sampling_rate:.1f} sec")
        print()

def main():
    """Main function: Generate test data"""
    print("=" * 50)
    print("ECG Test Data Generator")
    print("=" * 50)
    print()

    # Create generator
    generator = ECGDataGenerator(sampling_rate=250)

    # Ensure data folder exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate normal state ECG (30 sec)
    print("Generating normal state ECG signal...")
    ecg_normal = generator.generate_ecg(
        duration=30,
        heart_rate=75,
        noise_level=0.05,
        drowsy=False
    )
    generator.save_to_csv(ecg_normal, "data/ecg_normal.csv")

    # Generate drowsy state ECG (30 sec)
    print("Generating drowsy state ECG signal...")
    ecg_drowsy = generator.generate_ecg(
        duration=30,
        heart_rate=58,
        noise_level=0.08,
        drowsy=True
    )
    generator.save_to_csv(ecg_drowsy, "data/ecg_drowsy.csv")

    # Generate longer test data (60 sec)
    print("Generating long duration test data...")
    ecg_long = generator.generate_ecg(
        duration=60,
        heart_rate=65,
        noise_level=0.06,
        drowsy=True
    )
    generator.save_to_csv(ecg_long, "data/ecg_long_drowsy.csv")

    print("=" * 50)
    print("[OK] All test data generated!")
    print("=" * 50)
    print()
    print("You can now run the main program:")
    print("  streamlit run app.py")
    print()

if __name__ == "__main__":
    main()
