#added GSR and microphone
import profile

import profile

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SleepKalmanFilter:
    def __init__(self, q=0.01, r=2.0, p=1.0, x=5.0):
        self.q, self.r, self.p, self.x = q, r, p, x
    def update(self, z):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x

def generate_realistic_sleep_dataset_v4():
    num_users = 30
    days_per_user = 15
    all_records = []
    
    user_profiles = {}
    for uid in range(1, num_users + 1):
        age = np.random.randint(18, 75)
        base_rhr = np.random.uniform(55, 75)
        base_rmssd = max(15, 100 - age + np.random.normal(0, 10))
        # Base GSR (SCL) usually ranges from 2 to 20 uS
        base_gsr = np.random.uniform(2, 8) 
        user_profiles[uid] = {
            'age': age, 
            'base_hr': base_rhr, 
            'base_rmssd': base_rmssd,
            'base_gsr': base_gsr
        }

    for user_id, profile in user_profiles.items():
        # Initialize Kalman Filter for GSR to ensure smooth transitions
        kf_gsr = SleepKalmanFilter(x=profile['base_gsr'])
        
        for day in range(1, days_per_user + 1):
            base_ambient_temp = np.random.uniform(18, 26)
            base_humidity = np.random.uniform(35, 65)
            
            p_bad = 0.4 if (base_ambient_temp > 24 or base_ambient_temp < 18) else 0.2
            p_rem_avg = (1.0 - p_bad) / 2
            day_type = np.random.choice(['Good', 'Avg', 'Bad'], p=[p_rem_avg, p_rem_avg, p_bad])
            
            start_time = datetime(2026, 3, 1, np.random.randint(22, 25)%24, np.random.randint(0,60)) + timedelta(days=day)
            total_mins = np.random.randint(300, 540) if day_type != 'Bad' else np.random.randint(240, 400)
            
            current_stage = "Awake"
            mins_in_stage = 0

            for t in range(total_mins):
                timestamp = start_time + timedelta(minutes=t)
                time_progression = t / total_mins
                
                # --- 1. Light Exposure Logic ---
                if t < 8:  
                    light_lux = np.random.uniform(150, 500) 
                elif 8 <= t < 15:
                    light_lux = np.random.uniform(2, 10)
                elif time_progression > 0.9:
                    light_lux = np.random.uniform(100, 400)
                elif current_stage == "Awake":
                    prob = np.random.random()
                    if prob < 0.1: light_lux = np.random.uniform(50, 150)
                    elif prob < 0.4: light_lux = np.random.uniform(5, 25)
                    else: light_lux = np.random.uniform(0, 1)
                else:
                    light_lux = np.random.uniform(0, 0.5)

                # --- 2. Sleep Architecture ---
                if t < 15: 
                    current_stage = "Awake"
                elif mins_in_stage > np.random.randint(10, 100): 
                    weights = [0.05, 0.45, 0.45, 0.05] if time_progression < 0.4 else [0.10, 0.60, 0.05, 0.25]
                    if day_type == 'Bad': weights = [0.30, 0.55, 0.05, 0.10]
                    current_stage = np.random.choice(["Awake", "Light", "Deep", "REM"], p=weights)
                    mins_in_stage = 0
                
                mins_in_stage += 1

                # --- 3. Physiology & New GSR Logic ---
    
                if current_stage == "Deep":
                
                    # Variable HR instead of fixed value
                    hr = profile['base_hr'] - np.random.uniform(1, 5)
                
                    # HRV overlaps with Light sleep
                    hrv = np.random.uniform(50, 95)
                
                    svm = np.random.uniform(0.01, 0.04)
                
                    # Person-specific variability
                    hr_variance_multiplier = np.random.uniform(1.2, 1.8)
                    hr *= hr_variance_multiplier
                
                    target_gsr = profile['base_gsr'] * 0.8
                
                
                elif current_stage == "REM":
                
                    hr = profile['base_hr'] + np.random.uniform(3, 8)
                
                    # Some overlap with Light
                    hrv = np.random.uniform(25, 70)
                
                    svm = np.random.uniform(0.01, 0.06)
                
                    target_gsr = profile['base_gsr'] * 1.5
                

                elif current_stage == "Awake":
                
                    hr = profile['base_hr'] + np.random.uniform(10, 20)

                    hrv = profile['base_rmssd'] - np.random.uniform(10, 25)

                    svm = np.random.uniform(0.8, 5.0)

                    target_gsr = profile['base_gsr'] * 2.0


                else:  # Light
                
                    hr = profile['base_hr'] + np.random.uniform(-2, 3)

                    # Overlaps with Deep and REM
                    hrv = np.random.uniform(40, 80)

                    svm = np.random.uniform(0.02, 0.1)

                    target_gsr = profile['base_gsr']


# --- Transition / Misclassification Logic (5-10%) ---

                if np.random.rand() < 0.08:
                
                    # Slightly distort physiology to mimic transition states
                    hr += np.random.uniform(-3, 3)
                    hrv += np.random.uniform(-10, 10)

                # Apply Kalman filter to GSR for realistic drift
                filt_gsr = kf_gsr.update(target_gsr + np.random.normal(0, 0.2))

                # --- 4. SpO2, Resp & Microphone (Audio) ---
                spo2, resp_rate = np.random.uniform(96, 99.5), np.random.uniform(12, 16)
                apnea = 1 if (current_stage != "Awake" and np.random.random() < 0.003) else 0
                
                # Audio Logic: Snoring or movement
                if apnea: 
                    spo2 -= 8; resp_rate -= 8
                    audio_db = np.random.uniform(65, 85) # Loud gasping/snoring
                elif current_stage == "Awake":
                    audio_db = np.random.uniform(35, 50) # Bed rustling
                else:
                    audio_db = np.random.uniform(20, 32) # Ambient room floor

                # --- 5. Skin Temperature ---
                if current_stage == "Awake":
                    body_temp = 36.2 + np.random.normal(0, 0.1)
                else:
                    body_temp = 36.7 + 0.3 * np.sin(np.pi * time_progression) + np.random.normal(0, 0.05)

                all_records.append([
                    user_id, profile['age'], day, timestamp.strftime("%Y-%m-%d %H:%M"), current_stage,
                    round(hr, 2), round(hrv, 2), round(np.random.uniform(1, 10), 2), round(spo2, 1), 
                    round(resp_rate, 1), apnea, round(svm, 3), round(body_temp, 2),
                    round(base_ambient_temp + np.random.normal(0, 0.1), 2), 
                    round(base_humidity + np.random.normal(0, 0.5), 2),
                    round(light_lux, 2),
                    round(filt_gsr, 3), # New Column: GSR
                    round(audio_db, 1)   # New Column: Microphone
                ])

    cols = ['User_ID', 'Age', 'Day', 'Timestamp', 'Stage', 'HR', 'RMSSD', 'HR_Stability', 
            'SpO2', 'Resp_Rate', 'Apnea_Event', 'SVM', 'Body_Temp', 'Ambient_Temp', 'Humidity', 
            'Light_Lux', 'GSR_uS', 'Audio_dB']
    
    return pd.DataFrame(all_records, columns=cols)

# Execution
df_v4 = generate_realistic_sleep_dataset_v4()
df_v4.to_csv("test_new.csv", index=False)
print(f"Generated dataset with {len(df_v4)} rows.")