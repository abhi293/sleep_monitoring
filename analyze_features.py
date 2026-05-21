import pandas as pd
import numpy as np

df = pd.read_csv('realistic_sleep_dataset_v4.csv')

# Key cardiac features by stage
features_to_analyze = ['HR', 'RMSSD', 'HR_Stability', 'SVM', 'SpO2', 'Resp_Rate', 'Apnea_Event']

print("=" * 80)
print("FEATURE STATISTICS BY SLEEP STAGE (Why Models Perform So Well)")
print("=" * 80)

for stage in ['Awake', 'Light', 'Deep', 'REM']:
    stage_data = df[df['Stage'] == stage][features_to_analyze]
    print(f"\n{stage.upper()}:")
    print(stage_data.describe().round(2))

# Calculate feature separability
print("\n" + "=" * 80)
print("FEATURE SEPARABILITY ANALYSIS")
print("=" * 80)
for feature in features_to_analyze[:5]:  # Check first 5 features
    awake_mean = df[df['Stage'] == 'Awake'][feature].mean()
    light_mean = df[df['Stage'] == 'Light'][feature].mean()
    deep_mean = df[df['Stage'] == 'Deep'][feature].mean()
    rem_mean = df[df['Stage'] == 'REM'][feature].mean()
    
    awake_std = df[df['Stage'] == 'Awake'][feature].std()
    
    # If standard deviations are small relative to differences, features are well-separated
    print(f"\n{feature}:")
    print(f"  Awake: {awake_mean:.2f} ± {awake_std:.2f}")
    print(f"  Light: {light_mean:.2f}")
    print(f"  Deep:  {deep_mean:.2f}")
    print(f"  REM:   {rem_mean:.2f}")
    print(f"  CV (Coeff of Variation): {(awake_std/awake_mean):.3f}")  # Variability metric
