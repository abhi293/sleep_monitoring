# Sleep Intelligence System — Complete Documentation

> A comprehensive guide to the dataset attributes, their real-world relevance in sleep monitoring, and the full mathematical pipeline from raw sensor data to final Sleep Quality Score.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Dataset Attributes — Definitions, Formulas & Realism](#2-dataset-attributes--definitions-formulas--realism)
   - [Identification & Temporal Columns](#21-identification--temporal-columns)
   - [Cardiac Signals](#22-cardiac-signals)
   - [Respiratory Signals](#23-respiratory-signals)
   - [Movement & Actigraphy](#24-movement--actigraphy)
   - [Thermal Signals](#25-thermal-signals)
   - [Environmental Signals](#26-environmental-signals)
   - [Sleep Stage Label](#27-sleep-stage-label-ground-truth)
3. [How the Dataset Reflects Real-World Sleep Monitoring](#3-how-the-dataset-reflects-real-world-sleep-monitoring)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Stage 1 — Data Loading & Temporal Feature Engineering](#5-stage-1--data-loading--temporal-feature-engineering)
6. [Stage 2 — User-Stratified Splitting & Scaling](#6-stage-2--user-stratified-splitting--scaling)
7. [Stage 3 — Sliding Window Segmentation](#7-stage-3--sliding-window-segmentation)
8. [Stage 4 — Class Imbalance Handling](#8-stage-4--class-imbalance-handling)
9. [Stage 5 — Hybrid CNN–GRU–LSTM Model Architecture](#9-stage-5--hybrid-cnngru-lstm-model-architecture)
10. [Stage 6 — Loss Function (Sparse Focal Loss)](#10-stage-6--loss-function-sparse-focal-loss)
11. [Stage 7 — MOPSO Hyperparameter Optimisation](#11-stage-7--mopso-hyperparameter-optimisation)
12. [Stage 8 — Training & Optimisation](#12-stage-8--training--optimisation)
13. [Stage 9 — Evaluation Metrics](#13-stage-9--evaluation-metrics)
14. [Stage 10 — Final Sleep Quality Score](#14-stage-10--final-sleep-quality-score)
15. [End-to-End Formula Summary](#15-end-to-end-formula-summary)

---

## 1. Dataset Overview

The dataset (`realistic_sleep_dataset_v4.csv`) is a synthetically generated, physiologically-grounded sleep monitoring dataset designed to emulate what a real multi-sensor wearable + bedroom environment sensor system would capture.

| Property | Value |
|---|---|
| **Users** | 30 unique individuals |
| **Nights per user** | 15 |
| **Granularity** | 1-minute resolution (per-minute readings) |
| **Total rows** | ~176,000 minute-level records |
| **Features** | 16 columns (11 sensor signals + 5 metadata) |
| **Sleep stages** | 4 classes: Awake, Light, Deep, REM |
| **Recording duration** | 240–540 minutes per night (4–9 hours) |

Each row represents **one minute** of a user's sleep session, capturing simultaneous readings from cardiac, respiratory, motion, thermal, and environmental sensors — exactly the sensor fusion approach used in modern sleep monitoring research and consumer devices (e.g., Apple Watch, Oura Ring, Withings Sleep Analyzer).

---

## 2. Dataset Attributes — Definitions, Formulas & Realism

### 2.1 Identification & Temporal Columns

| Column | Type | Description |
|---|---|---|
| `User_ID` | Integer | Unique identifier for each participant (1–30) |
| `Age` | Integer | Participant age in years (range: 18–75) |
| `Day` | Integer | Night number within the study (1–15) |
| `Timestamp` | Datetime | Absolute timestamp at minute resolution (`YYYY-MM-DD HH:MM`) |

**Realism**: Each user has a unique physiological profile (baseline heart rate, baseline HRV), which is consistent across all their nights. Age directly influences baseline HRV — older users have lower baseline RMSSD, matching the well-documented age-related decline in heart rate variability in clinical literature.

$$\text{base\_rmssd} = \max\left(15,\; 100 - \text{Age} + \mathcal{N}(0, 10)\right)$$

$$\text{base\_hr} \sim \mathcal{U}(55, 75) \text{ bpm}$$

---

### 2.2 Cardiac Signals

#### **HR — Heart Rate (bpm)**

The instantaneous heart rate, varying by sleep stage:

| Stage | Formula | Typical Range |
|---|---|---|
| Deep | $\text{HR} = \text{base\_hr} - 6$ | 49–69 bpm |
| Light | $\text{HR} = \text{base\_hr}$ | 55–75 bpm |
| REM | $\text{HR} = \text{base\_hr} + 5$ | 60–80 bpm |
| Awake | $\text{HR} = \text{base\_hr} + 15$ | 70–90 bpm |

**Impact on sleep monitoring**: Heart rate is the most fundamental cardiac biomarker. During deep sleep, parasympathetic dominance lowers HR significantly. During REM, sympathetic activation causes HR to rise and become irregular. Wearables like Fitbit and Apple Watch primarily rely on HR patterns to distinguish sleep stages.

**Realism**: Clinical polysomnography (PSG) studies confirm HR decreases 8–15% below resting during deep NREM and increases 5–10% during REM. The dataset faithfully follows this trajectory.

---

#### **RMSSD — Root Mean Square of Successive RR Differences (ms)**

RMSSD is the gold-standard time-domain measure of heart rate variability (HRV), reflecting parasympathetic (vagal) tone.

$$\text{RMSSD} = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N-1}(RR_{i+1} - RR_i)^2}$$

Where $RR_i$ is the interval between successive heartbeats.

| Stage | Formula | Interpretation |
|---|---|---|
| Deep | $\text{RMSSD} = \text{base\_rmssd} + 15$ | High vagal tone → restorative sleep |
| Light | $\text{RMSSD} = \text{base\_rmssd}$ | Baseline parasympathetic activity |
| REM | $\text{RMSSD} = \text{base\_rmssd} - 12$ | Reduced HRV, sympathetic surges |
| Awake | $\text{RMSSD} = \text{base\_rmssd} - 20$ | Lowest HRV, active sympathetic system |

**Impact on sleep monitoring**: RMSSD is the single most informative feature for distinguishing deep sleep (high HRV = parasympathetic recovery) from REM/Awake states (low HRV = sympathetic activation). Used by Oura Ring, Whoop, and clinical research for "recovery" scoring.

**Realism**: Clinical data shows RMSSD increases 20–50% during deep NREM relative to waking baseline and decreases during REM. The age-dependent baseline ensures older subjects naturally exhibit lower HRV, matching epidemiological data.

---

#### **HR_Stability — Heart Rate Stability Index**

A measure of beat-to-beat heart rate variability/steadiness within each minute. Low values indicate a very stable, metronomic heart rate; high values indicate erratic rhythm.

| Stage | Range | Interpretation |
|---|---|---|
| Deep | 0.1–0.8 | Extremely stable (parasympathetic dominance) |
| Light | 1.0–3.5 | Moderately stable |
| REM | 4.0–9.0 | Erratic (sympathetic bursts, dream activity) |
| Awake | 5.0–15.0 | Most variable (voluntary movements, arousal) |

**Impact on sleep monitoring**: HR stability differentiates REM (erratic but low-movement) from Awake (erratic with high movement). It acts as a complementary signal to HR and RMSSD — when all three agree, stage classification confidence is highest.

**Realism**: In polysomnography, REM sleep is characterised by irregular heart rhythm ("saw-tooth" waves on ECG), which is directly captured by this metric.

---

### 2.3 Respiratory Signals

#### **Resp_Rate — Respiratory Rate (breaths/min)**

| Stage | Range | Notes |
|---|---|---|
| Deep/Light/Awake | $\mathcal{U}(12, 16)$ | Normal breathing |
| REM | $\mathcal{U}(15, 22)$ | Elevated and irregular |
| During Apnea | Reduced by 8 | Simulates cessation of breathing |

**Impact on sleep monitoring**: Respiratory rate elevation during REM reflects irregular breathing tied to dream activity. This is a critical signal for REM detection and apnea screening in devices like ResMed and Withings.

**Realism**: Normal adult respiratory rate is 12–20 breaths/min. During REM, it becomes faster and more irregular (confirmed by PSG). The dataset accurately reflects this.

---

#### **Apnea_Event — Binary Apnea Indicator**

$$\text{Apnea\_Event} = \begin{cases} 1 & \text{if stage} \neq \text{Awake and } p < 0.003 \\ 0 & \text{otherwise} \end{cases}$$

When an apnea event fires:
- $\text{SpO2} \leftarrow \text{SpO2} - 8$ (oxygen desaturation)  
- $\text{Resp\_Rate} \leftarrow \text{Resp\_Rate} - 8$ (breathing pause)

**Impact on sleep monitoring**: Obstructive Sleep Apnea (OSA) is one of the most common sleep disorders (affects ~26% of adults aged 30–70). Real-time apnea detection through SpO2 dips and respiratory disturbances is a primary clinical application of sleep monitoring.

**Realism**: The 0.3% per-minute probability produces approximately 1–3 apnea events per night for most users, matching the mild apnea profile (AHI < 5 events/hour). The coupled SpO2 desaturation of ~8% during events matches clinical observation of moderate-severity apnea episodes.

---

#### **SpO2 — Blood Oxygen Saturation (%)**

$$\text{SpO2} \sim \mathcal{U}(96, 99.5)\%$$

During apnea events:
$$\text{SpO2}_{\text{apnea}} = \text{SpO2} - 8$$

This can push SpO2 down to ~88–91.5%, which is clinically significant.

**Impact on sleep monitoring**: SpO2 below 90% is a medical red flag. Consumer pulse oximeters (e.g., Apple Watch, Garmin) track overnight SpO2 to screen for sleep apnea. Sustained desaturations correlate with fragmented sleep and cardiovascular risk.

**Realism**: Healthy adults maintain SpO2 of 95–99% during sleep. Intermittent drops to 85–92% during apnea events are well-documented in PSG and are the primary diagnostic criterion for OSA severity grading.

---

### 2.4 Movement & Actigraphy

#### **SVM — Signal Vector Magnitude (g)**

SVM is the Euclidean norm of the 3-axis accelerometer signal, representing total body movement magnitude:

$$\text{SVM} = \sqrt{a_x^2 + a_y^2 + a_z^2}$$

| Stage | Range (g) | Interpretation |
|---|---|---|
| Deep | 0.01–0.04 | Muscle atonia (sleep paralysis) |
| REM | 0.01–0.06 | Slight twitches despite atonia |
| Light | 0.02–0.10 (occasional 0.3–1.2) | Minor positional shifts |
| Awake | 0.80–5.00 | Voluntary movement, tossing/turning |

**Impact on sleep monitoring**: Actigraphy (movement-based sleep/wake discrimination) is the foundation of consumer sleep trackers. SVM cleanly separates Awake (high movement) from asleep (low movement). Differentiating between sleep stages using movement alone is poor, which is why multi-sensor fusion is necessary.

**Realism**: During deep sleep and REM, skeletal muscles are largely paralysed (atonia), so SVM is near-zero. During light sleep, occasional micro-arousals cause brief movement spikes (~5% of light sleep minutes). The dataset faithfully models these distributions.

---

### 2.5 Thermal Signals

#### **Body_Temp — Skin Temperature (°C)**

Body temperature follows the circadian thermoregulatory cycle. During sleep, the body dissipates core heat through peripheral vasodilation, causing skin temperature to rise slightly:

$$\text{Body\_Temp} = \begin{cases} 36.2 + \mathcal{N}(0, 0.1) & \text{if Awake} \\ 36.7 + 0.3 \cdot \sin(\pi \cdot t_{\text{prog}}) + \mathcal{N}(0, 0.05) & \text{if Asleep} \end{cases}$$

Where $t_{\text{prog}} = \frac{t}{T_{\text{total}}}$ is the fractional time progression through the night (0 to 1).

**Impact on sleep monitoring**: Skin temperature rise is a reliable marker of sleep onset. The sinusoidal pattern (peaking mid-sleep, declining toward morning) mirrors the core body temperature's circadian nadir. Devices like the Oura Ring use temperature as a key feature for sleep stage scoring and illness/cycle detection.

**Realism**: Clinical thermography confirms distal skin temperature rises 0.3–0.8°C during sleep due to vasodilation. The circadian sinusoidal pattern is well-established in chronobiology research.

---

#### **Ambient_Temp — Room Temperature (°C)**

$$\text{Ambient\_Temp} = \text{base\_ambient} + \mathcal{N}(0, 0.1)$$

Where $\text{base\_ambient} \sim \mathcal{U}(18, 26)$°C per night.

**Impact on sleep monitoring**: Room temperature significantly affects sleep quality. Temperatures above 24°C or below 18°C increase the probability of fragmented sleep (the dataset increases `p_bad` from 0.2 to 0.4 in non-ideal temperatures). The optimal sleep temperature is 18–22°C.

**Realism**: The dataset directly models the environment–sleep quality interaction: hot/cold rooms → more awakenings and less deep sleep. This matches the National Sleep Foundation's recommendations and published sleep hygiene research.

---

### 2.6 Environmental Signals

#### **Humidity — Relative Humidity (%)**

$$\text{Humidity} = \text{base\_humidity} + \mathcal{N}(0, 0.5)$$

Where $\text{base\_humidity} \sim \mathcal{U}(35, 65)\%$ per night.

**Impact on sleep monitoring**: Humidity outside the 30–50% range can affect respiratory comfort, increase snoring, and exacerbate sleep apnea. It serves as a contextual environmental feature.

**Realism**: Typical indoor humidity varies between 30–65%. The near-constant per-night value with minor fluctuation matches real bedroom sensors (humidity changes slowly over hours, not minutes).

---

#### **Light_Lux — Ambient Light Level (lux)**

Light follows a physically driven event-based model:

| Time Period | Condition | Lux Range | Rationale |
|---|---|---|---|
| First 8 min | Lights still on | 150–500 | User settling in |
| 8–15 min | Lights off / dim | 2–10 | Sleep onset routine |
| Mid-sleep (asleep) | Pitch dark | 0–0.5 | Normal sleeping room |
| Mid-sleep (awake) | Variable | 0–150 | Phone use (10%), big light (10%), dark (60%) |
| Last 10% of night | Morning/sunrise | 100–400 | Dawn simulation or sunrise |

**Impact on sleep monitoring**: Light exposure directly regulates circadian rhythm through melanopsin-containing retinal ganglion cells. Evening light exposure suppresses melatonin and delays sleep onset. The dataset's event-based model captures realistic light patterns that a bedside or wearable light sensor would detect.

**Realism**: The conditional probability model (10% chance of turning on a bright light during an awakening, 30% chance of phone use) reflects real human behaviour. The dawn light ramp is consistent with sunrise or alarm-lamp patterns. Pitch-dark sleep periods (0–0.5 lux) match properly prepared bedrooms.

---

### 2.7 Sleep Stage Label (Ground Truth)

| Stage | Encoding | Description |
|---|---|---|
| **Awake** | 0 | Eyes open, conscious, voluntary movement |
| **Light** (N1+N2) | 1 | Drowsy / light NREM sleep (theta waves, K-complexes) |
| **Deep** (N3) | 2 | Slow-wave sleep — physically restorative (delta waves) |
| **REM** | 3 | Rapid eye movement — cognitively restorative (dreaming) |

**Sleep architecture model**:
- First 15 minutes: enforced Awake (realistic sleep latency)
- First half of night (t < 40%): biased toward Deep sleep — weights: `[Awake:0.05, Light:0.45, Deep:0.45, REM:0.05]`
- Second half of night (t ≥ 40%): biased toward REM and Light — weights: `[Awake:0.10, Light:0.60, Deep:0.05, REM:0.25]`
- Bad nights: heavily biased toward Awake/Light — weights: `[Awake:0.30, Light:0.55, Deep:0.05, REM:0.10]`

**Realism**: This precisely mirrors the clinically observed sleep architecture: deep sleep concentrates in the first third of the night (driven by homeostatic sleep pressure), while REM cycles lengthen in the second half (driven by circadian rhythm). Bad night profiles (more fragmentation, less deep/REM) simulate the effects of stress, caffeine, or environmental disruption.

---

## 3. How the Dataset Reflects Real-World Sleep Monitoring

| Real-World Aspect | Dataset Implementation |
|---|---|
| **Individual differences** | Unique (Age, base_hr, base_rmssd) per user; age-dependent HRV decline |
| **Night-to-night variability** | Random day type (Good/Avg/Bad), variable bedtime (10 PM–1 AM), variable duration (4–9 hrs) |
| **Sleep architecture** | Physiologically accurate stage transition probabilities that shift across the night |
| **Sensor noise** | Gaussian noise on all signals ($\sigma$ varies by feature) |
| **Apnea modelling** | Stochastic events with coupled SpO2 and respiratory effects |
| **Circadian thermoregulation** | Sinusoidal skin temperature with stage-dependent baseline |
| **Environmental influence** | Temperature extremes increase probability of bad sleep nights |
| **Light ecology** | Event-driven model (lights on → lights off → dark → phone use → dawn) |
| **Multi-sensor fusion** | 11 simultaneous sensor channels, matching modern wearable + smart bedroom setups |
| **Clinical validity** | All physiological ranges calibrated to published PSG reference values |

---

## 4. Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Raw CSV (176K rows)                        │
│   16 columns × 1-min resolution × 30 users × 15 nights      │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 1: Load, Smooth & Feature Engineer                    │
│  • Moving-average smoothing on environmental channels        │
│  • Delta features (Δ per minute)                             │
│  • Rolling statistics (5-min window mean & std)              │
│  → 11 base + 6 delta + 5 rolling = 22 features              │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 2: User-Stratified Split & RobustScaler               │
│  • 70% train / 15% val / 15% test (by User_ID)              │
│  • RobustScaler fitted on TRAIN only                         │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 3: Sliding Window Segmentation                        │
│  • 30-minute windows, stride 3 (parallelised on 4 cores)    │
│  • Each window: [30 timesteps × 22 features]                │
│  • Centre-of-window labelling                                │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 4: Class Imbalance Handling                           │
│  • Minority oversampling (target 60% of majority)            │
│  • Boosted class weights (balanced × 1.5 for minorities)     │
│  • Focal Loss (γ=2.0)                                        │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 7: MOPSO Hyperparameter Optimisation (optional)       │
│  • Internally builds CNN–GRU–LSTM variants (Stages 5-6)     │
│  • Quick-trains each variant (~5 epochs) to evaluate fitness │
│  • 3 objectives: accuracy, false alarm rate, param count     │
│  • Selects Pareto-optimal hyperparameters for final model    │
│  • Re-windows data if MOPSO selects a different window size  │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 8: Build & Train Final Model                          │
│  (using MOPSO-found or default hyperparameters)              │
│  ┌─────────────────────────────────────────┐                 │
│  │  CNN Encoder (Residual + Squeeze-Excite)│                 │
│  └────────────────┬────────────────────────┘                 │
│       ┌───────────┼───────────┐                              │
│  [BiGRU Pool] [BiLSTM Pool] [CNN Pool]                       │
│       └───────────┼───────────┘                              │
│            Concatenate → Dense → Softmax(4)                  │
│  • Train+Val merged (85%) with 10% monitor hold-out          │
│  • Focal Loss + Adam + cosine LR + early stopping            │
│  • Full training (50+ epochs), saves best checkpoint         │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGES 9-10: Evaluation & Sleep Quality Score               │
│  • Accuracy, Cohen's κ, per-stage P/R/F1, FAR               │
│  • HRV Recovery Score                                        │
│  • Composite Sleep Quality Score (0–100)                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Stage 1 — Data Loading & Temporal Feature Engineering

### 5.1 Moving-Average Smoothing

Environmental channels (`Ambient_Temp`, `Humidity`, `Light_Lux`) are smoothed to remove sensor jitter:

$$\text{smooth}(x_t) = \frac{1}{k}\sum_{i=0}^{k-1} x_{t - \lfloor k/2 \rfloor + i}$$

Where $k = 3$ (3-minute symmetric kernel). This is applied via `np.convolve(..., mode="same")`.

### 5.2 Delta Features (Minute-to-Minute Change)

For each source signal $s \in \{\text{HR, RMSSD, SpO2, Resp\_Rate, SVM, Body\_Temp}\}$, the first-order difference is computed per user-day session:

$$\Delta s_t = s_t - s_{t-1} \quad (\Delta s_0 = 0)$$

This captures the **rate of change** — critical for detecting sleep stage transitions. For example, a sharp HR drop ($\Delta \text{HR} \ll 0$) often signals transition from Awake → Deep.

### 5.3 Rolling Statistics (5-Minute Window)

Rolling means and standard deviations provide **short-term trend context**:

$$\text{roll5\_mean}(s_t) = \frac{1}{\min(5, t+1)} \sum_{i=\max(0, t-4)}^{t} s_i$$

$$\text{roll5\_std}(s_t) = \sqrt{\frac{1}{\min(5, t+1) - 1} \sum_{i=\max(0,t-4)}^{t} \left(s_i - \text{roll5\_mean}(s_t)\right)^2}$$

Computed for:
- **HR**: `HR_roll5_mean`, `HR_roll5_std`
- **RMSSD**: `RMSSD_roll5_mean`, `RMSSD_roll5_std`
- **SpO2**: `SpO2_roll5_mean` (mean only)

### 5.4 Final Feature Vector (22 dimensions)

| # | Feature | Type |
|---|---|---|
| 1 | HR | Base sensor |
| 2 | RMSSD | Base sensor |
| 3 | HR_Stability | Base sensor |
| 4 | SpO2 | Base sensor |
| 5 | Resp_Rate | Base sensor |
| 6 | Apnea_Event | Base sensor |
| 7 | SVM | Base sensor |
| 8 | Body_Temp | Base sensor |
| 9 | Ambient_Temp | Base sensor |
| 10 | Humidity | Base sensor |
| 11 | Light_Lux | Base sensor |
| 12 | HR_delta | Engineered (Δ) |
| 13 | RMSSD_delta | Engineered (Δ) |
| 14 | SpO2_delta | Engineered (Δ) |
| 15 | Resp_Rate_delta | Engineered (Δ) |
| 16 | SVM_delta | Engineered (Δ) |
| 17 | Body_Temp_delta | Engineered (Δ) |
| 18 | HR_roll5_mean | Engineered (rolling) |
| 19 | HR_roll5_std | Engineered (rolling) |
| 20 | RMSSD_roll5_mean | Engineered (rolling) |
| 21 | RMSSD_roll5_std | Engineered (rolling) |
| 22 | SpO2_roll5_mean | Engineered (rolling) |

---

## 6. Stage 2 — User-Stratified Splitting & Scaling

### 6.1 User-Level Split

Data is split by **entire users** (not by individual rows) to prevent data leakage:

| Split | Users | Rows | Purpose |
|---|---|---|---|
| Train | 70% of users (~21) | ~123K rows | Model training |
| Validation | 15% of users (~5) | ~26K rows | Merged into train after MOPSO; 10% carved out as monitor set |
| Test | 15% of users (~4) | ~26K rows | Final blind evaluation (never touched until the end) |

This ensures the model is evaluated on **completely unseen individuals**, testing generalization to new physiological profiles.

### 6.2 RobustScaler

The `RobustScaler` is fitted **only on training data**, then applied to all splits. It uses the median and interquartile range (IQR) instead of mean/std, making it robust to outliers (e.g., apnea-induced SpO2 drops):

$$x_{\text{scaled}} = \frac{x - \text{median}(x)}{\text{IQR}(x)} = \frac{x - Q_2}{Q_3 - Q_1}$$

Where $Q_1$, $Q_2$, $Q_3$ are the 25th, 50th, and 75th percentiles of the training data.

---

## 7. Stage 3 — Sliding Window Segmentation

### 7.1 Windowing Parameters

| Parameter | Default Value | Description |
|---|---|---|
| `window_size` | 30 | Number of consecutive timesteps (minutes) per sample |
| `stride` | 3 | Step between window starts (overlapping windows) |

Each window produces a tensor of shape `[30, 22]` — 30 minutes of 22 features.

### 7.2 Centre-of-Window Labelling

Instead of majority-vote (which would drown out minority stages at transition boundaries), the label is taken from the **centre timestep**:

$$y_{\text{window}} = \text{Label}\left[t_{\text{start}} + \left\lfloor\frac{W}{2}\right\rfloor\right]$$

Where $W = 30$ is the window size. This preserves REM and Deep labels that would otherwise be outvoted by the dominant Light class during stage transitions.

### 7.3 Parallelisation

Windowing is parallelised over user-day groups using `joblib.Parallel` with 4 CPU cores (loky backend), processing all user-day sessions concurrently.

---

## 8. Stage 4 — Class Imbalance Handling

Sleep datasets are heavily skewed — Light sleep dominates (~50% of data), while Deep and REM are under-represented.

### 8.1 Minority Oversampling

Minority class windows are duplicated until they reach 60% of the majority class count:

$$n_{\text{target}} = \lfloor 0.6 \times n_{\text{majority}} \rfloor$$

For each class where $n_{\text{class}} < n_{\text{target}}$, windows are randomly duplicated (with replacement) to reach the target count.

### 8.2 Balanced Class Weights with Minority Boost

sklearn's `compute_class_weight("balanced")` produces:

$$w_c = \frac{N}{K \cdot n_c}$$

Where $N$ = total samples, $K$ = number of classes, $n_c$ = samples of class $c$.

An additional **1.5× boost** is applied to any weight > 1.0 (i.e., under-represented classes):

$$w_c^{\text{final}} = \begin{cases} w_c \times 1.5 & \text{if } w_c > 1.0 \\ w_c & \text{otherwise} \end{cases}$$

---

## 9. Stage 5 — Hybrid CNN–GRU–LSTM Model Architecture

> This section describes the model architecture template. When MOPSO is enabled, it instantiates many variants of this architecture (with different filter counts, unit sizes, dropout, etc.) during its hyperparameter search. The final model is then built with the MOPSO-selected (or default) hyperparameters and trained for the full epoch budget in Stage 8.

### 9.1 Input

$$\mathbf{X} \in \mathbb{R}^{B \times T \times F}$$

Where $B$ = batch size, $T = 30$ (window size), $F = 22$ (features).

### 9.2 CNN Encoder (Shared)

The CNN branch extracts local temporal patterns through a stack of convolutional blocks:

**Conv Block**:
$$\mathbf{h} = \text{ReLU}\left(\text{BN}\left(\text{Conv1D}(\mathbf{x}; \mathbf{W}, b)\right)\right)$$

Where Conv1D with kernel size $k$ and $f$ filters computes:
$$\text{Conv1D}(\mathbf{x})_j = \sum_{i=0}^{k-1} \mathbf{W}_{j,i} \cdot \mathbf{x}_{t+i} + b_j$$

**Residual Block**:
$$\mathbf{h}_{\text{res}} = \text{ConvBlock}_2\left(\text{ConvBlock}_1(\mathbf{x})\right) + \text{Proj}(\mathbf{x})$$

Where $\text{Proj}(\mathbf{x})$ is a 1×1 convolution if the channel dimensions differ.

**Squeeze-and-Excitation (SE) Block** — channel attention:

$$\mathbf{z} = \text{GlobalAvgPool}(\mathbf{h}) \in \mathbb{R}^C$$

$$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}\left(\mathbf{W}_1 \cdot \mathbf{z}\right)\right) \in \mathbb{R}^C$$

$$\hat{\mathbf{h}} = \mathbf{h} \odot \mathbf{s}$$

Where $\mathbf{W}_1 \in \mathbb{R}^{C/r \times C}$, $\mathbf{W}_2 \in \mathbb{R}^{C \times C/r}$, $r=4$ (reduction ratio), and $\odot$ is channel-wise multiplication. This learns to re-weight feature channels — e.g., attending more to HR/RMSSD during transition detection.

**Full CNN Path**:
1. Conv1D(64, k=3) → BN → ReLU
2. Residual Conv Block(64, k=3) → SE(ratio=4)
3. Conv1D(128, k=3) → BN → ReLU → MaxPool(2)
4. Residual Conv Block(128, k=3) → SE(ratio=4)

Output: $\mathbf{C} \in \mathbb{R}^{B \times T/2 \times 128}$

### 9.3 Three Parallel Branches

The CNN feature map feeds three branches simultaneously:

**1. Bidirectional GRU** (short-term disturbance dynamics):

$$\overrightarrow{\mathbf{h}}_t = \text{GRU}_{\text{fwd}}(\mathbf{C}_t, \overrightarrow{\mathbf{h}}_{t-1})$$

$$\overleftarrow{\mathbf{h}}_t = \text{GRU}_{\text{bwd}}(\mathbf{C}_t, \overleftarrow{\mathbf{h}}_{t+1})$$

$$\mathbf{g}_t = [\overrightarrow{\mathbf{h}}_t \| \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2 \times 64}$$

GRU update equations:

$$z_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1})$$

$$r_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1})$$

$$\tilde{h}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (r_t \odot \mathbf{h}_{t-1}))$$

$$h_t = (1 - z_t) \odot \mathbf{h}_{t-1} + z_t \odot \tilde{h}_t$$

→ LayerNorm → GlobalAvgPool → $\mathbf{g}_{\text{pool}} \in \mathbb{R}^{128}$

**2. Bidirectional LSTM** (long-term sleep cycle dynamics):

$$\mathbf{l}_t = [\overrightarrow{\text{LSTM}}(\mathbf{C}_t) \| \overleftarrow{\text{LSTM}}(\mathbf{C}_t)]$$

LSTM gating equations:

$$f_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$$

$$i_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$$

$$o_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$$

$$\tilde{c}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c)$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$h_t = o_t \odot \tanh(c_t)$$

→ LayerNorm → GlobalAvgPool → $\mathbf{l}_{\text{pool}} \in \mathbb{R}^{128}$

**3. CNN Global Context**:

$$\mathbf{c}_{\text{pool}} = \text{GlobalAvgPool}(\mathbf{C}) \in \mathbb{R}^{128}$$

### 9.4 Merge & Classification Head

$$\mathbf{m} = [\mathbf{c}_{\text{pool}} \| \mathbf{g}_{\text{pool}} \| \mathbf{l}_{\text{pool}}] \in \mathbb{R}^{384}$$

$$\mathbf{h}_1 = \text{Dropout}_{0.30}\left(\text{BN}\left(\text{ReLU}\left(\mathbf{W}_1 \mathbf{m} + \mathbf{b}_1\right)\right)\right) \quad (\mathbf{W}_1 \in \mathbb{R}^{128 \times 384})$$

$$\mathbf{h}_2 = \text{Dropout}_{0.15}\left(\text{ReLU}\left(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2\right)\right) \quad (\mathbf{W}_2 \in \mathbb{R}^{64 \times 128})$$

$$\hat{\mathbf{y}} = \text{Softmax}\left(\mathbf{W}_o \mathbf{h}_2 + \mathbf{b}_o\right) \in \mathbb{R}^4$$

Where:
$$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{4} e^{z_j}}$$

Output: probability distribution over {Awake, Light, Deep, REM}.

---

## 10. Stage 6 — Loss Function (Sparse Focal Loss)

Standard cross-entropy fails for imbalanced sleep data because the model can achieve decent accuracy by always predicting "Light". **Focal Loss** addresses this by down-weighting easy (well-classified) examples and focusing on hard (minority/ambiguous) ones:

$$\mathcal{L}_{\text{FL}} = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

Where:
- $p_t$ = model's predicted probability for the true class
- $\gamma = 2.0$ (focusing parameter — higher = more focus on hard examples)
- $\alpha_t$ = per-class weight from the boosted class weights (Section 8.2)

**Intuition**: When the model correctly predicts Deep sleep with $p_t = 0.95$, the modulating factor $(1 - 0.95)^2 = 0.0025$ makes this loss negligible. When it struggles with $p_t = 0.2$, the factor $(1 - 0.2)^2 = 0.64$ keeps the gradient strong.

L2 regularisation ($\lambda = 10^{-4}$) is applied to all convolutional and dense kernels:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{FL}} + \lambda \sum_{\mathbf{W}} \|\mathbf{W}\|_2^2$$

---

## 11. Stage 7 — MOPSO Hyperparameter Optimisation

> **Execution order note**: When the `--mopso` flag is passed, MOPSO runs **before** the final model is built and trained. It performs a multi-objective search over 8 hyperparameters, internally building and quick-training candidate models (typically 5 epochs each) to evaluate fitness. Once the Pareto-optimal hyperparameters are selected, they are used to construct and train the final model in Stage 8.

### 11.1 Multi-Objective Particle Swarm Optimisation

MOPSO simultaneously optimises three conflicting objectives:

$$\min_{\theta} \begin{bmatrix} f_1(\theta) \\ f_2(\theta) \\ f_3(\theta) \end{bmatrix} = \begin{bmatrix} 1 - \text{Accuracy}_{\text{val}} \\ \text{FAR}_{\text{val}} \\ \log_{10}(\text{param\_count}) \end{bmatrix}$$

### 11.2 Particle Dynamics

Each particle $i$ has position $\mathbf{x}_i \in [0,1]^8$ and velocity $\mathbf{v}_i$:

$$\mathbf{v}_i^{(t+1)} = w \cdot \mathbf{v}_i^{(t)} + c_1 r_1 \left(\mathbf{p}_i - \mathbf{x}_i^{(t)}\right) + c_2 r_2 \left(\mathbf{g} - \mathbf{x}_i^{(t)}\right)$$

$$\mathbf{x}_i^{(t+1)} = \text{clip}\left(\mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)},\; 0,\; 1\right)$$

Where:
- $w = 0.5$ (inertia weight)
- $c_1 = 1.5$ (cognitive coefficient — attraction to personal best)
- $c_2 = 1.5$ (social coefficient — attraction to Pareto guide)
- $r_1, r_2 \sim \mathcal{U}(0,1)^8$ (stochastic exploration)
- $\mathbf{p}_i$ = particle's personal best position
- $\mathbf{g}$ = randomly selected guide from Pareto archive

### 11.3 Search Space Decoding

The 8-dimensional $[0,1]$ position vector is decoded to hyperparameters:

| Dimension | Parameter | Decoding |
|---|---|---|
| 0 | `cnn_filters_1` | Index into [32, 64, 128] |
| 1 | `cnn_filters_2` | Index into [64, 128, 256] |
| 2 | `gru_units` | Index into [32, 64, 128] |
| 3 | `lstm_units` | Index into [32, 64, 128] |
| 4 | `dense_units` | Index into [64, 128, 256] |
| 5 | `dropout` | $0.10 + x \times 0.40 \in [0.10, 0.50]$ |
| 6 | `learning_rate` | $10^{-4 + 2x} \in [10^{-4}, 10^{-2}]$ (log-scale) |
| 7 | `window_size` | Index into [20, 25, 30, 35, 40] |

### 11.4 Pareto Dominance

Solution $\mathbf{a}$ **dominates** $\mathbf{b}$ iff:

$$\forall k: f_k(\mathbf{a}) \leq f_k(\mathbf{b}) \quad \text{and} \quad \exists k: f_k(\mathbf{a}) < f_k(\mathbf{b})$$

The Pareto archive stores all non-dominated solutions discovered across all iterations.

### 11.5 Best Solution Selection

| Priority Mode | Selection Method |
|---|---|
| `accuracy` | $\arg\min_i \; f_1(\mathbf{s}_i)$ (minimize error) |
| `efficiency` | $\arg\min_i \; 0.5 \cdot \hat{f}_1 + 0.5 \cdot \hat{f}_3$ (normalised accuracy + params) |
| `balanced` | $\arg\min_i \; \frac{1}{3}(\hat{f}_1 + \hat{f}_2 + \hat{f}_3)$ (equal weight on all objectives) |

Where $\hat{f}_k$ are min-max normalised objectives.

### 11.6 Post-MOPSO Handoff

After MOPSO completes, the best hyperparameters (`cnn_filters_1`, `cnn_filters_2`, `gru_units`, `lstm_units`, `dense_units`, `dropout`, `learning_rate`, `window_size`) are injected into the model configuration. If MOPSO selected a different `window_size` than the default, the data is re-windowed before proceeding to Stage 8.

---

## 12. Stage 8 — Training & Optimisation

> The final model is now built using the hyperparameters discovered by MOPSO (or defaults if MOPSO was skipped). Train and validation sets are merged (85% of data) for maximum training signal, with a 10% stratified hold-out carved from the merged set as a monitor set for early stopping. The test set (15%) remains fully blind until final evaluation.

### 12.1 Optimizer

**Adam** with gradient clipping:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

With `clipnorm=1.0`: if $\|\nabla\| > 1.0$, gradients are rescaled to unit norm.

### 12.2 Cosine Annealing Learning Rate Schedule

$$\eta_{\text{epoch}} = \eta_0 \cdot \frac{1}{2}\left(1 + \cos\left(\frac{\pi \cdot \text{epoch}}{E_{\text{max}}}\right)\right)$$

Where $\eta_0 = 10^{-3}$ (or MOPSO-found value) and $E_{\text{max}} = 50$.

### 12.3 Callbacks

| Callback | Monitor | Config |
|---|---|---|
| ModelCheckpoint | `val_accuracy` (max) | Saves best model to `best_model.keras` |
| EarlyStopping | `val_accuracy` | Patience = 15 epochs, restores best weights |
| CSVLogger | — | Logs all epoch metrics to `training_log.csv` |
| LearningRateScheduler | — | Cosine annealing |

### 12.4 Train+Val Merge

After MOPSO hyperparameter search is complete (or skipped), train and validation sets are **merged** for final training (85% of data), with a 10% stratified hold-out from the merged set carved out as a monitor set for early stopping. The test set remains fully blind.

---

## 13. Stage 9 — Evaluation Metrics

### 13.1 Standard Classification Metrics

**Accuracy**:
$$\text{Acc} = \frac{\text{TP}_{\text{total}}}{N}$$

**Per-class Precision, Recall, F1**:
$$P_c = \frac{TP_c}{TP_c + FP_c}, \quad R_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 P_c R_c}{P_c + R_c}$$

### 13.2 Cohen's Kappa

Measures agreement beyond chance:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where $p_o$ = observed accuracy, $p_e$ = expected accuracy by chance (sum of product of marginal proportions).

### 13.3 False Alarm Rate (FAR)

Per-class:
$$\text{FAR}_c = \frac{FP_c}{FP_c + TN_c}$$

Mean:
$$\overline{\text{FAR}} = \frac{1}{K}\sum_{c=1}^{K} \text{FAR}_c$$

### 13.4 Sleep Efficiency

Fraction of windows classified as any sleep stage (not Awake):

$$\text{SE} = \frac{|\{i : \hat{y}_i > 0\}|}{N}$$

### 13.5 HRV Recovery Score

Measures RMSSD trend across the night session via linear regression:

$$\text{slope} = \frac{\text{d(RMSSD)}}{\text{d}t} \quad \text{(from } \texttt{np.polyfit(x, rmssd, 1)}\text{)}$$

$$\text{HRV\_Recovery} = \text{clip}\left(0.5 + \frac{\text{slope}}{2 \cdot |\text{slope}|_{\max}},\; 0,\; 1\right)$$

Positive slope → RMSSD improving overnight → good recovery (score > 0.5).
Negative slope → RMSSD declining → poor recovery (score < 0.5).

---

## 14. Stage 10 — Final Sleep Quality Score

The composite **Sleep Quality Score** (SQS) compresses all evaluation dimensions into a single 0–100 metric:

$$\boxed{\text{SQS} = \left(0.40 \cdot \text{Acc} + 0.25 \cdot \max(\kappa, 0) + 0.20 \cdot \text{SE} + 0.15 \cdot (1 - \overline{\text{FAR}})\right) \times 100}$$

| Component | Weight | Rationale |
|---|---|---|
| Accuracy | 40% | Overall classification correctness |
| Cohen's Kappa ($\kappa$) | 25% | Agreement quality beyond chance; penalises class-biased models |
| Sleep Efficiency | 20% | Proportion of night spent asleep (clinical benchmark ≥ 85%) |
| 1 − Mean FAR | 15% | Low false alarm rate = fewer spurious wake detections = reliable monitoring |

**Interpretation**:
| SQS Range | Quality |
|---|---|
| 85–100 | Excellent — clinical-grade reliability |
| 70–84 | Good — suitable for consumer health tracking |
| 50–69 | Fair — useful for trend analysis, not individual epochs |
| < 50 | Poor — model needs retraining or more data |

---

## 15. End-to-End Formula Summary

Here is the complete chain of mathematical operations from raw input to final output:

```
Raw CSV row (16 columns, 1 minute)
        │
        ▼
Smooth: x̃ = movAvg(x, k=3)  [environmental channels]
        │
        ▼
Delta:  Δsₜ = sₜ - sₜ₋₁
Roll:   μ₅(sₜ), σ₅(sₜ)    → 22 features per minute
        │
        ▼
Scale:  x_scaled = (x - Q₂) / (Q₃ - Q₁)   [RobustScaler]
        │
        ▼
Window: X ∈ ℝ^{30×22}  with label = Label[t + 15]
        │
        ▼
 ┌── MOPSO (optional, runs FIRST) ──────────────────────┐
 │  For each particle (hyperparameter set):              │
 │    Quick-build model → train 5 epochs → evaluate      │
 │    Objectives: [1-Acc, FAR, log₁₀(params)]           │
 │  → Select best from Pareto front                      │
 │  → Inject optimal hyperparams into model config       │
 └───────────────────────┬──────────────────────────────-┘
                         │
                         ▼
CNN:    Conv1D → BN → ReLU → Residual → SE → MaxPool
        │
    ┌───┴───┐
    ▼       ▼
  BiGRU   BiLSTM   (+ CNN GlobalAvgPool)
    │       │              │
    └───┬───┘──────────────┘
        ▼
  Concat[128 + 128 + 128] = 384-dim
        │
        ▼
  Dense(128) → BN → Dropout(0.3) → Dense(64) → Dropout(0.15)
        │
        ▼
  Softmax(4) → P(Awake), P(Light), P(Deep), P(REM)
        │
        ▼
  ŷ = argmax P
        │
        ▼
  Focal Loss: ℒ = -αₜ(1-pₜ)^γ log(pₜ)  +  λΣ‖W‖²
        │
        ▼
  Full Training: Adam + cosine LR + early stopping (50+ epochs)
        │
        ▼
  Metrics: Acc, κ, FAR, SE, HRV Recovery
        │
        ▼
  SQS = (0.40·Acc + 0.25·κ + 0.20·SE + 0.15·(1-FAR)) × 100
```

---

*Generated for the Sleep Intelligence System — Hybrid CNN–GRU–LSTM Architecture with MOPSO Optimisation*
