# Sleep Intelligence IoT — Hybrid CNN–GRU–LSTM Prototype

Comprehensive repository README for a non-invasive sleep intelligence prototype combining physiological, breathing, movement and environmental sensing with a hybrid deep learning pipeline (CNN + GRU + LSTM) optimized via MOPSO.

---

## Overview

This project aims to build a real-world sleep monitoring prototype that goes beyond total sleep duration by integrating multi-modal signals to estimate sleep stages, detect respiratory disturbances and restlessness, and compute sleep & recovery scores.

Key goals:
- Detect sleep stages (light / deep / REM)
- Detect respiratory irregularities (apnea-like events, SpO₂ drops)
- Track movement & restlessness (micro-movements, tossing)
- Produce sleep quality, efficiency and recovery metrics
- Optimize models and thresholds with multi-objective PSO (MOPSO)

---

## Features

- Physiological: HR, HRV, SpO₂, skin temperature
- Breathing: respiration rate, irregular breathing events
- Movement: micro-movements, tossing & turning
- Environment: temperature, humidity, light exposure
- On-device logging (SD card), RTC timestamping, local summary on OLED

---

## Sensors & Hardware

Recommended sensor list and role:

- **MAX30102**: Heart rate (HR), heart rate variability (HRV) and SpO₂
- **DS18B20**: Skin/body temperature
- **MPU6050**: 3-axis accelerometer + gyroscope (movement, restlessness)
- **HC-SR04**: Chest motion (simple breathing rate proxy) or use a piezo/belt for better signal
- **DHT11 / DHT22**: Ambient temperature & humidity
- **BH1750**: Ambient light exposure
- **ESP32**: Main microcontroller for acquisition, transient processing and connectivity
- **SD card module**: Overnight raw data logging
- **RTC (e.g., DS3231)**: Accurate timestamps and sleep-session scheduling
- **OLED (GME-12864)**: Local summary/notifications

Hardware notes:
- Prefer I2C sensors (MPU6050, BH1750) on the same bus with unique addresses.
- Use level shifting where needed; ensure clean power and decoupling for MAX30102.

---

## System Architecture (High-level)

1. Data acquisition on ESP32 → synchronized time from RTC → write raw samples to SD
2. Preprocessing & artifact detection (basic filtering, motion gating)
3. Feature extraction / short-window representations (pulse waveform, SpO₂ dips, breathing envelope)
4. Hybrid model pipeline:
   - CNN: local feature extraction from raw waveforms (pulse morphology, short SpO₂ dips, breathing micro-patterns)
   - GRU: short-term dynamics and transient disturbances (apnea-like pauses, micro-arousals)
   - LSTM: long-term sequence modeling for sleep-cycle structure and recovery patterns
5. MOPSO: multi-objective optimization for model hyperparameters, feature selection and alert thresholds
6. Output: sleep stage timeline, sleep quality score, respiratory disturbance events, summary for OLED & cloud

---

## Data

- Primary dataset (collected / example): [realistic_sleep_dataset_v4.csv](realistic_sleep_dataset_v4.csv)


Data guidelines:
- Record synchronized timestamps, raw sensor channels, and a minimal metadata header per file (device id, timezone, firmware version).
- Segment data into overlapping windows (example: 30s window with 10s stride) for model training and annotation.

---

## Software Stack & Dependencies

Device (ESP32):
- Arduino/PlatformIO or ESP-IDF
- Libraries: `Wire`, `SPI`, `SD`, sensor-specific libs for MAX30102, MPU6050, DHT, BH1750, DS18B20, RTC, OLED

Model training & analysis (PC/Python):
- Python 3.8+
- Typical packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `tensorflow`/`keras` or `pytorch`, `pywt` (wavelets), `tsfresh` or custom feature extractors
- PSO / MOPSO: `pyswarms` (PSO) or multi-objective libraries like `pymoo` / custom MOPSO implementation

Quick pip install (example):

```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn tensorflow pyswarms pymoo
```

---

## Setup & Flashing (ESP32)

1. Wire sensors to ESP32 (I2C, 1-Wire, GPIOs for HC-SR04/SD)
2. Install PlatformIO or Arduino IDE and required sensor libraries
3. Configure `config.h` / `secrets.h` with Wi-Fi, RTC offset, SD pinout
4. Build & flash firmware onto ESP32
5. Start a sleep session: ESP32 will timestamp, log to SD and optionally stream summaries over Wi‑Fi

Example PlatformIO tasks:

```bash
# Build
pio run
# Upload
pio run --target upload
```

---

## Model Training — Suggested Workflow

1. Preprocessing
   - Synchronize channels and resample if needed
   - Apply bandpass filters for PPG (HR) and breathing band
   - Motion-artifact rejection via MPU thresholds
2. Windowing & augmentation
   - Fixed windows (e.g., 30s) + overlap
   - Augment: noise injection, time-scaling, amplitude scaling
3. Feature & label creation
   - Short-term features: pulse peak-to-peak, SpO₂ drops, breathing envelope
   - HRV features (time & frequency domain)
   - Labels: sleep stage (if available), event flags (desaturation, apnea-like pause), restlessness
4. Hybrid model training
   - CNN front-end for local pattern extraction
   - GRU module for disturbance detection (short sequences)
   - LSTM for long-term sleep architecture
5. Optimization with MOPSO
   - Objectives: maximize sleep stage accuracy, minimize false alarms, minimize model inference latency, minimize energy (optional)
   - Search variables: filter sizes, number of units, window size, dropout, learning rate

Example training command (placeholder):

```bash
python train.py --config configs/hybrid_cnn_gru_lstm.yaml
```

---

## Evaluation & Metrics

- Sleep-stage accuracy, precision, recall (per-stage)
- Cohen's kappa for stage agreement
- Event detection metrics: sensitivity & false-alarm rate for breathing irregularities
- HRV-based recovery / stress indices
- End-to-end latency & model size (for on-device viability)

---

## Logging, Alerts & Thresholds

- Store raw sensor data to SD with per-session folders and a small JSON metadata file
- Keep an events CSV for detected events with timestamps
- Use MOPSO to tune alert thresholds and balance sensitivity vs false alarms

---

## Roadmap & Next Steps

Immediate:
- Prototype acquisition firmware for ESP32 with SD logging and OLED summary
- Create a minimal `train.py` and preprocessing pipeline for the dataset

Near-term:
- Implement hybrid CNN–GRU–LSTM training code and baseline metrics
- Integrate MOPSO hyperparameter tuning
- Build simple cloud dashboard and OTA updates

Research / Publication-level:
- Validate against polysomnography (PSG) references
- Improve generalization & personalization

---

## Contributing

Contributions are welcome. Please open issues for bugs, feature requests or to propose dataset/collection protocols.

Guidelines:
- Follow repository style (PEP8 for Python)
- Add tests for preprocessing & model inference when adding code

---

## License

This project uses the MIT License — see the `LICENSE` file for details.

---

## References & Further Reading

- Sleep staging and physiology literature
- PSO and multi-objective optimization papers
- Sensor datasheets: MAX30102, MPU6050, DS18B20

---

If you'd like, I can now:
- add example `train.py` and `requirements.txt`, or
- scaffold the ESP32 firmware folder with basic sensor-read logging, or
- create a minimal data preprocessing notebook using [realistic_sleep_dataset_v4.csv](realistic_sleep_dataset_v4.csv)
