# Arduino Sleep Monitoring System

This folder contains Arduino code for a comprehensive sleep monitoring system using an ESP32 microcontroller. The system collects multiple physiological and environmental parameters for sleep analysis.

## Overview

Two versions of the sleep monitoring firmware are available:
- **sleep_monitoring.ino** - Basic version with core sensors
- **sleep_monitoring2.ino** - Extended version with additional sensors (microphone, GSR)

## Hardware Requirements

### Microcontroller
- **ESP32** (WROOM or WROVER module)
- 3.3V power supply with adequate current capacity (~1A+)

### Sensors Included

| Sensor | Model | Purpose | Interface |
|--------|-------|---------|-----------|
| **Pulse Oximeter** | MAX30102 | Heart rate & SpO2 measurement | I2C |
| **Temperature (Body)** | DS18B20 | Core body temperature | 1-Wire |
| **Temperature/Humidity** | DHT11 | Room conditions | Digital |
| **Light Sensor** | BH1750 | Ambient light level | I2C |
| **Accelerometer** | MPU6050 (6-axis IMU) | Motion & movement | I2C |
| **Real-Time Clock** | DS3231 | Timestamp data | I2C |
| **OLED Display** | SSD1309 (128x64) | UI display | SPI |
| **SD Card Module** | Generic | Data logging | SPI |
| **Ultrasonic Sensor** | HC-SR04 | Breathing rate estimation | Digital |
| **Digital Microphone** | INMP441 / SPH0645 | Audio (sleep_monitoring2.ino) | I2S |
| **GSR Sensor** | Generic | Galvanic Skin Response (sleep_monitoring2.ino) | ADC |
| **Buzzer** | Piezo | Alerts (sleep_monitoring2.ino) | Digital |

## Pin Configuration

### I2C Bus (Shared)
- **SDA**: GPIO 21
- **SCL**: GPIO 22
- Devices: MPU6050, MAX30102, BH1750, RTC_DS3231

### SPI Bus (Shared)
- **SCK**: GPIO 18
- **MOSI**: GPIO 23
- **MISO**: GPIO 19

### OLED Display (SSD1309)
- **CS**: GPIO 5
- **DC**: GPIO 17
- **RST**: GPIO 16

### SD Card Module
- **CS**: GPIO 2

### Sensors
- **Ultrasonic (HC-SR04)**: TRIG = GPIO 12, ECHO = GPIO 13
- **DS18B20 Temperature**: GPIO 15 (Requires 4.7kΩ pull-up resistor)
- **DHT11**: GPIO 4
- **Buzzer**: GPIO 14 (sleep_monitoring2.ino)

### I2S Microphone (sleep_monitoring2.ino only)
- **SCK**: GPIO 26
- **WS**: GPIO 25
- **SD**: GPIO 33

### ADC Input (sleep_monitoring2.ino only)
- **GSR Sensor**: GPIO 34 (ADC1_CHANNEL_6)

## Required Software Tools

### 1. **Arduino IDE** (Recommended)
- Download: https://www.arduino.cc/en/software
- Version: 1.8.x or 2.0+
- Install ESP32 board package via Board Manager

### 2. **VS Code + PlatformIO** (Alternative)
- Install PlatformIO extension in VS Code
- Automatic dependency management
- Better code navigation and debugging

### 3. **USB to Serial Driver**
- **CH340/CH341 Driver** (if your ESP32 board uses it)
- **CP2102 Driver** (alternative USB chip)
- Download from: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers

## Required Libraries

Install these libraries via Arduino IDE Library Manager (`Sketch → Include Library → Manage Libraries`):

| Library | Version | Purpose | Source |
|---------|---------|---------|--------|
| **U8g2** | Latest | OLED display driver | Arduino Library Manager |
| **MAX30105 Lib** | Latest | Pulse oximeter driver | SparkFun (by SparkFun Electronics) |
| **DallasTemperature** | Latest | DS18B20 temperature sensor | Miles Burton |
| **DHT sensor library** | Latest | DHT11/22 sensor | Adafruit |
| **BH1750** | Latest | Light sensor driver | Christopher Law |
| **Adafruit MPU6050** | Latest | 6-axis accelerometer | Adafruit Industries |
| **Adafruit Unified Sensor** | Latest | Sensor abstraction layer | Adafruit Industries |
| **RTClib** | Latest | RTC_DS3231 driver | Adafruit Industries |
| **OneWire** | Latest | 1-Wire protocol driver | Jim Studt |
| **SD** | Latest (Built-in) | SD card support | Arduino (included with ESP32) |
| **SPI** | Latest (Built-in) | SPI communication | Arduino (included) |

### Installation Steps (Arduino IDE)

1. Open Arduino IDE
2. Go to `Sketch → Include Library → Manage Libraries`
3. Search for each library name and click **Install**
4. Restart Arduino IDE

### Installation Steps (PlatformIO)

Add to `platformio.ini`:
```ini
[env:esp32]
platform = espressif32
board = esp32doit-devkit-v1
framework = arduino
lib_deps =
    u8g2/u8g2@^2.34
    sparkfun/SparkFun MAX3010x Pulse and Proximity Sensor Library@^1.1.1
    milesburton/DallasTemperature@^3.9.0
    adafruit/DHT sensor library@^1.4.4
    christopher-law/BH1750@^1.3.1
    adafruit/Adafruit MPU6050@^2.1.0
    adafruit/Adafruit Unified Sensor@^1.1.14
    adafruit/RTClib@^2.1.1
    paulstoffregen/OneWire@^2.3.7
```

## VS Code Extensions (Optional but Recommended)

- **PlatformIO IDE** - Complete IDE for embedded development
- **Arduino** - Arduino support for VS Code
- **Serial Monitor** - Enhanced serial debugging
- **C/C++ IntelliSense** - Code completion and analysis

## Hardware Setup Instructions

### 1. **Power Connection**
- Connect **3.3V** to all sensor VCC pins
- Connect **GND** to all sensor GND pins
- Use adequate power supply (500mA+ recommended)

### 2. **I2C Pull-up Resistors**
- Connect 10kΩ pull-up resistors on SDA and SCL lines

### 3. **DS18B20 Configuration**
- Connect **4.7kΩ pull-up resistor** between signal line and 3.3V
- Update probe address in code if using different sensor:
  ```cpp
  DeviceAddress probeAddress = { 0x28, 0x39, 0xBD, 0x02, 0x00, 0x02, 0x24, 0x12 };
  ```
- Use OneWire protocol to find address if needed

### 4. **SD Card Module**
- Ensure card is FAT32 formatted
- CS pin must be properly configured

### 5. **OLED Display**
- Verify SSD1309 module matches I2C or SPI configuration
- Current setup uses SPI mode

## Uploading Code

### Using Arduino IDE:
1. Connect ESP32 to computer via USB
2. Select `Tools → Board → esp32 → ESP32 Dev Module`
3. Select correct COM port under `Tools → Port`
4. Click **Upload** button (→ icon)
5. Watch Serial Monitor (`Tools → Serial Monitor`) at 115200 baud

### Using PlatformIO:
1. Open project in VS Code
2. Click **Upload** in PlatformIO toolbar
3. Select port if prompted

## Data Logging

- Sleep data is logged to **SD card** in CSV format
- File: `/sleep_log.csv`
- Logs every **5 seconds** by default
- Columns: Timestamp, HR, SpO2, BodyTemp, RoomTemp, Humidity, AccX, AccY, AccZ, Lux, BreathingRate

## Features by Version

### sleep_monitoring.ino (Basic)
✓ Heart rate & SpO2 monitoring  
✓ Body & environment temperature  
✓ Humidity tracking  
✓ Light level monitoring  
✓ Motion/acceleration detection  
✓ Real-time clock & data logging  
✓ OLED UI display  

### sleep_monitoring2.ino (Extended)
✓ All basic features PLUS  
✓ Audio recording (snoring detection)  
✓ GSR (stress) monitoring  
✓ Alert buzzer  
✓ I2S microphone support  
✓ ADC sensor integration  

## Troubleshooting

| Issue | Solution |
|-------|----------|
| USB port not recognized | Install CH340 or CP2102 driver for your OS |
| Libraries not found | Ensure all dependencies are installed via Library Manager |
| Sensor not initializing | Check I2C/SPI connections and pull-up resistors |
| SD card errors | Format card to FAT32 and verify CS pin connection |
| Display not showing | Verify SPI pins and reset sequence |
| Slow performance | Reduce sensor sampling rate or increase clock speed |

## Useful Resources

- ESP32 Documentation: https://docs.espressif.com/projects/esp-idf/
- Arduino Reference: https://www.arduino.cc/reference/
- MAX30102 Datasheet: https://pdfserv.maximintegrated.com/en/ds/MAX30102.pdf
- MPU6050 Guide: https://invensense.tdk.com/
- PlatformIO Docs: https://docs.platformio.org/

## Notes

- Calibrate sensors for accurate readings (especially MAX30102 and DS18B20)
- Update DS18B20 probe address to match your specific sensor
- Adjust sampling rates based on power consumption requirements
- OLED UI cycles through 3 pages automatically (4-second intervals)

---

**Version**: 2.0  
**Last Updated**: May 2026  
**Status**: Production Ready
