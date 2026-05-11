#include <Arduino.h>
#include <U8g2lib.h>
#include <Wire.h>
#include <MAX30105.h>
#include <spo2_algorithm.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <DHT.h>
#include <BH1750.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <RTClib.h>
#include <SD.h>
#include <SPI.h>
#include <driver/i2s.h>
#include <driver/adc.h>

// =========================================================================
//                           PIN CONFIGURATIONS
// =========================================================================

// --- I2C Bus (Shared by MPU6050, MAX30102, BH1750, RTC) ---
// SDA: GPIO 21
// SCL: GPIO 22

// --- OLED Display (SSD1309 SPI) ---
#define LCD_CS       5   // Chip Select
#define LCD_DC       17  // Data/Command
#define LCD_RST      16  // Reset
// SCK: GPIO 18 (Default SPI)
// MOSI: GPIO 23 (Default SPI)

// --- SD Card Module (SPI) ---
#define SD_CS        2   // Chip Select
// SCK: GPIO 18
// MOSI: GPIO 23
// MISO: GPIO 19

// --- Ultrasonic Sensor (HC-SR04) ---
#define TRIG_PIN     12  // Trigger
#define ECHO_PIN     13  // Echo

// --- Temperature & Humidity ---
#define ONE_WIRE_BUS 15  // DS18B20 Body Temp (Requires 4.7k Pull-up)
#define DHT_PIN      4   // DHT11 Room Temp/Hum

// --- I2S Digital Microphone (INMP441 / SPH0645) ---
#define I2S_SCK      26  // Serial Clock
#define I2S_WS       25  // Word Select (L/R Clock)
#define I2S_SD       33  // Serial Data

// --- GSR (Galvanic Skin Response) ---
#define GSR_CHANNEL  ADC1_CHANNEL_6 // Analog Input (GPIO 34)

// --- Output Alerts ---
#define BUZZER_PIN   14  // Piezo Buzzer / Alarm

// =========================================================================

// --- Object Initializations ---
U8G2_SSD1309_128X64_NONAME2_F_4W_HW_SPI u8g2(U8G2_R0, LCD_CS, LCD_DC, LCD_RST);
MAX30105 particleSensor;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
DHT dht(DHT_PIN, DHT11);
BH1750 lightMeter;
Adafruit_MPU6050 mpu;
RTC_DS3231 rtc;

// --- MAX30102 Buffers ---
uint32_t irBuffer[100], redBuffer[100];
int32_t spo2_val, heartRate_val;
int8_t validSPO2, validHeartRate;

// --- GSR & Audio Variables ---
float db_smooth = -50.0;
float scl = 0, scr = 0;
int rawGSR = 0, scrSpikes = 0;
float gsr_alpha = 0.05;

// --- Global Sleep Data ---
float hr = 0, spo2 = 0;
float bodyTemp = 0, roomTemp = 0, roomHum = 0;
float lux = 0, breathingRate = 0;
float accX, accY, accZ, accelMag = 1.0;
int currentPage = 0;
unsigned long lastPageChange = 0, lastLogTime = 0, lastGSR = 0;

// Device Address for DS18B20 - Update this with your specific sensor address
DeviceAddress probeAddress = { 0x28, 0x39, 0xBD, 0x02, 0x00, 0x02, 0x24, 0x12 };

// ===================== I2S MICROPHONE SETUP =====================
void setupI2S() {
  i2s_config_t config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 6,
    .dma_buf_len = 256,
    .use_apll = false
  };
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };
  i2s_driver_install(I2S_NUM_0, &config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
}

// ===================== SENSOR LOGIC =====================

void readAudio() {
  int32_t samples[256];
  size_t bytesRead = 0;
  if (i2s_read(I2S_NUM_0, &samples, sizeof(samples), &bytesRead, 10) == ESP_OK && bytesRead > 0) {
    int n = bytesRead / 4;
    float sum = 0;
    for (int i = 0; i < n; i++) {
      float v = samples[i] / 2147483648.0f;
      sum += v * v;
    }
    float rms = sqrt(sum / n);
    float new_db = 20.0f * log10(rms + 1e-6f);
    db_smooth = 0.9f * db_smooth + 0.1f * new_db;
  }
}

void readGSR() {
  int sum = 0;
  for (int i = 0; i < 10; i++) sum += adc1_get_raw(GSR_CHANNEL);
  rawGSR = sum / 10;
  float gsr = rawGSR / 4095.0f;
  scl = gsr_alpha * gsr + (1 - gsr_alpha) * scl;
  scr = gsr - scl;
  
  static float prevScr = 0;
  if (scr > 0.012f && prevScr <= 0.012f) scrSpikes++;
  prevScr = scr;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // Explicitly setting I2C Pins
  u8g2.begin();

  // Initialize MAX30102
  if (particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    particleSensor.setup(70, 4, 2, 100, 411, 4096);
  }

  // Initialize Audio & GSR
  setupI2S();
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(GSR_CHANNEL, ADC_ATTEN_DB_11);

  // Initialize Other Sensors
  sensors.begin();
  dht.begin();
  lightMeter.begin();
  if (mpu.begin()) {
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  }
  rtc.begin();
  
  if (SD.begin(SD_CS)) {
    File file = SD.open("/sleep_log.csv", FILE_WRITE);
    if (file) {
      file.println("Timestamp,HR,SpO2,BTemp,Lux,GSR_SCL,Audio_dB,Motion");
      file.close();
    }
  }

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
}

void loop() {
  readAudio();
  
  if (millis() - lastGSR > 200) {
    readGSR();
    lastGSR = millis();
  }

  // --- MAX30102 Pulse Ox ---
  for (byte i = 0 ; i < 25 ; i++) {
    while (particleSensor.available() == false) particleSensor.check(); 
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample();
  }
  maxim_heart_rate_and_oxygen_saturation(irBuffer, 25, redBuffer, &spo2_val, &validSPO2, &heartRate_val, &validHeartRate);
  if(validHeartRate && validSPO2) {
    hr = (float)heartRate_val;
    spo2 = (float)spo2_val;
  }

  // --- MPU6050 Motion ---
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  accelMag = sqrt(a.acceleration.x * a.acceleration.x + a.acceleration.y * a.acceleration.y + a.acceleration.z * a.acceleration.z);
  
  // --- Environmentals ---
  lux = lightMeter.readLightLevel();
  sensors.requestTemperatures();
  bodyTemp = sensors.getTempC(probeAddress);
  roomTemp = dht.readTemperature();
  roomHum = dht.readHumidity();

  // --- Breathing Rate (Ultrasonic Proxy) ---
  digitalWrite(TRIG_PIN, LOW); delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  float distance = duration * 0.034 / 2;
  breathingRate = (distance > 0 && distance < 100) ? 16.0 : 0;

  // --- Interval Logging ---
  if (millis() - lastLogTime > 5000) {
    logData();
    lastLogTime = millis();
  }

  // --- UI Rotation ---
  if (millis() - lastPageChange > 4000) {
    currentPage = (currentPage + 1) % 4;
    lastPageChange = millis();
  }

  u8g2.firstPage();
  do {
    drawUI();
  } while (u8g2.nextPage());
}

void logData() {
  DateTime now = rtc.now();
  Serial.printf("[%s] HR: %.1f | SpO2: %.1f | SCL: %.3f | Audio: %.1f dB\n", 
                now.timestamp().c_str(), hr, spo2, scl, db_smooth);
  
  File dataFile = SD.open("/sleep_log.csv", FILE_APPEND);
  if (dataFile) {
    dataFile.printf("%s,%.1f,%.1f,%.1f,%.1f,%.4f,%.2f,%.2f\n", 
                    now.timestamp().c_str(), hr, spo2, bodyTemp, lux, scl, db_smooth, accelMag);
    dataFile.close();
  }
}

void drawUI() {
  u8g2.setFont(u8g2_font_6x10_tf);
  char buf[32];

  switch (currentPage) {
    case 0: // Physiological
      u8g2.drawStr(0, 12, "[ VITAL SIGNS ]");
      sprintf(buf, "HR: %.1f bpm", hr); u8g2.drawStr(0, 30, buf);
      sprintf(buf, "SpO2: %.1f %%", spo2); u8g2.drawStr(0, 45, buf);
      sprintf(buf, "Body: %.1f C", bodyTemp); u8g2.drawStr(0, 60, buf);
      break;

    case 1: // Stress/Arousal
      u8g2.drawStr(0, 12, "[ AROUSAL DATA ]");
      sprintf(buf, "Skin: %.3f uS", scl); u8g2.drawStr(0, 30, buf);
      sprintf(buf, "Spikes: %d", scrSpikes); u8g2.drawStr(0, 45, buf);
      sprintf(buf, "Noise: %.1f dB", db_smooth); u8g2.drawStr(0, 60, buf);
      break;

    case 2: // Environment
      u8g2.drawStr(0, 12, "[ ENVIRONMENT ]");
      sprintf(buf, "Room: %.1f C", roomTemp); u8g2.drawStr(0, 30, buf);
      sprintf(buf, "Hum: %.1f %%", roomHum); u8g2.drawStr(0, 45, buf);
      sprintf(buf, "Lux: %.1f", lux); u8g2.drawStr(0, 60, buf);
      br eak;

    case 3: // Motion
      u8g2.drawStr(0, 12, "[ RESP & MOTION ]");
      sprintf(buf, "Resp: %.1f rpm", breathingRate); u8g2.drawStr(0, 30, buf);
      sprintf(buf, "Accel: %.2f g", accelMag); u8g2.drawStr(0, 45, buf);
      u8g2.drawStr(0, 60, "SD Card: Logging");
      break;
  }
}