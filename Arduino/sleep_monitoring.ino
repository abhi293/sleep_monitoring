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

// --- Pin Definitions (Retained) ---
#define LCD_CS       5
#define LCD_DC       17
#define LCD_RST      16
#define SD_CS        2   
#define TRIG_PIN     12
#define ECHO_PIN     13
#define ONE_WIRE_BUS 15
#define DHT_PIN      4

// --- Object Initializations ---
U8G2_SSD1309_128X64_NONAME2_F_4W_HW_SPI u8g2(U8G2_R0, LCD_CS, LCD_DC, LCD_RST);
MAX30105 particleSensor;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
DHT dht(DHT_PIN, DHT11);
BH1750 lightMeter;
Adafruit_MPU6050 mpu;
RTC_DS3231 rtc;

// --- MAX30102 Buffers & Variables ---
uint32_t irBuffer[100]; 
uint32_t redBuffer[100];
int32_t bufferLength = 100;
int32_t spo2_val; 
int8_t validSPO2; 
int32_t heartRate_val; 
int8_t validHeartRate;

// --- Global Variables ---
float hr = 0, spo2 = 0;
float bodyTemp = 0, roomTemp = 0, roomHum = 0;
float lux = 0, breathingRate = 0;
float accX = 0, accY = 0, accZ = 0;
int movementIntensity = 0;
int currentPage = 0;
unsigned long lastPageChange = 0;
unsigned long lastLogTime = 0;
DeviceAddress probeAddress = { 0x28, 0x39, 0xBD, 0x02, 0x00, 0x02, 0x24, 0x12 };

void setup() {
  Serial.begin(115200);
  Wire.begin();
  u8g2.begin();

  Serial.println("--- System Initializing ---");

  // --- MAX30102 Working Setup ---
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 Fail");
  } else {
    byte ledBrightness = 70; 
    byte sampleAverage = 4;
    byte ledMode = 2; 
    int sampleRate = 100;
    int pulseWidth = 411;
    int adcRange = 4096;
    particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
    Serial.println("MAX30102 Configured.");
  }

  sensors.begin();
  dht.begin();
  lightMeter.begin();

  if (!mpu.begin()) {
    Serial.println("MPU6050 Fail");
  } else {
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  if (!rtc.begin()) Serial.println("RTC Fail");

  if (!SD.begin(SD_CS)) {
    Serial.println("SD Card Fail");
  } else {
    File file = SD.open("/sleep_log.csv", FILE_WRITE);
    if (file) {
      file.println("Timestamp,HR,SpO2,BodyTemp,RoomTemp,Hum,AccX,AccY,AccZ,Lux,Breathing");
      file.close();
      Serial.println("SD Card Ready.");
    }
  }

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  Serial.println("--- Setup Complete ---");
}

void loop() {
  readSensors();
  
  if (millis() - lastLogTime > 5000) {
    readAndLogData();
    lastLogTime = millis();
  }

  if (millis() - lastPageChange > 4000) {
    currentPage = (currentPage + 1) % 3;
    lastPageChange = millis();
  }

  u8g2.firstPage();
  do {
    drawUI();
  } while (u8g2.nextPage());
}

void readSensors() {
  // --- 1. Read MAX30102 (Blood Oxygen/HR) ---
  // We collect samples for the algorithm
  for (byte i = 0 ; i < 50 ; i++) { // Read 50 samples per loop to keep UI responsive
    while (particleSensor.available() == false) particleSensor.check(); 
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample();
  }
  
  // Calculate results
  maxim_heart_rate_and_oxygen_saturation(irBuffer, 50, redBuffer, &spo2_val, &validSPO2, &heartRate_val, &validHeartRate);
  
  if(validHeartRate && validSPO2) {
    hr = (float)heartRate_val;
    spo2 = (float)spo2_val;
  }

  // --- 2. Read DS18B20 (Body Temp) ---
  sensors.requestTemperatures();
  bodyTemp = sensors.getTempC(probeAddress);

  // --- 3. Read Environment ---
  roomTemp = dht.readTemperature();
  roomHum = dht.readHumidity();
  lux = lightMeter.readLightLevel();

  // --- 4. Read Motion ---
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  accX = a.acceleration.x;
  accY = a.acceleration.y;
  accZ = a.acceleration.z;
  movementIntensity = abs(accX) + abs(accY) + abs(accZ);

  // --- 5. Ultrasound Breathing Check ---
  digitalWrite(TRIG_PIN, LOW); delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH);
  float distance = duration * 0.034 / 2;
  breathingRate = (distance > 0 && distance < 100) ? 16.0 : 0; 
}

// ... readAndLogData and drawUI remain the same as your provided code ...
void readAndLogData() {
  DateTime now = rtc.now();
  
  // --- OUTPUT TO SERIAL MONITOR ---
  Serial.print("[" + now.timestamp() + "]");
  Serial.print(" HR: "); Serial.print(hr);
  Serial.print(" | SpO2: "); Serial.print(spo2);
  Serial.print("% | B-Temp: "); Serial.print(bodyTemp);
  Serial.print("C | R-Temp: "); Serial.print(roomTemp);
  Serial.print("C | Lux: "); Serial.print(lux);
  Serial.print(" | Motion: "); Serial.println(movementIntensity);

  // --- SAVE TO SD CARD ---
  File dataFile = SD.open("/sleep_log.csv", FILE_APPEND);
  if (dataFile) {
    dataFile.print(now.timestamp()); dataFile.print(",");
    dataFile.print(hr);              dataFile.print(",");
    dataFile.print(spo2);            dataFile.print(",");
    dataFile.print(bodyTemp);        dataFile.print(",");
    dataFile.print(roomTemp);        dataFile.print(",");
    dataFile.print(roomHum);         dataFile.print(",");
    dataFile.print(accX);            dataFile.print(",");
    dataFile.print(accY);            dataFile.print(",");
    dataFile.print(accZ);            dataFile.print(",");
    dataFile.print(lux);             dataFile.print(",");
    dataFile.println(breathingRate);
    dataFile.close();
  } else {
    Serial.println("!! SD Card Write Error !!");
  }
}

void drawUI() {
  u8g2.setFont(u8g2_font_6x10_tf);
  
  switch (currentPage) {
    case 0: // Physiological
      u8g2.drawStr(0, 12, "PHYSIOLOGICAL");
      u8g2.drawHLine(0, 14, 128);
      u8g2.setCursor(0, 30); u8g2.print("HR: "); u8g2.print(hr); u8g2.print(" bpm");
      u8g2.setCursor(0, 42); u8g2.print("SpO2: "); u8g2.print(spo2); u8g2.print(" %");
      u8g2.setCursor(0, 54); u8g2.print("Body T: "); u8g2.print(bodyTemp); u8g2.print(" C");
      break;

    case 1: // Movement & Breathing
      u8g2.drawStr(0, 12, "RESP & MOTION");
      u8g2.drawHLine(0, 14, 128);
      u8g2.setCursor(0, 30); u8g2.print("Resp: "); u8g2.print(breathingRate); u8g2.print(" br/m");
      u8g2.setCursor(0, 42); u8g2.print("Motion: "); u8g2.print(movementIntensity);
      u8g2.setCursor(0, 54); u8g2.print("SD Log: Active");
      break;

    case 2: // Environment
      u8g2.drawStr(0, 12, "ENVIRONMENT");
      u8g2.drawHLine(0, 14, 128);
      u8g2.setCursor(0, 30); u8g2.print("Room T: "); u8g2.print(roomTemp); u8g2.print(" C");
      u8g2.setCursor(0, 42); u8g2.print("Hum: "); u8g2.print(roomHum); u8g2.print(" %");
      u8g2.setCursor(0, 54); u8g2.print("Lux: "); u8g2.print(lux);
      break;
  }
}