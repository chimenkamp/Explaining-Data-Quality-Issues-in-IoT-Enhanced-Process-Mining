# Open Smart Home IoT-IEQ-Energy Data - Dataset Analysis

## Dataset Overview

**Source**: Fraunhofer Institute for Building Physics (IBP), Nürnberg, Germany  
**Domain**: Smart Home / Building Management Systems (BMS)  
**Dataset Type**: Multi-room IoT Sensor Time-Series with Semantic Building Model  
**Task**: Indoor Environmental Quality (IEQ) Monitoring & Energy Management  
**Collection Period**: March 9, 2017 - June 6, 2017 (89 days)  
**Total Measurements**: ~285,954 sensor readings across 37 time-series files  
**Spatial Coverage**: 6 rooms (Kitchen, Bathroom, Toilet, Room 1, Room 2, Room 3)  

## 1. Physical System Description

### 1.1 Smart Home Architecture

The dataset originates from a real-world smart home flat located in Nuremberg, Germany (coordinates: 49.460899°N, 11.069208°E, 300m altitude), equipped with:

1. **Building Structure**: Two-story residential building with instrumented flat
2. **Monitored Spaces**: 6 rooms with sensors + 3 uninstrumented spaces (Lobby, Staircase, Room Before Toilet)
3. **Smart Home System**: EnOcean wireless sensor/actuator network
4. **HVAC System**: Individual room heating with thermostatic radiator valves (TRVs)
5. **Control Strategy**: Schedule-based temperature setpoints per room

### 1.2 Room Inventory and Function

| Room | Type | Function | Sensor Count | Key Characteristics |
|------|------|----------|--------------|---------------------|
| **Kitchen** | Kitchen | Food preparation, cooking | 5-6 sensors | High humidity from cooking, heat generation |
| **Room 1** | Bedroom | Sleeping, resting | 5-6 sensors | Lower setpoints at night, minimal activity |
| **Room 2** | Bedroom | Sleeping, resting | 5-6 sensors | Similar to Room 1 |
| **Room 3** | Living Room | Main activity space | 7-8 sensors | Two thermostats (left/right), outdoor temp sensor |
| **Bathroom** | Bathroom | Hygiene | 5-6 sensors | High humidity spikes, variable occupancy |
| **Toilet** | Toilet | Hygiene | 5-6 sensors | Low occupancy, stable conditions |

### 1.3 Heating and Control System

**Components per Room**:
- **Thermostat Temperature Sensor**: Mounted on radiator, measures air temperature at heating element
- **Wall Temperature Sensor**: Independent air temperature measurement (room center/wall)
- **Thermostatic Radiator Valve (TRV)**: Actuator controlling hot water flow to radiator
- **Setpoint Schedule**: Pre-programmed temperature targets varying by time-of-day and day-of-week
- **Outdoor Temperature**: Virtual weather service data for heating control optimization

**Control Process**:
1. Schedule determines desired setpoint temperature for room
2. Thermostat sensor measures actual radiator air temperature
3. TRV adjusts valve opening to maintain setpoint
4. Wall temperature sensor provides independent room condition verification
5. Outdoor temperature influences heating demand prediction

## 2. Sensor Configuration and Data Structure

### 2.1 Sensor Types and Coverage

The dataset contains **6 primary sensor types** deployed across **6 rooms**:

| Sensor Type | Symbol | Physical Quantity | Unit | Number of Sensors | Total Files |
|-------------|--------|-------------------|------|-------------------|-------------|
| **Temperature** | temp | Indoor air temperature | °C | 6 (one per room) | 6 |
| **Thermostat Temperature** | tempT | Radiator-mounted temperature | °C | 7 (Room 3 has 2) | 7 |
| **Humidity** | humid | Relative humidity | % | 6 (one per room) | 6 |
| **Brightness** | brigh | Illuminance (light level) | lux | 6 (one per room) | 6 |
| **Setpoint History** | tempS | Target temperature | °C | 6 (one per room) | 6 |
| **Outdoor Temperature** | outTemp | External weather | °C | Virtual sensor | 6 (duplicated) |

**Total**: 37 CSV files (31 unique sensor streams + 6 duplicated outdoor temp)

### 2.2 Sensor Deployment Details

**Room 3 Special Configuration**:
- Room 3 (Living Room) has **two thermostat sensors**: `Room3_left_ThermostatTemperature.csv` and `Room3_right_ThermostatTemperature.csv`
- Indicates two separate heating zones or large room requiring dual control
- Most spatially complex room in the flat

**Outdoor Temperature**:
- **Virtual sensor** data from weather service (not physical sensor)
- Same data replicated across 6 files (one per room association)
- Files: `Bathroom_Virtual_OutdoorTemperature.csv`, `Kitchen_Virtual_OutdoorTemperature.csv`, etc.
- Note: `Room2_OutdoorTemperature.csv` lacks "Virtual" prefix but is same data type
- Used for heating control optimization and energy management

**Communication Protocol**:
- **EnOcean wireless technology**: Energy-harvesting wireless sensors (no battery changes)
- Communication connection points defined in semantic model
- Decentralized sensor network with low maintenance requirements

### 2.3 Data File Structure and Characteristics

**File Format**:
- **Format**: Tab-separated values (TSV)
- **Columns**: 2 columns per file
  - Column 1: UNIX timestamp (seconds since epoch)
  - Column 2: Sensor reading (numeric value)
- **No headers**: Data starts immediately
- **Encoding**: UTF-8 text

**Temporal Characteristics**:

| Metric | Value | Notes |
|--------|-------|-------|
| **Collection Period** | 2017-03-09 to 2017-06-06 | 89.1 days continuous |
| **Season** | Late winter → late spring | Heating season transition |
| **Sampling Irregularity** | Yes | Event-driven, not fixed interval |
| **Median Interval** | ~600 seconds (10 min) | Typical between consecutive readings |
| **Interval Std Dev** | ~1,177 seconds (19.6 min) | High variability |
| **Min Interval** | <1 second | During rapid changes |
| **Max Interval** | Hours | During stable conditions |

**Data Volume per Sensor Type**:

| Sensor Type | Typical Records per File | Interval Characteristics |
|-------------|-------------------------|--------------------------|
| **Temperature** | ~10,400-10,800 | Irregular, change-driven |
| **Thermostat Temp** | ~10,200-11,000 | Irregular, similar to temp |
| **Humidity** | ~10,100-10,700 | Irregular, slow-changing |
| **Brightness** | ~10,800-11,200 | More frequent (daylight cycles) |
| **Setpoint History** | ~340-360 | Very sparse (only on schedule changes) |
| **Outdoor Temp** | ~3,700-3,800 | Hourly from weather service |

**Key Observation**: Sensors use **change-driven sampling** (report on significant change) rather than fixed periodic sampling. This is typical for energy-efficient IoT sensors but creates challenges for time-series analysis requiring regular intervals.

### 2.4 Data Quality Characteristics (Raw Data)

**Confirmed Characteristics**:
- ✅ **No explicit missing values**: All files contain complete records
- ✅ **No null/NaN entries**: Numeric values always present
- ✅ **Long-term continuity**: Full 89-day coverage
- ✅ **Consistent formatting**: All files follow same structure

**Potential Quality Issues**:
- ⚠️ **Irregular sampling intervals**: Complicates temporal analysis and aggregation
- ⚠️ **Sampling rate variation**: Different sensors sample at different implicit rates
- ⚠️ **Sensor synchronization**: No guarantee sensors across rooms sample simultaneously
- ⚠️ **Clock drift**: Timestamps rely on local system clocks (potential drift over 89 days)
- ⚠️ **Sensor calibration drift**: Long-term deployment may experience calibration shifts
- ⚠️ **Duplicate outdoor data**: Same outdoor temp in 6 files (redundancy, potential confusion)

## 3. Semantic Building Model (Linked Data)

### 3.1 Ontology Stack

The dataset is accompanied by rich **semantic metadata** using W3C/OGC standards:

| Ontology | Purpose | Prefix | Usage |
|----------|---------|--------|-------|
| **BOT** (Building Topology) | Spatial structure | `bot:` | Sites, buildings, storeys, spaces |
| **SOSA/SSN** | Sensors & Observations | `sosa:`, `ssn:` | Sensors, actuators, properties, features |
| **DogOnt** | Smart home devices | `dog:` | Device types (thermostat, humidity sensor, etc.) |
| **SEAS** | Energy systems | `seas:` | Communication connection points |
| **QUDT/OM** | Units of measure | `qudt:`, `om:` | Celsius, lux, percent, etc. |
| **Schema.org** | General properties | `schema:` | Min/max values, ranges |
| **GeoSpatial** | Location | `geo:` | Latitude, longitude, altitude |

### 3.2 Building Structure Hierarchy

```
:Site1 (bot:Site)
  └── :Building1 (bot:Building)
      ├── :Level1 (Ground Floor)
      └── :Level2 (First Floor)
          └── Spaces (bot:Space):
              ├── :Kitchen (dog:Kitchen)
              ├── :Room1 (dog:Bedroom)
              ├── :Room2 (dog:Bedroom)
              ├── :Room3 (dog:Livingroom)
              ├── :Bathroom (dog:Bathroom)
              ├── :Toilet (dog:Bathroom)
              ├── :Lobby (no sensors)
              ├── :RoomBeforeToilet (no sensors)
              └── :Staircase (no sensors)
```

### 3.3 Sensor-Space Relationships

Each instrumented space contains:
- **Sensors** (`bot:containsElement`, `sosa:Sensor`):
  - Temperature sensor (wall-mounted, room center)
  - Thermostat temperature sensor (radiator-mounted)
  - Humidity sensor (wall-mounted)
  - Brightness sensor (luminance measurement)
  
- **Actuators** (`sosa:Actuator`):
  - Thermostatic radiator valve (TRV)
  
- **Properties** (`ssn:hasProperty`, `sosa:ObservableProperty`, `sosa:ActuatableProperty`):
  - Observable: temperature, humidity, brightness
  - Actuatable: temperature setpoint

**Example for Kitchen**:
```turtle
:Kitchen rdf:type bot:Space, dog:Kitchen, sosa:FeatureOfInterest ;
    bot:containsElement :Kitchen-temp-Sensor, :Kitchen-tempT-Sensor, 
                        :Kitchen-humid-Sensor, :Kitchen-brigh-Sensor,
                        :Kitchen-tempS-Actuator, :Kitchen-heater ;
    ssn:hasProperty :Kitchen-temp, :Kitchen-tempT, :Kitchen-humid, 
                    :Kitchen-brigh, :Kitchen-tempS .
```

### 3.4 Sensor Capabilities and Operating Ranges

The semantic model includes **sensor specifications**:

**Temperature Sensors**:
- Operating Range: 0-40°C
- Measurement Range: 0-40°C
- Type: Wall-mounted air temperature sensor

**Humidity Sensors**:
- Measurement Range: 0-100%
- Type: Relative humidity sensor

**Brightness Sensors**:
- Measurement Range: 0-60,000 lux
- Type: Illuminance sensor (light intensity)

**Communication**:
- Protocol: EnOcean wireless
- Each sensor has unique identifier (e.g., "KITemp", "Room1humid")

### 3.5 Linked Building Data (IFC/Revit Integration)

The dataset includes **3D building model files**:
- `04_Flat.ifc` - Industry Foundation Classes model (open standard)
- `05_Flat.rvt` - Autodesk Revit model (native BIM format)
- `02_BotFromRevit.ttl`, `02_GeoFromRevit.ttl`, `02_PropsFromRevit.ttl` - Extracted RDF from Revit
- `03_GEOM.ttl` - Geometric data from IFC
- `06_ifcOWLfromIFC.ttl` - IFC to OWL conversion

**Purpose**: Links sensor data to precise 3D spatial locations, enabling:
- Spatial reasoning (which rooms are adjacent?)
- Propagation modeling (how does heat transfer between spaces?)
- Visualization (plot sensor values on 3D floor plan)
- Energy simulation integration (building physics models)

## 4. Underlying Process Understanding

### 4.1 Daily Living Process Pattern

The smart home system supports the following **recurring processes**:

#### **Thermal Comfort Management Process**
```
[Wake Up] → [Morning Heating] → [Day Mode] → [Evening Heating] → [Night Cooling] → [Sleep]
    ↓               ↓                ↓               ↓                  ↓            ↓
Setpoint +2°C   Heat to 20°C   Reduce to 18°C  Heat to 21°C      Lower to 16°C   Monitor
```

#### **Room-Specific Process Variants**

**Kitchen Process**:
1. Morning (6-9 AM): Heating for breakfast preparation
2. Cooking events: Humidity spikes, temperature increase, brightness (light on)
3. Ventilation: Humidity decrease, temperature drop
4. Evening (5-8 PM): Dinner preparation (similar to morning)
5. Night: Minimal activity, low setpoint

**Bedroom Process (Room 1, Room 2)**:
1. Morning: Minimal heating during vacancy
2. Day: Low setpoint (16-18°C), room unoccupied
3. Evening: Pre-heating before bedtime
4. Night: Comfortable sleeping temperature (18-20°C)
5. Brightness: Low during sleep, increases at wake

**Living Room Process (Room 3)**:
1. Morning: Moderate heating
2. Day: Main activity space, sustained occupancy
3. Evening: Highest occupancy, entertainment, reading → high brightness
4. Night: Reduced activity, lower setpoint
5. **Dual heating zones**: Left/right thermostats respond to occupancy patterns

**Bathroom Process**:
1. Morning: Shower/bathing → **extreme humidity spike** (60-80%+), temperature increase
2. Ventilation: Rapid humidity decrease
3. Evening: Secondary bathing period
4. Night: Minimal use, stable low values

**Toilet Process**:
1. Low frequency usage throughout day
2. Minimal environmental changes
3. Stable humidity, temperature, brightness
4. Most "boring" room (ideal for baseline/control)

### 4.2 Event Abstraction Strategy

To convert time-series to process mining event logs, we can detect the following **activities**:

#### **Temperature-Based Events**
- `Heating_Start`: Setpoint increased + thermostat temperature rising
- `Heating_Active`: Sustained temperature increase (∆T > 0.5°C in 30 min)
- `Target_Reached`: Thermostat temperature ≈ setpoint (±0.5°C)
- `Cooling_Start`: Setpoint decreased or heating off
- `Stable_Temperature`: Temperature variance < 0.2°C over 1 hour

#### **Humidity-Based Events**
- `Humidity_Spike`: Increase > 10% in < 15 minutes (shower, cooking, washing)
- `Ventilation_Start`: Rapid humidity decrease (>5% in 15 min)
- `High_Humidity_Alert`: Humidity > 60% for > 30 min (mold risk)
- `Normal_Humidity`: Humidity in range 40-55% (comfort zone)

#### **Brightness-Based Events**
- `Light_On`: Brightness increase > 10 lux in < 5 minutes
- `Light_Off`: Brightness decrease < 5 lux
- `Daylight_Start`: Gradual brightness increase (sunrise, 5-7 AM)
- `Daylight_End`: Gradual brightness decrease (sunset, 6-8 PM)
- `Occupancy_Detected`: Light on during dark period

#### **Setpoint-Based Events**
- `Schedule_Change`: Setpoint modified (distinct timestamp in SetpointHistory)
- `Comfort_Mode`: High setpoint (≥20°C)
- `Economy_Mode`: Low setpoint (≤18°C)
- `Night_Mode`: Minimal setpoint (≤16°C)

#### **Multi-Sensor Events (Composite)**
- `Shower_Event`: Humidity spike + temperature increase + light on (Bathroom)
- `Cooking_Event`: Humidity spike + temperature increase (Kitchen)
- `Occupancy_Pattern`: Brightness + temperature + humidity changes correlated
- `Room_Vacant`: Stable all sensors, low brightness, economy setpoint
- `Energy_Waste`: Heating active + room vacant (light off, stable conditions)

### 4.3 Process-Level Trace Construction

**Case Identification Strategy**:

**Option 1: Daily Cycles**
- One case = one day (00:00-23:59) for each room
- Case ID: `Room_YYYY-MM-DD` (e.g., `Kitchen_2017-03-15`)
- Captures daily routine patterns
- Total cases: 6 rooms × 89 days = 534 cases

**Option 2: Event-Driven Episodes**
- One case = one continuous activity episode (e.g., shower, cooking)
- Case ID: `Room_Activity_Timestamp` (e.g., `Bathroom_Shower_2017-03-15_07:30`)
- Captures specific behavior patterns
- Variable case count (depends on event extraction)

**Option 3: Setpoint-Driven Periods**
- One case = one setpoint schedule period
- Case ID: `Room_Period_ID`
- Aligns with HVAC control logic
- ~340-360 cases per room (low granularity)

**Recommended**: **Option 1 (Daily Cycles)** - provides consistent case boundaries, aligns with human routines, sufficient granularity for pattern discovery.

### 4.4 Temporal Patterns and Seasonality

**Seasonal Effect (March → June)**:
- **Outdoor Temperature**: Gradual increase from ~6°C (March) to ~20°C (June)
- **Heating Demand**: Decreases over time (more heating in March, minimal in June)
- **Daylight Hours**: Increase (earlier sunrise, later sunset)
- **Humidity**: Varies with outdoor weather, rainfall events

**Weekly Patterns**:
- Weekday vs. Weekend differences (occupancy patterns)
- Potential work-from-home vs. office days

**Daily Patterns**:
- Morning peak (6-9 AM): Heating, cooking, bathing
- Midday trough (10 AM-4 PM): Lower activity, many rooms vacant
- Evening peak (5-10 PM): Return home, cooking, entertainment
- Night plateau (10 PM-6 AM): Sleep mode, minimal activity

## 5. Data Quality Issues (Potential and Realistic)

### 5.1 Inherent/Natural Quality Challenges

| Issue Type | Manifestation | Source | Impact |
|------------|---------------|--------|--------|
| **Irregular Sampling** | Non-uniform timestamps | Event-driven sensors | Complicates time-series analysis, aggregation |
| **Multi-Rate Sensors** | Different implicit frequencies | Sensor design | Synchronization challenges |
| **Sensor Drift** | Gradual calibration offset | Long-term deployment (89 days) | Temperature/humidity bias |
| **Network Latency** | Delayed transmission | Wireless EnOcean protocol | Timestamp may not reflect measurement time |
| **Clock Drift** | Time synchronization errors | Distributed sensor clocks | Events from different sensors not aligned |
| **Weather Data Lag** | Outdoor temp updated hourly | Virtual sensor (web service) | Lags actual outdoor conditions |
| **Duplicate Data** | Outdoor temp in 6 files | Data organization choice | Redundancy, potential inconsistencies |

### 5.2 Realistic Injectable Quality Issues

For explainability pipeline testing, the following issues are **realistic and relevant**:

#### **1. Sensor Failure Patterns**
- **Battery Depletion**: EnOcean sensors can lose energy harvesting → increasing sampling gaps
- **Communication Dropout**: Wireless signal loss → missing data intervals
- **Complete Failure**: Sensor stops reporting → prolonged gaps
- **Stuck Sensor**: Reports constant value (last known value repeated)

#### **2. Calibration Issues**
- **Zero-Point Drift**: Temperature offset (e.g., +2°C systematic error)
- **Scaling Error**: Multiplicative bias (e.g., humidity reads 1.1× actual)
- **Hysteresis**: Sensor responds differently to increasing vs. decreasing values
- **Saturation**: Sensor hits min/max range (e.g., brightness capped at 60,000 lux)

#### **3. Environmental Interference**
- **Radiator Heat Bias**: Thermostat sensor reads higher than room air (placement issue)
- **Direct Sunlight**: Brightness sensor saturated, temperature sensor biased
- **Ventilation Impact**: Open windows cause rapid, realistic temperature/humidity drops
- **Occupant Behavior**: Manual override of TRV (setpoint not followed)

#### **4. Data Collection Issues**
- **Timestamp Errors**: Clock reset, NTP sync failures
- **Precision Loss**: Rounding errors in temperature/humidity
- **Data Truncation**: File corruption, incomplete records
- **Outliers**: Electromagnetic interference spikes

#### **5. Inter-Sensor Inconsistencies**
- **Spatial Heterogeneity**: Wall temp ≠ thermostat temp (expected), but excessive difference indicates issue
- **Conflicting Readings**: Humidity rising while temperature dropping (physically implausible under closed conditions)
- **Causality Violations**: Heating starts before setpoint change recorded

### 5.3 Propagation Impact on Process Mining

**Quality Issue → Event Extraction → Process Discovery Chain**:

1. **Sensor Drift** → Incorrect event detection (false "Heating_Start") → Spurious activities in process model
2. **Missing Data** → Events not detected → Incomplete traces, lower conformance
3. **Timestamp Errors** → Event ordering violations → Impossible process variants
4. **Outliers** → False alarms ("High_Humidity_Alert" when none occurred) → Noisy process model
5. **Calibration Bias** → Systematic shift in event patterns → Incorrect process timing, performance metrics

**Example Propagation Scenario**:
```
Ground Truth: Kitchen cooking event (17:30-18:00)
    ↓
Quality Issue: Humidity sensor drift (-10% bias)
    ↓
Event Extraction: "Humidity_Spike" not detected (below threshold)
    ↓
Process Mining: "Cooking_Event" not recorded in trace
    ↓
Insight: Incorrectly conclude "Kitchen unused in evening" → False maintenance alert
    ↓
Business Impact: Wasted energy optimization opportunity, user dissatisfaction
```

## 6. Use Cases for Data Quality Framework

### 6.1 Smart Home Process Mining Applications

This dataset supports multiple **process mining use cases**:

1. **Energy Optimization**: Discover inefficient heating patterns (heating while room vacant)
2. **Occupancy Detection**: Infer room usage from sensor combinations
3. **Routine Discovery**: Learn typical daily/weekly activity patterns
4. **Anomaly Detection**: Detect unusual behavior (e.g., extended high humidity → leak?)
5. **Predictive Maintenance**: Forecast HVAC component failures (TRV stuck, sensor drift)
6. **User Comfort Analysis**: Assess if setpoints maintain comfort (temp ≈ setpoint?)
7. **Multi-Room Coordination**: Discover dependencies (heating Room 3 affects Room 2?)

### 6.2 Data Quality Explainability Scenarios

**Scenario 1: Drift Detection & Impact**
- **Inject**: Gradual temperature drift in Bathroom sensor (+0.5°C over 30 days)
- **Detect**: Quality detector identifies systematic bias via comparison with thermostat sensor
- **Propagate**: False "Overheating" events generated → incorrect process variant
- **Explain**: "Overheating alerts in Bathroom caused by uncalibrated wall temperature sensor, not actual overheating. Thermostat sensor shows normal values."

**Scenario 2: Missing Data & Trace Incompleteness**
- **Inject**: 20% of brightness readings dropped (simulated wireless packet loss)
- **Detect**: Quality detector identifies irregular sampling gaps exceeding normal variance
- **Propagate**: "Light_On"/"Light_Off" events not detected → incomplete occupancy inference
- **Explain**: "Occupancy estimate unreliable for Room 1 due to intermittent brightness sensor connectivity. Recommend using temperature+humidity as backup indicators."

**Scenario 3: Outlier Injection & False Alarms**
- **Inject**: Random humidity spikes (10% for 5 minutes) in Toilet
- **Detect**: Statistical outlier detection (z-score > 3)
- **Propagate**: False "Shower_Event" in Toilet (physically implausible)
- **Explain**: "Shower events in Toilet are artifacts of sensor malfunction. Toilet has no shower; high humidity spikes inconsistent with room function and lack correlated temperature increase."

**Scenario 4: Synchronization Issues**
- **Inject**: 10-minute timestamp shift in Kitchen humidity sensor
- **Detect**: Causality check identifies humidity change before cooking temperature rise
- **Propagate**: "Cooking_Event" appears to start with humidity (incorrect sequence)
- **Explain**: "Cooking event sequence incorrect due to timestamp desynchronization. Humidity change should follow temperature rise during cooking, not precede."

### 6.3 Ground Truth for Validation

**Advantages for Explainability Testing**:

✅ **Semantic Model**: Provides spatial relationships, sensor specifications → validate physical plausibility  
✅ **Multi-Sensor Redundancy**: Temperature (wall) vs. thermostat temp → cross-validation  
✅ **Outdoor Context**: Weather data → validate indoor-outdoor relationships  
✅ **Long Duration**: 89 days → detect slow drift, seasonal effects  
✅ **Functional Rooms**: Known purposes (Kitchen, Bathroom) → validate activity inference  
✅ **Schedule Data**: Setpoint changes → ground truth for heating control evaluation  

**Validation Strategies**:

1. **Physical Plausibility Checks**:
   - Temperature ∈ [0°C, 40°C] (sensor range)
   - Humidity ∈ [0%, 100%]
   - Brightness ≥ 0 lux
   - Thermostat temp ≥ wall temp (radiator warmer than room)
   - Indoor temp > outdoor temp during heating season

2. **Temporal Causality**:
   - Setpoint change precedes temperature change
   - Heating precedes target reached
   - Humidity spike precedes ventilation (bathroom shower sequence)

3. **Spatial Consistency**:
   - Adjacent rooms have correlated temperature (heat transfer)
   - Outdoor temperature affects all rooms similarly

4. **Functional Consistency**:
   - Bathroom has higher humidity variance than bedroom
   - Kitchen has humidity spikes at meal times
   - Bedrooms have lower brightness at night

## 7. Recommended Pipeline Approach

### 7.1 Phase 1: Data Preparation & Understanding

**1.1 Load and Parse Data**
- Read 37 CSV files (handle tab-separation, no headers)
- Parse UNIX timestamps → datetime objects
- Assign metadata (room, sensor type) from filenames

**1.2 Data Consolidation**
- Merge outdoor temperature duplicates (keep one reference)
- Create multi-sensor DataFrame per room
- Resample to common time grid (e.g., 10-minute intervals) using interpolation

**1.3 Semantic Model Integration**
- Parse RDF/Turtle files (`00_OpenSmartHomeData.ttl`, `01_LinkOsh.ttl`)
- Extract sensor-space-property relationships
- Load building geometry (optional: parse IFC for 3D visualization)

**1.4 Exploratory Analysis**
- Visualize typical daily patterns per room/sensor
- Correlation heatmaps (which sensors co-vary?)
- Identify outliers, gaps, anomalies in raw data
- Seasonal trend analysis (March vs. June)

### 7.2 Phase 2: Event Abstraction & Process Transformation

**2.1 Define Event Detection Rules**

Implement threshold-based, pattern-based, and composite detectors:

```python
# Example pseudo-code
def detect_heating_event(temp, setpoint, threshold=0.5, window='30min'):
    heating_start = (temp.diff() > threshold) & (temp < setpoint - threshold)
    return heating_start

def detect_shower_event(humidity, temp, brightness, room='Bathroom'):
    spike = (humidity.diff() > 10) within 15 minutes
    temp_rise = (temp.diff() > 1.0) within 15 minutes
    light_on = (brightness > 10)
    return spike & temp_rise & light_on
```

**2.2 Extract Events per Room**
- Apply detection rules to each room's time-series
- Generate event tuples: `(timestamp, room, activity, attributes)`
- Validate events (remove physically implausible patterns)

**2.3 Build Event Log**
- Group events into daily cases: `case_id = f"{room}_{date}"`
- Order events chronologically within case
- Add attributes:
  - `room`: Kitchen, Bathroom, etc.
  - `setpoint`: Current target temperature
  - `outdoor_temp`: Weather context
  - `season`: March/April/May/June
  - `weekday`: Monday-Sunday
  - `hour`: Time-of-day bucket

**2.4 Export to XES/CSV**
- Format: PM4Py-compatible event log
- Include case attributes (room, date, average outdoor temp)
- Include event attributes (sensor values at event time)

### 7.3 Phase 3: Quality Issue Injection

**3.1 Stratified Sampling**
- Select representative days (weekday/weekend, different months)
- Select representative rooms (high/low activity, different functions)

**3.2 Inject Realistic Issues**

| Issue Type | Parameters | Affected Sensors | Rooms |
|------------|-----------|------------------|-------|
| **Sensor Drift** | Linear: +0.5°C over 30 days | Temperature (Bathroom) | 1 room |
| **Missing Data** | 20% random dropout | Brightness (Room 1) | 1 room |
| **Outliers** | 10 spikes, 10% magnitude, 5 min duration | Humidity (Toilet) | 1 room |
| **Clock Drift** | +10 min offset after day 30 | Humidity (Kitchen) | 1 room |
| **Stuck Sensor** | Repeat last value for 2 days | Thermostat temp (Room 2) | 1 room |
| **Calibration Bias** | ×1.15 multiplier | Humidity (Room 3) | 1 room |

**3.3 Document Ground Truth**
- JSON manifest: `{issue_type, sensor, room, start_time, end_time, parameters}`
- Enable precision/recall evaluation of quality detectors

### 7.4 Phase 4: Quality Detection & Propagation

**4.1 Apply Quality Detectors**
- **Drift Detector**: Compare wall temp vs. thermostat temp (should be similar long-term)
- **Missing Data Detector**: Identify sampling gaps > 2× median interval
- **Outlier Detector**: Statistical methods (z-score, IQR, Isolation Forest)
- **Synchronization Checker**: Cross-sensor causality validation
- **Range Validator**: Check sensor values within specification ranges

**4.2 Measure Detection Performance**
- True Positive Rate: Correct detections / actual issues
- False Positive Rate: False alarms / clean data points
- Localization Accuracy: Temporal precision of detection

**4.3 Trace Propagation Analysis**
- **Before Injection**: Extract events, build process model
- **After Injection**: Extract events (with quality issues), build process model
- **Compare**: Which events changed? New activities? Missing activities?
- **Quantify**: % of traces affected, % of events altered, process model divergence metrics

### 7.5 Phase 5: Explainability & Insights

**5.1 Generate Explanations**

For each detected quality issue:
- **What**: "Bathroom temperature sensor exhibits +0.5°C drift over March-April"
- **Where**: "Sensor: Bathroom-temp-Sensor, Room: Bathroom, Timespan: 2017-03-09 to 2017-04-08"
- **How Detected**: "Cross-validation with thermostat sensor reveals systematic offset exceeding 0.5°C"
- **Impact on Events**: "Generated 15 false 'Overheating' events; heating control incorrectly reduced"
- **Impact on Process**: "Bathroom daily routine appears to have 'Overheating_Response' activity not present in other rooms or previous weeks"
- **Recommendation**: "Recalibrate Bathroom temperature sensor; filter affected events; recompute process model"

**5.2 Validate Physical Plausibility**

Check explanations against domain knowledge:
- Does drift direction make sense? (Sensors typically drift positive with age/dust)
- Do event changes align with issue type? (Missing data → missing events)
- Are spatial relationships respected? (Drift in one room shouldn't affect distant room directly)
- Is timing consistent? (Stuck sensor → repeated values → prolonged single-state events)

**5.3 Produce Actionable Insights**

**For Building Managers**:
- "Bathroom temperature sensor requires recalibration (detected drift: +0.5°C)"
- "Room 1 brightness sensor experiencing connectivity issues (20% packet loss)"
- "Toilet humidity sensor malfunctioning (spurious spikes, no correlated events)"

**For Data Scientists**:
- "Filter events in Bathroom traces from 2017-03-09 to 2017-04-08 before process discovery"
- "Use humidity+temperature composite for occupancy inference; brightness unreliable in Room 1"
- "Synchronization issue in Kitchen: apply -10 min offset to humidity timestamps"

**For Process Analysts**:
- "Apparent 'Overheating' activity in Bathroom is data artifact, not process variant"
- "Missing 'Light_On' events in Room 1 due to sensor, not actual behavior change"
- "Toilet 'Shower_Event' physically implausible (no shower in toilet); disregard"

## 8. Expected Challenges

### 8.1 Technical Challenges

1. **Irregular Sampling**: Time-series algorithms expect regular intervals; requires resampling/interpolation
2. **Multi-Resolution Synchronization**: Aligning sparse setpoint changes with frequent brightness readings
3. **Event Detection Ambiguity**: Threshold selection (when is humidity "spike" vs. gradual increase?)
4. **Semantic Model Parsing**: RDF/Turtle parsing, SPARQL queries for relationship extraction
5. **Large File Handling**: 37 files, ~285K records → memory management
6. **Duplicate Data**: Managing 6 copies of outdoor temperature, ensuring consistency

### 8.2 Domain-Specific Challenges

1. **Physical Coupling**: Rooms are not independent (heat transfer through walls)
2. **Lag Effects**: Heating control has delay (radiator takes minutes to heat room)
3. **Occupant Behavior**: Manual overrides, unpredictable actions (windows, doors)
4. **Sensor Placement**: Thermostat on radiator ≠ room center (systematic difference)
5. **Seasonal Confounding**: Outdoor temp increase over 89 days affects all patterns
6. **Weather Events**: Rainy days, cold snaps → sudden pattern changes

### 8.3 Process Mining Challenges

1. **Activity Granularity**: How detailed should events be? (e.g., "Heating" vs. "Heating_Ramp_Up", "Heating_Steady_State")
2. **Case Boundary Definition**: Daily cases may split multi-hour activities (long cooking session 11 PM → 1 AM)
3. **Attribute Selection**: Which sensor values to include as event attributes? (all? summary stats?)
4. **Process Complexity**: 6 rooms × many events = large, complex process model
5. **Temporal Constraints**: Some activities have duration (heating takes 30 min), not instant
6. **Parallelism**: Multiple rooms active simultaneously (conformance checking challenge)

### 8.4 Explainability Challenges

1. **Attribution**: Which sensor and which time window caused event detection failure?
2. **Propagation Tracking**: Quality issue in sensor X → affects event Y → changes trace Z → alters process model M (multi-step chain)
3. **Counterfactual Reasoning**: "What would process have looked like without quality issue?"
4. **Human Interpretation**: Explanations must be understandable to non-technical building managers
5. **Validation**: How to prove explanation is correct? (No ground truth for real-world process)

## 9. Key Differences from Other Datasets

### 9.1 Comparison with Hydraulic Systems Dataset

| Aspect | Hydraulic Systems | Smart Home IoT |
|--------|-------------------|----------------|
| **Domain** | Industrial manufacturing | Residential building |
| **Process Type** | Fixed 60s work cycles | Open-ended daily living |
| **Sampling** | Fixed rate (1/10/100 Hz) | Irregular, event-driven |
| **Sensor Count** | 17 sensors, 1 system | 37 time-series, 6 rooms (distributed) |
| **Spatial Distribution** | Single test rig | Multi-room building |
| **Labels** | Ground truth degradation states | No explicit labels (infer from patterns) |
| **Duration** | 2,205 cycles (~30 hours total?) | 89 days continuous |
| **Process Variants** | Single fixed process | Multiple daily routines |
| **Quality Issues** | Sensor drift focus | Diverse (network, calibration, missing data) |
| **Semantic Model** | None | Rich (BOT, SOSA, DogOnt, IFC/BIM) |
| **Explainability Target** | Component health classification | Activity inference, energy optimization |

**Key Insight**: Smart Home dataset is **spatially distributed** and **long-term**, enabling **spatial propagation** and **temporal drift** studies not possible with Hydraulic Systems data.

### 9.2 Advantages for Explainability Pipeline

✅ **Semantic richness**: BOT/SOSA model enables reasoning about spatial relationships, sensor types, physical constraints  
✅ **Long duration**: 89 days sufficient for slow drift, seasonal effects, long-term patterns  
✅ **Multi-room parallelism**: Test cross-sensor, cross-room propagation  
✅ **Realistic IoT characteristics**: Irregular sampling, wireless communication issues, energy-harvesting sensors  
✅ **Functional diversity**: Different room types (kitchen, bathroom, bedroom) → varied process patterns  
✅ **Actuator feedback**: Setpoint changes provide control loop context (closed-loop system)  
✅ **External context**: Outdoor temperature enables indoor-outdoor relationship modeling  

### 9.3 Limitations

⚠️ **No explicit labels**: Must infer activities from sensor patterns (no ground truth process model)  
⚠️ **Single building**: Cannot generalize across buildings, occupancy patterns  
⚠️ **No occupancy ground truth**: Cannot validate room occupancy inferences  
⚠️ **Limited fault data**: No real sensor faults recorded (must inject artificial issues)  
⚠️ **Complex causality**: Many confounding factors (weather, occupant, HVAC, building physics)  

## 10. Summary and Recommendations

### 10.1 Dataset Strengths

✅ **Rich semantic metadata**: W3C/OGC standard ontologies (BOT, SOSA, SSN)  
✅ **Spatial context**: Multi-room distribution, building geometry (IFC/Revit)  
✅ **Long-term monitoring**: 89 days, ~285K readings  
✅ **Multi-sensor fusion**: 6 sensor types per room  
✅ **Realistic IoT data**: Irregular sampling, wireless protocol  
✅ **Closed-loop system**: Actuators (TRVs) + sensors + schedule  
✅ **External context**: Outdoor weather data  
✅ **Diverse room functions**: Kitchen, bathroom, bedroom, living room → varied processes  
✅ **Open data**: CC BY-SA 4.0 license, published with DOI  
✅ **Reproducible**: Code and documentation available (GitHub repo)  

### 10.2 Dataset Limitations

⚠️ **No activity labels**: Must infer events from sensor patterns  
⚠️ **Irregular sampling**: Complicates time-series analysis  
⚠️ **Duplicate outdoor data**: Same weather data in 6 files  
⚠️ **Limited real faults**: Must inject artificial quality issues  
⚠️ **Single instance**: One building, one flat, one family  
⚠️ **Semantic complexity**: Requires RDF/ontology expertise  

### 10.3 Suitability for Explainability Pipeline

**Overall Assessment**: ⭐⭐⭐⭐⭐ (Excellent - Best Available for IoT Process Mining)

**Ideal for**:
- IoT sensor-to-event abstraction methodology
- Spatial data quality propagation (multi-room)
- Semantic reasoning with building ontologies
- Long-term drift detection and impact analysis
- Real-world irregular sampling challenges
- Multi-sensor fusion under quality degradation
- Explainability in smart home/building domain

**Best Practices**:
1. Start with single room (e.g., Bathroom) for prototype
2. Use daily cases for consistent trace boundaries
3. Leverage semantic model for validation (physical plausibility)
4. Inject diverse quality issues (drift, missing, outliers, sync)
5. Validate explanations against domain knowledge (building physics)
6. Document transformation methodology (sensor → event → trace)

### 10.4 Recommended Next Steps

**Phase 1: Data Preparation** (Week 1)
1. Load and parse 37 CSV files → unified DataFrame
2. Parse semantic model (RDF/Turtle) → sensor-space relationships
3. Exploratory analysis → typical patterns, correlations, anomalies
4. Resample to 10-minute grid → handle irregular sampling

**Phase 2: Event Abstraction Prototype** (Week 2)
1. Implement 5-10 event detectors (heating, humidity spike, light on/off, etc.)
2. Test on single room (Bathroom) for 1 week of data
3. Validate events manually (do they make sense?)
4. Refine thresholds and window parameters

**Phase 3: Process Log Generation** (Week 3)
1. Apply event extraction to all rooms, full 89 days
2. Build daily case structure → event log
3. Export to XES format → load in PM4Py
4. Discover initial process model (Directly-Follows Graph, Petri Net)

**Phase 4: Quality Issue Injection** (Week 4)
1. Select 6 test scenarios (one per room, different issue types)
2. Inject quality issues into raw time-series data
3. Document ground truth (what, where, when, parameters)
4. Re-run event extraction on corrupted data

**Phase 5: Detection & Propagation** (Week 5)
1. Implement quality detectors (drift, missing, outliers, sync)
2. Measure detection performance (precision, recall)
3. Compare process models: clean vs. corrupted
4. Quantify propagation (event changes, trace changes, model divergence)

**Phase 6: Explainability & Validation** (Week 6)
1. Generate explanations for detected issues
2. Trace propagation chains (sensor → event → trace → model)
3. Validate physical plausibility (building physics checks)
4. Produce actionable insights (recommendations)
5. Document methodology and results

## 11. Conclusion

The **Open Smart Home IoT-IEQ-Energy Data** dataset is an **exceptional testbed** for developing and validating the explainability pipeline for IoT-enhanced process mining. Its combination of:

- **Rich semantic metadata** (ontologies, building models)
- **Realistic IoT characteristics** (irregular sampling, wireless, distributed)
- **Spatial distribution** (multi-room, building-scale)
- **Long-term coverage** (89 days, seasonal effects)
- **Diverse sensor modalities** (temperature, humidity, brightness, setpoints)
- **Closed-loop control** (HVAC actuation feedback)

...makes it **ideal** for studying how data quality issues propagate through the transformation pipeline from raw IoT sensors to process mining insights.

The dataset enables validation of core research questions:
1. **Can we detect quality issues in irregular IoT time-series?** (drift, missing, outliers)
2. **How do quality issues affect event extraction?** (false positives, false negatives)
3. **How do corrupted events alter process models?** (spurious activities, missing paths)
4. **Can we generate explainable, actionable recommendations?** (physical plausibility, domain validation)

Compared to the Hydraulic Systems dataset, the Smart Home data offers **greater complexity** (spatial distribution, process variety, irregular sampling) while maintaining **semantic grounding** (ontologies, building models) essential for explainability validation.

**Next step**: Implement data loading and event abstraction prototype for Bathroom (simplest room, clear shower/humidity pattern) to validate methodology before scaling to full 6-room, 89-day dataset.

---

**Dataset Citation**:  
Schneider, G. F., Rasmussen, M. H., Bonsma, P., Oraskari, J., & Pauwels, P. (2018). *Linked building data for modular building information modelling of a smart home*. In 11th European Conference on Product and Process Modelling (pp. 407-414). CRC Press.

**DOI**: [![DOI](https://zenodo.org/badge/120334357.svg)](https://zenodo.org/badge/latestdoi/120334357)

**GitHub Repository**: [TechnicalBuildingSystems/OpenSmartHomeData](https://github.com/TechnicalBuildingSystems/OpenSmartHomeData)

**Analysis Date**: November 19, 2025  
**Analysis Version**: 1.0
