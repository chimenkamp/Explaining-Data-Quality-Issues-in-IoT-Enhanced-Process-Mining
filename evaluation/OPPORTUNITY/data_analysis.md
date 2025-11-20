# OPPORTUNITY Activity Recognition Dataset - Dataset Analysis

## Dataset Overview

**Source**: OPPORTUNITY Project Consortium (ETH Zurich, EPFL, University of Passau, JKU Linz)  
**Domain**: Human Activity Recognition / Ambient Assisted Living  
**Dataset Type**: Multivariate Time-Series for Wearable Sensor-Based Activity Recognition  
**Task**: Multi-level Activity Classification (Locomotion, Gestures, High-level Activities)  
**Sampling Rate**: 30 Hz  
**Number of Subjects**: 4 users  
**Total Recordings**: 24 runs (4 subjects × 6 runs each)  
**Recording Environment**: Simulated studio flat with kitchen setup  
**Total Sensor Channels**: 250 columns (243 sensor channels + 7 label tracks)

## 1. Physical System Description

### 1.1 Recording Scenario

The OPPORTUNITY dataset captures naturalistic human activities in a sensor-rich environment designed to simulate Activities of Daily Living (ADL). The recording took place in a room configured as a studio flat with:

- **Kitchen area**: Fully equipped with instrumented appliances (fridge, dishwasher, drawers, coffee machine)
- **Living area**: Deckchair (lazy chair) for relaxation
- **Access doors**: Two doors (Door1, Door2) providing access to outside
- **Furniture**: Table, chairs, counters
- **Objects**: Instrumented everyday items (cups, plates, knives, food items)

### 1.2 Recording Protocol

Each subject performed **6 runs**:

1. **ADL Runs (5 per subject)**: Activity of Daily Living sequences
   - Natural, free-form execution of high-level scenarios
   - Users encouraged to introduce variability and personal style
   - Temporally unfolding situations (prepare coffee → drink coffee → prepare sandwich, etc.)
   
2. **Drill Run (1 per subject)**: Scripted repetitive sequence
   - 20 repetitions of fixed activity sequence
   - Designed to generate many labeled gesture instances
   - More consistent execution with less variability

### 1.3 ADL Run Scenario Structure

The ADL runs follow a high-level scenario with natural temporal progression:

1. **Start**: Lying on deckchair, get up
2. **Groom**: Move around, check object placements in drawers/shelves
3. **Relax**: Walk outside around the building
4. **Prepare Coffee**: Use coffee machine, add milk and sugar
5. **Drink Coffee**: Take coffee sips while moving around
6. **Prepare Sandwich**: Use bread cutter, knives, prepare bread/cheese/salami
7. **Eat Sandwich**: Consume sandwich
8. **Cleanup**: Return objects to original locations or dishwasher, clean table
9. **Break**: Return to deckchair

### 1.4 Drill Run Sequence

The drill run consists of 20 repetitions of:

1. Open then close fridge
2. Open then close dishwasher
3. Open then close 3 drawers (upper, middle, lower)
4. Open then close Door 1
5. Open then close Door 2
6. Toggle lights on then off
7. Clean the table
8. Drink while standing
9. Drink while seated

## 2. Sensor Configuration and Data Structure

### 2.1 Body-Worn Sensors (72 channels)

#### Inertial Measurement Units (IMUs) - 7 sensors × 13 channels = 91 channels
- **Locations**: BACK, RUA (Right Upper Arm), RLA (Right Lower Arm), LUA (Left Upper Arm), LLA (Left Lower Arm), L-SHOE, R-SHOE
- **Measurements per IMU**:
  - 3D Acceleration (accX, accY, accZ)
  - 3D Gyroscope (gyroX, gyroY, gyroZ)
  - 3D Magnetic field (magneticX, magneticY, magneticZ)
  - 4D Quaternion orientation (Quaternion1-4)
  
**Note**: Upper body IMUs (BACK, RUA, RLA, LUA, LLA) mounted in specialized jacket for reproducible placement. Shoe IMUs provide additional locomotion data.

#### 3D Acceleration Sensors - 12 sensors × 3 axes = 36 channels
- **Locations**: RKN^ (Right Knee), HIP, LUA^, RUA_, LH (Left Hand), BACK, RKN_, RWR (Right Wrist), RUA^, LUA_, LWR (Left Wrist), RH (Right Hand)
- **Characteristics**: Wireless sensors, expected data loss due to transmission issues
- **Sampling**: 3D acceleration (milli-g units)

#### Localization System - 4 tags × 3 coordinates = 12 channels
- **Tags**: TAG1, TAG2, TAG3, TAG4 (placed on left/right front/back shoulders)
- **Output**: 3D coordinates (X, Y, Z in millimeters) in room coordinate system
- **Quality**: Noted as extremely noisy, use with caution

**Total Body-Worn Channels**: Columns 2-243 (242 channels)

### 2.2 Object Sensors (60 channels)

12 instrumented objects with wireless sensors:

| Object | Sensor Type | Channels | Usage Context |
|--------|-------------|----------|---------------|
| **CUP** | 3D Acc + 2D Gyro | 5 | Coffee/drinking activities |
| **SALAMI** | 3D Acc + 2D Gyro | 5 | Sandwich preparation |
| **WATER/MILK** | 3D Acc + 2D Gyro | 5 each | Beverage preparation |
| **CHEESE/BREAD** | 3D Acc + 2D Gyro | 5 each | Sandwich ingredients |
| **KNIFE1/KNIFE2** | 3D Acc + 2D Gyro | 5 each | Cutting tools |
| **SPOON** | 3D Acc + 2D Gyro | 5 | Stirring coffee/sugar |
| **SUGAR** | 3D Acc + 2D Gyro | 5 | Coffee preparation |
| **PLATE/GLASS** | 3D Acc + 2D Gyro | 5 each | Serving/drinking |

**Channels**: 135-194 (60 channels)  
**Characteristics**: Wireless transmission, higher data loss especially when inside closed dishwasher

### 2.3 Ambient Sensors (38 channels)

#### Reed Switches - 13 binary sensors
- **DISHWASHER**: S1, S2, S3 (triplet for open/half-open/closed detection)
- **FRIDGE**: S1, S2, S3 (triplet configuration)
- **MIDDLE DRAWER**: S1, S2, S3 (triplet configuration)
- **LOWER DRAWER**: S1, S2, S3 (triplet configuration)
- **UPPER DRAWER**: Single switch (fires for all positions)
- **Purpose**: Binary state detection (0/1), multiple switches per furniture enable three-state detection (closed/half-open/fully open)

**Channels**: 195-207 (13 channels)

#### 3D Acceleration Sensors on Furniture - 8 sensors × 3 axes = 24 channels
- **Locations**: DOOR1, DOOR2, LAZYCHAIR, DISHWASHER, UPPERDRAWER, LOWERDRAWER, MIDDLEDRAWER, FRIDGE
- **Purpose**: Detect usage patterns, opening/closing dynamics
- **Advantage**: Wired system with minimal data loss

**Channels**: 208-231 (24 channels)

### 2.4 Data File Format

- **Format**: Space-delimited text matrices (.dat files)
- **Structure**: One line per sample (30 Hz sampling rate)
- **Columns**: 250 total
  - Column 1: Timestamp (MILLISEC)
  - Columns 2-243: Sensor channels
  - Columns 244-250: Label tracks (7 annotation layers)
- **Missing Data**: Indicated by "NaN" (Not-a-Number)
- **File Naming**: `S<subject>-ADL<run>.dat` or `S<subject>-Drill.dat`
  - Example: `S1-ADL1.dat`, `S2-Drill.dat`

### 2.5 Data Quality Characteristics

#### High-Quality Sensors (Minimal Data Loss)
- **IMUs (body-worn)**: Wired system, reproducible placement via jacket
- **Reed switches**: Wired, binary signals
- **Ambient accelerometers**: Wired system

#### Moderate Data Loss Expected
- **3D acceleration sensors (body-worn)**: Wireless, some packet loss
- **Object sensors**: Wireless, variable loss depending on location
- **Critical issue**: Objects in closed dishwasher experience significant occlusion

#### Noisy/Unreliable
- **Localization tags**: Extremely noisy 3D positioning, use with caution

## 3. Label Structure and Activity Annotations

### 3.1 Label Columns (Columns 244-250)

The dataset provides **5 parallel annotation tracks** at different granularity levels:

| Column | Track Name | Level | Classes | Description |
|--------|------------|-------|---------|-------------|
| **244** | Locomotion | Low-level | 4 | Body posture/movement mode |
| **245** | HL_Activity | High-level | 5 | Overall activity situation |
| **246** | LL_Left_Arm | Low-level | 13 | Left arm action verb |
| **247** | LL_Left_Arm_Object | Low-level | 23 | Object manipulated by left arm |
| **248** | LL_Right_Arm | Low-level | 13 | Right arm action verb |
| **249** | LL_Right_Arm_Object | Low-level | 23 | Object manipulated by right arm |
| **250** | ML_Both_Arms | Mid-level | 17 | Combined gesture (recommended) |

### 3.2 Locomotion Labels (Column 244)

| Label ID | Activity | Description |
|----------|----------|-------------|
| **1** | Stand | Standing posture |
| **2** | Walk | Walking/moving |
| **4** | Sit | Seated position |
| **5** | Lie | Lying on deckchair |

**Instances across all subjects/runs**: 3,653 labeled segments  
**Challenge**: Short gait sequences, sometimes single steps, making boundary detection difficult

### 3.3 High-Level Activities (Column 245)

| Label ID | Activity | Related ADL Scenario Steps |
|----------|----------|---------------------------|
| **101** | Relaxing | Start (lying on deckchair), Break |
| **102** | Coffee time | Prepare coffee, Drink coffee |
| **103** | Early morning | Groom, Relax (outside walk) |
| **104** | Cleanup | Return objects, clean table |
| **105** | Sandwich time | Prepare sandwich, Eat sandwich |

**Characteristics**:
- Smoothly transitioning boundaries (e.g., finishing sandwich while starting cleanup)
- Cannot be pinpointed to exact timestamps
- Reflects natural activity interleaving
- Most realistic but hardest to annotate precisely

### 3.4 Low-Level Actions (Columns 246-249)

#### Action Verbs (13 types)
| Label ID Range | Action | Example Usage |
|----------------|--------|---------------|
| **201-213** (Left Arm) | unlock, stir, lock, close, reach, open, sip, clean, bite, cut, spread, release, move | Fine-grained arm movements |
| **401-413** (Right Arm) | (same verbs) | Mirror of left arm actions |

#### Manipulated Objects (23 types)
| Label ID Range | Objects |
|----------------|---------|
| **301-323** (Left Arm) | Bottle, Salami, Bread, Sugar, Dishwasher, Switch, Milk, Drawer3, Spoon, Knife cheese, Drawer2, Table, Glass, Cheese, Chair, Door1, Door2, Plate, Drawer1, Fridge, Cup, Knife salami, Lazychair |
| **501-523** (Right Arm) | (same objects) |

**Characteristics**:
- Extremely short duration (often <1 second)
- Sensitive to annotation jitter
- Users predominantly used right hand
- Provides rich semantic information when combined

### 3.5 Mid-Level Gestures (Column 250) - **RECOMMENDED**

Automatically generated from low-level annotations by combining action verbs with objects. Examples:

| Label ID | Gesture | Component Actions |
|----------|---------|------------------|
| **406516** | Open Door 1 | reach door1 + open door1 |
| **404516** | Close Door 1 | close door1 |
| **406520** | Open Fridge | reach fridge + open fridge |
| **404520** | Close Fridge | close fridge |
| **406505** | Open Dishwasher | reach dishwasher + open dishwasher |
| **404505** | Close Dishwasher | close dishwasher |
| **406519** | Open Drawer 1 | reach drawer + open drawer |
| **406511** | Open Drawer 2 | reach drawer + open drawer |
| **406508** | Open Drawer 3 | reach drawer + open drawer |
| **408512** | Clean Table | clean table |
| **407521** | Drink from Cup | sip cup |
| **405506** | Toggle Switch | reach switch + toggle |

**Instances across all subjects/runs**: 2,551 labeled segments  
**Advantages**:
- Longer duration than low-level actions
- Less sensitive to annotation jitter
- Combines semantically meaningful action-object pairs
- **Recommended for initial analysis**

### 3.6 Label Distribution and Instance Counts

- **Drill runs**: High instance count per gesture type (20 repetitions × 9 activities)
- **ADL runs**: Natural distribution reflecting realistic activity frequencies
- **Variability**: ADL runs show larger intra-class variability in execution style
- **Missing labels**: Some periods have no annotations (transitions, unscripted activities)

## 4. Process Mining Perspective

### 4.1 Process Notion for ADL Activities

The OPPORTUNITY dataset represents a **human-centric process** with multiple abstraction levels:

#### High-Level Process View (Morning Routine)
```
[Wake Up] → [Groom] → [Relax] → [Prepare Coffee] → [Drink Coffee] → 
[Prepare Sandwich] → [Eat Sandwich] → [Cleanup] → [Rest]
```

#### Mid-Level Gesture Process (Kitchen Activity Example)
```
[Approach Fridge] → [Open Fridge] → [Reach Object] → [Close Fridge] → 
[Walk to Table] → [Open Drawer] → [Reach Utensil] → [Close Drawer] → 
[Prepare Food] → [Clean Table]
```

#### Low-Level Action Process (Object Manipulation)
```
[Reach Bottle] → [Grasp Bottle] → [Move Bottle] → [Release Bottle] → 
[Reach Glass] → [Grasp Glass] → [Move Glass] → [Pour] → [Release Glass]
```

### 4.2 Event Abstraction Functions

To transform sensor data into process mining events, we define abstraction functions at multiple levels:

#### Level 1: Sensor Pattern Events (Low-Level)
**Abstraction Function**: Threshold-based detection on sensor streams

| Sensor Signal | Event Type | Detection Method |
|---------------|------------|------------------|
| Reed switch state change (0→1) | `Furniture_Open` | Binary transition |
| Reed switch state change (1→0) | `Furniture_Close` | Binary transition |
| Object accelerometer spike (>threshold) | `Object_Manipulated` | Magnitude threshold |
| IMU acceleration change | `Body_Movement_Start` | Derivative threshold |
| Stable IMU orientation | `Posture_Stable` | Low variance window |

**Example Events**:
- `Reed_Fridge_S1_Closed_To_Open` (timestamp: 1234ms)
- `Acc_Cup_Movement_Detected` (timestamp: 5678ms)
- `IMU_RightHand_Motion_Start` (timestamp: 9012ms)

#### Level 2: Gesture Events (Mid-Level) - **Primary for Process Mining**
**Abstraction Function**: Label-based extraction from ML_Both_Arms track

| Label ID | Event Name | Duration Type |
|----------|------------|---------------|
| 406520 | `Open_Fridge` | Interval (start, end) |
| 404520 | `Close_Fridge` | Interval (start, end) |
| 406516 | `Open_Door1` | Interval (start, end) |
| 407521 | `Drink_From_Cup` | Interval (start, end) |
| 408512 | `Clean_Table` | Interval (start, end) |

**Pseudo-code**:
```python
def extract_gesture_events(label_column_250, timestamps):
    events = []
    current_gesture = None
    gesture_start = None
    
    for i, (label, time) in enumerate(zip(label_column_250, timestamps)):
        if label != 0 and label != current_gesture:
            # New gesture starts
            if current_gesture is not None:
                events.append({
                    'activity': gesture_name(current_gesture),
                    'start': gesture_start,
                    'end': timestamps[i-1]
                })
            current_gesture = label
            gesture_start = time
        elif label == 0 and current_gesture is not None:
            # Gesture ends
            events.append({
                'activity': gesture_name(current_gesture),
                'start': gesture_start,
                'end': time
            })
            current_gesture = None
            gesture_start = None
    
    return events
```

#### Level 3: Activity Events (High-Level)
**Abstraction Function**: Extract from HL_Activity track (Column 245) with fuzzy boundaries

| Label ID | Event Name | Characteristics |
|----------|------------|-----------------|
| 101 | `Start_Relaxing_Phase` | Long duration, smooth start |
| 102 | `Start_Coffee_Time` | Medium duration |
| 103 | `Start_Early_Morning` | Medium duration |
| 104 | `Start_Cleanup` | Can overlap with sandwich end |
| 105 | `Start_Sandwich_Time` | Medium duration |

**Challenge**: Boundaries are inherently fuzzy due to natural activity interleaving

#### Level 4: Locomotion Events (Context)
**Abstraction Function**: State changes in Locomotion track (Column 244)

| Transition | Event Name | Process Context |
|------------|------------|-----------------|
| Any → 1 (Stand) | `Start_Standing` | Stationary activity phase |
| Any → 2 (Walk) | `Start_Walking` | Movement between locations |
| Any → 4 (Sit) | `Start_Sitting` | Seated activity phase |
| Any → 5 (Lie) | `Start_Lying` | Rest period |

### 4.3 Case/Trace Identification

#### Option A: Recording-Level Cases
- **Case ID**: `S<subject>_<run_type><run_number>`
  - Example: `S1_ADL1`, `S2_Drill`
- **Trace content**: Complete sequence of all gestures/activities in that recording
- **Duration**: Variable (ADL runs longer than drill runs)
- **Use case**: Analyze overall daily routine patterns

#### Option B: Activity-Level Cases
- **Case ID**: `S<subject>_<run>_HL<activity_id>_<instance>`
  - Example: `S1_ADL1_HL102_1` (first coffee time in S1-ADL1)
- **Trace content**: All gestures within one high-level activity
- **Duration**: 30s to several minutes
- **Use case**: Analyze how users perform specific activities (coffee preparation, sandwich making)

#### Option C: Gesture-Level Cases (Most Granular)
- **Case ID**: `S<subject>_<run>_Gesture_<sequence_number>`
- **Trace content**: Single gesture with context (preceding/following gestures)
- **Duration**: Few seconds
- **Use case**: Detailed gesture analysis, object manipulation patterns

### 4.4 Attribute Enrichment

Each event can be enriched with contextual attributes:

| Attribute Type | Source | Example Values |
|----------------|--------|----------------|
| **Subject** | Filename | S1, S2, S3, S4 |
| **Run Type** | Filename | ADL, Drill |
| **Run Number** | Filename | 1, 2, 3, 4, 5 |
| **Current Posture** | Locomotion label | Stand, Walk, Sit, Lie |
| **Body Part Active** | Sensor fusion | Left_Hand, Right_Hand, Both_Hands |
| **Object Involved** | Low-level labels | Fridge, Cup, Knife, Door1 |
| **Location** | Localization tags | Kitchen, Living_Area, Near_Door |
| **Movement Intensity** | IMU statistics | Low, Medium, High |
| **Concurrent Activities** | Multi-label | Walk+OpenDoor, Sit+DrinkCup |

### 4.5 Recommended Process Abstraction for Pipeline

**Primary Recommendation**: Use **Mid-Level Gestures (Column 250)** as main events

**Rationale**:
1. ✅ Semantically meaningful (action + object)
2. ✅ Sufficient duration to avoid annotation jitter
3. ✅ 2,551 instances provide good statistical basis
4. ✅ Automatically generated from low-level labels (consistency)
5. ✅ Recommended by dataset authors for initial analysis

**Event Log Schema**:
```
Case ID: S<subject>_<run>
Event Attributes:
  - activity: Gesture name (e.g., "Open_Fridge")
  - timestamp: Event start time (milliseconds)
  - duration: Event duration (milliseconds)
  - subject: S1/S2/S3/S4
  - run_type: ADL/Drill
  - posture: Current locomotion state
  - hand: Left/Right/Both
  - object: Manipulated object name
```

## 5. Data Quality Issues and Characteristics

### 5.1 Systematic Data Quality Issues

| Issue Type | Affected Sensors | Manifestation | Frequency | Mitigation Strategy |
|------------|------------------|---------------|-----------|---------------------|
| **Missing Data (NaN)** | Wireless sensors (Object sensors, body-worn 3D acc) | Packet loss, transmission failures | Common | Data imputation, sensor fusion |
| **Occlusion-Related Loss** | Object sensors in dishwasher | Extended NaN periods when dishwasher closed | Predictable | Context-aware handling |
| **Sensor Drift** | All analog sensors | Gradual offset over recording session | Slow | Calibration compensation |
| **Synchronization Jitter** | Cross-system sensors | Time alignment errors (~100ms) | Systematic | Timestamp buffer windows |
| **Localization Noise** | TAG1-TAG4 (UWB system) | Extremely noisy 3D coordinates | Severe | Avoid or use only for coarse location |
| **Annotation Jitter** | All labels | Boundary timing uncertainty (±200-500ms) | Systematic | Fuzzy temporal matching |

### 5.2 NaN (Not-a-Number) Patterns

**Expected NaN Distribution**:
- **Body-worn IMUs**: <1% data loss (wired system)
- **Body-worn 3D accelerometers**: 5-15% data loss (wireless)
- **Object sensors**: 10-30% data loss (wireless + occlusion)
- **Ambient sensors**: <1% data loss (wired)
- **Localization**: Present but extremely noisy (not missing)

**Critical NaN Scenario**: Objects placed in dishwasher during cleanup phase
- Duration: Several minutes
- Impact: Complete signal loss for those object sensors
- Detection: Correlates with Dishwasher reed switch closed state

### 5.3 Annotation Quality Considerations

#### Challenge 1: Temporal Jitter
- **Source**: Annotator reaction time, activity boundary ambiguity
- **Magnitude**: ±200-500 milliseconds
- **Impact**: Low-level actions most affected, mid-level gestures more robust
- **Implication**: Use temporal windows rather than exact timestamps

#### Challenge 2: Interleaved Activities
- **Source**: Natural human behavior (finishing one activity while starting next)
- **Manifestation**: Overlapping high-level activity labels
- **Example**: Finishing sandwich while already starting cleanup
- **Implication**: High-level boundaries are fuzzy by nature

#### Challenge 3: Short Activity Duration
- **Source**: Low-level actions are very brief (<1 second)
- **Manifestation**: Sensitive to frame-level timing errors
- **Mitigation**: Use mid-level gestures (longer duration)

#### Challenge 4: Subjective Segmentation
- **Source**: Different annotators might segment activities differently
- **Impact**: Jitter of several hundred milliseconds between observers
- **Implication**: Evaluation metrics should account for temporal tolerance

### 5.4 Inter-Subject Variability

| Variability Dimension | ADL Runs | Drill Runs | Impact |
|----------------------|----------|------------|--------|
| **Execution Speed** | High | Low | Gesture duration varies 2-3x |
| **Object Selection** | High | None | Different utensil/food choices |
| **Movement Style** | High | Low | Approach paths, handedness |
| **Activity Ordering** | Medium | None | Can perform sub-tasks in different order |
| **Gesture Completion** | High | Low | Some steps skipped/abbreviated in ADL |

**Implication for Pipeline**: Need robust activity recognition that handles intra-class variability

## 6. Recommended Data Usage Strategy

### 6.1 Sensor Selection for Initial Analysis

**Tier 1: High-Reliability Sensors (Start Here)**
- ✅ IMUs (BACK, RUA, RLA, LUA, LLA): Best quality, reproducible placement
- ✅ Ambient accelerometers: Wired, reliable furniture interaction detection
- ✅ Reed switches: Binary, reliable, clear state transitions

**Tier 2: Useful but Data Loss Expected**
- ⚠️ Body-worn 3D accelerometers: Useful for fusion despite wireless loss
- ⚠️ Object sensors: Rich information when available, handle NaN gracefully

**Tier 3: Use with Caution**
- ❌ Localization tags (TAG1-TAG4): Extremely noisy, only for coarse position if needed

### 6.2 Recommended Subset for OPPORTUNITY Challenge Baseline

The original OPPORTUNITY Challenge used a subset for benchmarking:
- **Columns**: 1-37, 38-46, 51-59, 64-72, 77-85, 90-98, 103-134, 244, 250
- **Includes**: Body-worn sensors (IMUs without quaternions, 3D acc sensors), Locomotion labels, Mid-level gestures
- **Excludes**: Object sensors, ambient sensors, localization, low-level labels
- **Purpose**: Focus on body-worn sensing for ADL recognition

### 6.3 Train/Test Split Recommendations

**Recommended Split (from Literature)**:
- **Training**: ADL1-ADL4 runs from all subjects
- **Validation**: ADL5 runs from all subjects
- **Testing**: Drill runs (for gesture recognition) or ADL5 (for naturalistic testing)

**Rationale**:
- ADL runs provide naturalistic variability
- Drill runs provide many labeled instances for gesture classes
- Cross-subject validation more challenging but more realistic

### 6.4 Data Quality Issue Injection Strategy

For pipeline testing, inject realistic data quality issues:

1. **Missing Data Injection**:
   - Random packet loss (5-20% of wireless sensor samples)
   - Occlusion-based loss (correlated with dishwasher closure)
   - Burst losses (consecutive samples, more realistic than random)

2. **Sensor Drift**:
   - Gradual offset in accelerometer readings over time
   - Temperature-dependent drift in IMU gyroscopes

3. **Synchronization Errors**:
   - Time shift between sensor modalities (±50-200ms)
   - Clock drift over long recordings

4. **Calibration Errors**:
   - Scaling errors in acceleration magnitude
   - Zero-offset errors in gyroscopes

5. **Outlier Injection**:
   - Electromagnetic interference spikes
   - Sensor saturation events

6. **Label Noise**:
   - Temporal boundary jitter (shift start/end times ±200-500ms)
   - Label confusion (swap similar gestures)

## 7. Process Mining Transformation Pipeline

### 7.1 Proposed Transformation Steps

#### Step 1: Data Loading and Preprocessing
```python
# Pseudo-code structure
def load_opportunity_data(file_path):
    """
    Load .dat file and separate sensor data from labels
    Returns: timestamps, sensor_data (243 channels), labels (7 tracks)
    """
    data = read_space_delimited(file_path)
    timestamps = data[:, 0]  # Column 1
    sensor_data = data[:, 1:243]  # Columns 2-243
    labels = data[:, 243:250]  # Columns 244-250
    return timestamps, sensor_data, labels
```

#### Step 2: Event Extraction
```python
def extract_gesture_events(timestamps, gesture_labels, subject, run):
    """
    Extract mid-level gesture events from Column 250
    Returns: List of events with start/end times and attributes
    """
    events = []
    current_gesture_id = 0
    gesture_start_time = None
    gesture_start_idx = None
    
    for i, (time, label_id) in enumerate(zip(timestamps, gesture_labels)):
        if label_id != 0 and label_id != current_gesture_id:
            # New gesture detected
            if current_gesture_id != 0:
                # Close previous gesture
                events.append({
                    'case_id': f"{subject}_{run}",
                    'activity': GESTURE_NAMES[current_gesture_id],
                    'start_time': gesture_start_time,
                    'end_time': timestamps[i-1],
                    'duration_ms': timestamps[i-1] - gesture_start_time,
                    'gesture_id': current_gesture_id
                })
            current_gesture_id = label_id
            gesture_start_time = time
            gesture_start_idx = i
            
        elif label_id == 0 and current_gesture_id != 0:
            # Gesture ends
            events.append({
                'case_id': f"{subject}_{run}",
                'activity': GESTURE_NAMES[current_gesture_id],
                'start_time': gesture_start_time,
                'end_time': time,
                'duration_ms': time - gesture_start_time,
                'gesture_id': current_gesture_id
            })
            current_gesture_id = 0
            gesture_start_time = None
            gesture_start_idx = None
    
    return events
```

#### Step 3: Attribute Enrichment
```python
def enrich_events(events, sensor_data, timestamps, locomotion_labels):
    """
    Add contextual attributes to events from sensor data
    """
    for event in events:
        # Find sensor data window for this event
        start_idx = find_timestamp_index(timestamps, event['start_time'])
        end_idx = find_timestamp_index(timestamps, event['end_time'])
        
        # Extract sensor statistics for event duration
        window_data = sensor_data[start_idx:end_idx, :]
        
        # Add locomotion context
        window_locomotion = locomotion_labels[start_idx:end_idx]
        event['posture'] = most_common(window_locomotion[window_locomotion != 0])
        
        # Add movement statistics (from IMU BACK)
        imu_back_acc = window_data[:, IMU_BACK_ACC_COLS]
        event['movement_intensity'] = np.linalg.norm(imu_back_acc, axis=1).mean()
        event['movement_variability'] = np.linalg.norm(imu_back_acc, axis=1).std()
        
        # Add hand activity (from wrist/hand accelerometers)
        right_hand_activity = np.sum(np.abs(window_data[:, RH_ACC_COLS])) > threshold
        left_hand_activity = np.sum(np.abs(window_data[:, LH_ACC_COLS])) > threshold
        if right_hand_activity and left_hand_activity:
            event['hand'] = 'Both'
        elif right_hand_activity:
            event['hand'] = 'Right'
        elif left_hand_activity:
            event['hand'] = 'Left'
        else:
            event['hand'] = 'None'
        
        # Detect data quality issues in this event window
        event['nan_percentage'] = np.isnan(window_data).sum() / window_data.size * 100
        event['has_quality_issues'] = event['nan_percentage'] > 5.0
    
    return events
```

#### Step 4: Event Log Creation
```python
def create_event_log(events):
    """
    Convert event list to PM4Py-compatible event log
    """
    # Sort events by case_id and timestamp
    events_sorted = sorted(events, key=lambda x: (x['case_id'], x['start_time']))
    
    # Convert to PM4Py EventLog format
    from pm4py.objects.log.obj import EventLog, Trace, Event
    
    log = EventLog()
    current_case_id = None
    current_trace = None
    
    for event_data in events_sorted:
        if event_data['case_id'] != current_case_id:
            if current_trace is not None:
                log.append(current_trace)
            current_trace = Trace()
            current_trace.attributes['concept:name'] = event_data['case_id']
            current_case_id = event_data['case_id']
        
        event = Event()
        event['concept:name'] = event_data['activity']
        event['time:timestamp'] = pd.Timestamp(event_data['start_time'], unit='ms')
        event['duration'] = event_data['duration_ms']
        event['posture'] = event_data.get('posture', 'Unknown')
        event['hand'] = event_data.get('hand', 'Unknown')
        event['movement_intensity'] = event_data.get('movement_intensity', 0)
        event['nan_percentage'] = event_data.get('nan_percentage', 0)
        event['has_quality_issues'] = event_data.get('has_quality_issues', False)
        
        current_trace.append(event)
    
    if current_trace is not None:
        log.append(current_trace)
    
    return log
```

### 7.2 Gesture Name Mapping

Based on label legend, create mapping dictionary:

```python
GESTURE_NAMES = {
    406516: "Open_Door_1",
    406517: "Open_Door_2",
    404516: "Close_Door_1",
    404517: "Close_Door_2",
    406520: "Open_Fridge",
    404520: "Close_Fridge",
    406505: "Open_Dishwasher",
    404505: "Close_Dishwasher",
    406519: "Open_Drawer_1",
    404519: "Close_Drawer_1",
    406511: "Open_Drawer_2",
    404511: "Close_Drawer_2",
    406508: "Open_Drawer_3",
    404508: "Close_Drawer_3",
    408512: "Clean_Table",
    407521: "Drink_From_Cup",
    405506: "Toggle_Switch"
}
```

### 7.3 Example Event Log Output

```
Case: S1_ADL1
  Event 1: Open_Drawer_1 (timestamp: 125340ms, duration: 2100ms, posture: Stand, hand: Right, nan%: 2.3)
  Event 2: Close_Drawer_1 (timestamp: 127450ms, duration: 1800ms, posture: Stand, hand: Right, nan%: 1.9)
  Event 3: Open_Fridge (timestamp: 134200ms, duration: 2500ms, posture: Walk, hand: Right, nan%: 5.7)
  Event 4: Close_Fridge (timestamp: 136710ms, duration: 2200ms, posture: Stand, hand: Right, nan%: 4.1)
  Event 5: Drink_From_Cup (timestamp: 145600ms, duration: 3200ms, posture: Stand, hand: Right, nan%: 8.2)
  ...

Case: S1_Drill
  Event 1: Open_Fridge (timestamp: 5230ms, duration: 1900ms, posture: Stand, hand: Right, nan%: 1.1)
  Event 2: Close_Fridge (timestamp: 7140ms, duration: 1700ms, posture: Stand, hand: Right, nan%: 0.9)
  Event 3: Open_Dishwasher (timestamp: 11050ms, duration: 2300ms, posture: Stand, hand: Right, nan%: 1.5)
  Event 4: Close_Dishwasher (timestamp: 13360ms, duration: 2100ms, posture: Stand, hand: Right, nan%: 12.3)
  ...
```

## 8. Expected Data Quality Issues for Pipeline Testing

### 8.1 Naturally Occurring Issues (Ground Truth)

| Issue Type | Detection Method | Expected Cases | Pipeline Test Value |
|------------|------------------|----------------|---------------------|
| **Wireless NaN Bursts** | Count consecutive NaN samples | ~15-20% of object sensor windows | ✅ Real missing data patterns |
| **Dishwasher Occlusion** | Correlate NaN with reed switch | Every cleanup phase | ✅ Contextual data loss |
| **Annotation Jitter** | Compare gesture duration variance | All low-level actions | ✅ Label quality assessment |
| **Short Gestures** | Duration < 500ms | Low-level actions | ✅ Boundary detection challenge |
| **Activity Interleaving** | Overlapping high-level labels | High-level activities | ✅ Fuzzy process boundaries |

### 8.2 Synthetic Issues for Enhanced Testing

| Issue Type | Injection Method | Purpose | Detection Challenge |
|------------|------------------|---------|---------------------|
| **Drift** | Add linearly increasing offset to sensor | Test adaptation methods | Medium |
| **Calibration Error** | Multiply sensor values by factor | Test invariance | Easy |
| **Synchronization Lag** | Shift timestamps between modalities | Test temporal alignment | Hard |
| **Sensor Dropout** | Set sensor channel to NaN for period | Test redundancy handling | Easy |
| **Electromagnetic Noise** | Add high-frequency random spikes | Test filtering | Medium |
| **Label Noise** | Randomly shift boundaries ±500ms | Test robustness | Hard |

## 9. Literature Baseline Results

From published papers using OPPORTUNITY dataset:

### 9.1 Mid-Level Gesture Recognition
- **Baseline methods**: k-NN, QDA, naive Bayes
- **Features**: Time-domain statistics (mean, variance, energy)
- **Best F1-score**: ~0.7-0.8 for common gestures
- **Challenging classes**: Gestures with similar motion patterns

### 9.2 Locomotion Recognition
- **Classes**: Stand, Walk, Sit, Lie
- **Performance**: >95% accuracy achievable
- **Challenge**: Short walk sequences, single steps

### 9.3 Known Challenges
- **Null class** (no activity): Dominates timeline, class imbalance
- **Gesture boundary detection**: Hard to pinpoint exact start/end
- **Subject generalization**: Leave-one-subject-out significantly harder than cross-validation

## 10. Summary and Recommendations

### 10.1 Key Strengths for Data Quality Pipeline
✅ **Multi-modal sensor richness**: 243 sensor channels enable comprehensive data quality analysis  
✅ **Multi-level annotations**: Hierarchical labels support abstraction level analysis  
✅ **Realistic missing data**: NaN patterns reflect real-world wireless sensor challenges  
✅ **Known quality issues**: Documented jitter, synchronization issues for validation  
✅ **Multiple subjects**: Inter-subject variability tests generalization  
✅ **Drill vs. ADL runs**: Controlled vs. naturalistic data for comparative analysis  

### 10.2 Recommended Pipeline Configuration

**Primary Focus**: Mid-level gestures (Column 250) with IMU sensors (high quality)  
**Event Extraction**: Label-based gesture segmentation with temporal attributes  
**Data Quality Indicators**: NaN percentage, synchronization lag, annotation jitter  
**Case Definition**: Per-run traces (`S<N>_ADL<M>` or `S<N>_Drill`)  
**Attribute Enrichment**: Posture, hand usage, movement intensity, data completeness  

### 10.3 Expected Pipeline Outcomes

1. **Process Discovery**: Should reveal drill run structure (20 repetitions of fixed sequence) vs. free-form ADL patterns
2. **Data Quality Impact**: Correlation between NaN percentage and gesture recognition confidence
3. **Explainability**: IMU sensors (especially RUA, RH) should be most important for hand gesture recognition
4. **Anomaly Detection**: Unusual gesture durations, missing expected gestures, high NaN periods
5. **Conformance**: Drill runs should show high conformance, ADL runs lower conformance with higher variability

### 10.4 Implementation Priority

1. **Phase 1**: Load S1-ADL1 and S1-Drill, extract mid-level gestures, create basic event log
2. **Phase 2**: Add IMU-based attribute enrichment (movement, posture)
3. **Phase 3**: Implement NaN detection and data quality metrics
4. **Phase 4**: Expand to all subjects, compare ADL vs. Drill patterns
5. **Phase 5**: Inject synthetic quality issues, validate detection pipeline
