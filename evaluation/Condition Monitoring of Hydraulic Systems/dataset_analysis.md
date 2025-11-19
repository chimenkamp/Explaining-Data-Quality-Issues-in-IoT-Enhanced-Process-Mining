# Condition Monitoring of Hydraulic Systems - Dataset Analysis

## Dataset Overview

**Source**: ZeMA gGmbH (Zentrum für Mechatronik und Automatisierungstechnik)  
**Domain**: Industrial Manufacturing / Predictive Maintenance  
**Dataset Type**: Multivariate Time-Series for Condition Monitoring  
**Task**: Classification/Regression of Component Degradation States  
**Number of Cycles**: 2,205 measurement cycles  
**Cycle Duration**: 60 seconds per cycle  
**Total Attributes**: 43,680 time-series measurements per cycle

## 1. Physical System Description

### 1.1 Hydraulic Test Rig Architecture

The dataset originates from a controlled hydraulic test rig with two main circuits:

1. **Primary Working Circuit**: Main hydraulic power system that performs work cycles
2. **Secondary Cooling-Filtration Circuit**: Supports primary circuit via oil tank connection

### 1.2 System Components Under Monitoring

Four critical hydraulic components are monitored with quantitatively varied degradation states:

| Component | Function | Degradation Parameter | States |
|-----------|----------|----------------------|--------|
| **Cooler** | Heat dissipation | Cooling efficiency (%) | 3 states: 3%, 20%, 100% |
| **Valve** | Flow control | Switching behavior (%) | 4 states: 73%, 80%, 90%, 100% |
| **Pump** | Pressure generation | Internal leakage | 3 states: 0 (none), 1 (weak), 2 (severe) |
| **Accumulator** | Pressure storage | Pre-charge pressure (bar) | 4 states: 90, 100, 115, 130 bar |

### 1.3 Operating Procedure

- System executes **constant load cycles** (60 seconds each)
- Component conditions are **intentionally varied** to generate labeled training data
- Each cycle represents one complete working sequence with fixed load pattern
- Sensor data captured continuously throughout each cycle

## 2. Sensor Configuration and Data Structure

### 2.1 Sensor Inventory (17 Total Sensors)

| Sensor ID | Physical Quantity | Unit | Sampling Rate | Samples/Cycle | Sensor Count |
|-----------|------------------|------|---------------|---------------|--------------|
| **PS1-PS6** | Pressure | bar | 100 Hz | 6,000 | 6 sensors |
| **EPS1** | Motor Power | W | 100 Hz | 6,000 | 1 sensor |
| **FS1, FS2** | Volume Flow | l/min | 10 Hz | 600 | 2 sensors |
| **TS1-TS4** | Temperature | °C | 1 Hz | 60 | 4 sensors |
| **VS1** | Vibration | mm/s | 1 Hz | 60 | 1 sensor |
| **SE** | Efficiency Factor | % | 1 Hz | 60 | 1 sensor (virtual) |
| **CE** | Cooling Efficiency | % | 1 Hz | 60 | 1 sensor (virtual) |
| **CP** | Cooling Power | kW | 1 Hz | 60 | 1 sensor (virtual) |

**Total measurements per cycle**: 43,680 data points (6×6,000 + 1×6,000 + 2×600 + 8×60)

### 2.2 Data File Structure

- **Format**: Tab-delimited text matrices
- **Rows**: Each row = one complete 60-second cycle (2,205 rows total)
- **Columns**: Time-series measurements within cycle (column count varies by sensor)
- **Missing Values**: None reported
- **File Organization**: One file per sensor (17 files total)

### 2.3 Virtual Sensors

Three "virtual" sensors represent calculated/derived quantities:
- **CE (Cooling Efficiency)**: Computed from temperature and flow measurements
- **CP (Cooling Power)**: Derived from cooling circuit performance
- **SE (Efficiency Factor)**: Overall system efficiency metric

Note: Virtual sensors show distinctive patterns (many zero values at cycle start, then stable values), indicating they may require warm-up time or represent periodic calculations.

## 3. Target Labels and Condition Distribution

### 3.1 Profile Data Structure

**File**: `profile.txt` (2,205 rows × 5 columns, tab-delimited)

Each row contains condition labels for the corresponding cycle:

| Column | Component | Type | Description |
|--------|-----------|------|-------------|
| **1** | Cooler | Categorical | Efficiency level (%, degradation measure) |
| **2** | Valve | Categorical | Switching behavior (%, degradation measure) |
| **3** | Pump | Categorical | Leakage severity (ordinal scale) |
| **4** | Accumulator | Categorical | Pressure level (bar, degradation measure) |
| **5** | System | Binary Flag | Stability indicator |

### 3.2 Condition State Distributions

#### Cooler Condition (Efficiency %)
- **100% (Full efficiency)**: 741 cycles (33.6%)
- **20% (Reduced efficiency)**: 732 cycles (33.2%)
- **3% (Near total failure)**: 732 cycles (33.2%)
- **Classification Difficulty**: Easy (balanced distribution, well-separated states)

#### Valve Condition (Switching Behavior %)
- **100% (Optimal)**: 1,125 cycles (51.0%)
- **90% (Small lag)**: 360 cycles (16.3%)
- **80% (Severe lag)**: 360 cycles (16.3%)
- **73% (Near total failure)**: 360 cycles (16.3%)
- **Classification Difficulty**: Easy (imbalanced but previously reported as perfectly classifiable)

#### Pump Leakage (Internal Leakage Level)
- **0 (No leakage)**: 1,221 cycles (55.4%)
- **1 (Weak leakage)**: 492 cycles (22.3%)
- **2 (Severe leakage)**: 492 cycles (22.3%)
- **Classification Difficulty**: Moderate (ordinal degradation, imbalanced)

#### Hydraulic Accumulator (Pre-charge Pressure bar)
- **130 bar (Optimal)**: 599 cycles (27.2%)
- **115 bar (Slightly reduced)**: 399 cycles (18.1%)
- **100 bar (Severely reduced)**: 399 cycles (18.1%)
- **90 bar (Near total failure)**: 808 cycles (36.6%)
- **Classification Difficulty**: Difficult (highest error rates reported in literature - subtle pressure variations)

#### Stable Flag (System Stability)
- **0 (Stable conditions reached)**: 1,449 cycles (65.7%)
- **1 (Static conditions not yet reached)**: 756 cycles (34.3%)
- **Purpose**: Indicates whether system reached steady-state during cycle

### 3.3 Multi-Fault Scenarios

The dataset contains **all combinations** of the four component states, creating a complex multi-fault condition monitoring scenario. This reflects realistic industrial settings where multiple components may degrade simultaneously at different rates.

## 4. Underlying Process Understanding

### 4.1 Process Flow

```
CYCLE START (t=0)
    ↓
[System Initialization]
    ↓
[Ramp-up Phase] (High motor power, increasing pressures/flows)
    ↓
[Steady-State Operation] (Constant load, stable sensor readings)
    ↓
[Cooling/Temperature Management] (Active throughout cycle)
    ↓
[Pressure Regulation] (Accumulator compensates variations)
    ↓
CYCLE END (t=60s)
```

### 4.2 Physical Process Characteristics

**Observable Patterns in Sensor Data:**

1. **Temperature Sensors (TS1-TS4)**:
   - Gradual increase throughout 60-second cycle
   - Reflects heat generation from hydraulic work
   - Cooling efficiency impacts temperature rise rate

2. **Flow Sensors (FS1, FS2)**:
   - High initial spike (system startup/fill)
   - Drops to near-zero during most of cycle
   - Indicates discrete pumping/valve actuation events
   - May show small fluctuations during steady-state

3. **Pressure Sensors (PS1-PS6)**:
   - Expected to show work cycle pressure patterns
   - Pump leakage affects pressure maintenance
   - Accumulator condition impacts pressure stability
   - Files are large (>50MB), indicating high-frequency rich data

4. **Motor Power (EPS1)**:
   - High during startup and load application
   - Correlates with pump efficiency and system resistance
   - Large file size indicates complex power signature

5. **Virtual Sensors (CE, CP, SE)**:
   - CE/SE show zero values in early samples, then stabilize
   - Suggests calculation-based sensors that require system warm-up
   - Provide derived health indicators

### 4.3 Process-Level Activities

Based on the hydraulic system operation, we can infer the following **event sequence** per cycle:

1. **Initialization** (≈0-5s): System startup, pressure building
2. **Load Application** (≈5-15s): Valve actuation, primary work phase
3. **Steady-State Operation** (≈15-50s): Constant load, monitoring phase
4. **Cycle Completion** (≈50-60s): Load release, preparation for next cycle

**Note**: Exact timing would require detailed examination of sensor traces, but flow sensor data suggests most activity occurs at cycle start/end.

## 5. Data Quality Characteristics

### 5.1 Inherent Data Quality Issues

| Issue Type | Manifestation | Source | Impact |
|------------|---------------|--------|--------|
| **Sensor Drift** | Gradual offset over time | Temperature sensors, analog components | Addressed in literature: up to 5 sensors can be compensated |
| **Calibration Offset** | Systematic bias | All analog sensors | Feature extraction can compensate |
| **Sampling Rate Variation** | Different temporal resolutions | Design choice (cost/bandwidth) | Requires synchronization for fusion |
| **Virtual Sensor Delay** | Zero values at cycle start | Calculation requirements | Excludes early-cycle features |
| **Transient Noise** | High-frequency variations | Switching events, pump pulses | Common in hydraulic systems |

### 5.2 Confirmed Data Characteristics

- **No missing values**: All 2,205 cycles complete
- **Fixed cycle duration**: All cycles exactly 60 seconds
- **Controlled conditions**: Experimental setup with known degradation states
- **Balanced/semi-balanced classes**: Good distribution for ML training
- **High-quality labels**: Manually controlled component conditions

### 5.3 Stable Flag Interpretation

The **stable flag** (34.3% unstable cycles) indicates:
- Some cycles did not reach thermal/hydraulic equilibrium
- May affect temperature-based features
- Could represent transient startup effects
- Important for filtering training data in regression tasks

## 6. Use Cases for Data Quality Framework

### 6.1 Potential Data Quality Issues to Inject/Detect

For explainability pipeline development, this dataset supports:

1. **Sensor Drift Simulation**:
   - Gradual offset in temperature/pressure sensors
   - Realistic degradation pattern over cycle sequences

2. **Sampling Irregularities**:
   - Simulated packet loss in high-frequency sensors (PS, EPS)
   - Missing samples in time-series

3. **Calibration Errors**:
   - Multiplicative bias in flow/pressure sensors
   - Zero-point drift in analog measurements

4. **Outlier Injection**:
   - Spurious spikes in sensor readings
   - Electromagnetic interference simulation

5. **Synchronization Issues**:
   - Time-shift between sensor streams
   - Clock drift effects

6. **Truncation/Incompleteness**:
   - Partial cycle recordings
   - Sensor dropout during cycle

### 6.2 Process Mining Transformation Strategy

**Challenge**: This is **pure time-series data**, not event log format.

**Transformation Approach**:

1. **Event Abstraction Layer**:
   - Define events based on sensor patterns:
     - `Cycle_Start`: Flow spike detected
     - `Load_Applied`: Pressure threshold exceeded
     - `Steady_State_Reached`: Temperature/pressure stabilization
     - `Cooling_Active`: CE/CP values non-zero
     - `Cycle_Complete`: End of 60s period

2. **Case Identification**:
   - Each cycle (row) = one case/trace
   - Case ID: `Cycle_XXXX` (1-2205)

3. **Temporal Ordering**:
   - Extract events from time-series patterns
   - Timestamp events within 60-second window
   - Maintain intra-cycle event sequence

4. **Activity Labels**:
   - Derive from multi-sensor feature extraction
   - Example: "High_Pressure_Operation", "Flow_Regulation", "Temperature_Management"

5. **Attributes**:
   - Component conditions (cooler, valve, pump, accumulator)
   - Statistical features from sensor windows (mean, std, max, min)
   - Stable flag

### 6.3 Ground Truth for Evaluation

**Strong advantage**: Known component conditions provide ground truth for:
- **Impact analysis**: How does cooler degradation affect temperature patterns?
- **Propagation**: Does pump leakage cause pressure instability?
- **Classification**: Which sensors are most informative for each component?
- **Explainability validation**: Do explanations align with physical causality?

## 7. Literature-Based Insights

### 7.1 Previous Performance Results

From cited papers:
- **Cooler & Valve**: Perfectly classifiable (100% accuracy achieved)
- **Pump**: Moderate difficulty (multiple state distinctions needed)
- **Accumulator**: Most challenging (error rate reduced from 9.6% to 0.35% with advanced features)

### 7.2 Feature Engineering Approaches

- **Pearson correlation** for automated feature extraction
- **ALA (Automatic Learning Algorithm)** and **RFESVM** for feature selection
- Time-domain statistics (mean, variance, peaks)
- Frequency-domain features (FFT, power spectral density)

### 7.3 Sensor Redundancy

- Up to **5 sensors can fail** and system remains classifiable
- Indicates high redundancy and correlation between sensors
- Supports robust operation in degraded conditions

## 8. Recommended Pipeline Approach

### 8.1 Phase 1: Data Understanding & Preprocessing

1. **Load sensor data** (17 files)
2. **Synchronize temporal resolution** (align 1Hz, 10Hz, 100Hz streams)
3. **Load profile labels** (component conditions)
4. **Exploratory analysis**:
   - Visualize typical cycle patterns per sensor
   - Correlation analysis between sensors
   - Feature importance for each target variable

### 8.2 Phase 2: Event Abstraction

1. **Define event detection rules**:
   - Threshold-based (e.g., flow > 1.0 l/min = "Flow_Active")
   - Gradient-based (e.g., temperature increase rate)
   - Pattern-based (e.g., pressure oscillation frequency)

2. **Extract events per cycle**:
   - Map continuous signals to discrete events
   - Timestamp events within 60s window
   - Create event attributes from sensor windows

3. **Build event log**:
   - Format: XES or CSV with case_id, activity, timestamp, attributes
   - Include component condition labels
   - Tag stable/unstable cycles

### 8.3 Phase 3: Quality Issue Injection

1. **Select cycles for injection** (stratified by component conditions)
2. **Inject realistic issues**:
   - Sensor drift: Linear/nonlinear offset
   - Missing data: Random sample dropout
   - Outliers: Gaussian noise bursts
   - Synchronization: Time-shift between sensors

3. **Propagate effects**:
   - Track how quality issues affect event detection
   - Document ground truth of injected issues

### 8.4 Phase 4: Quality Detection & Propagation

1. **Apply quality detectors** from framework
2. **Measure propagation**:
   - Does drift in TS1 affect "Steady_State" event detection?
   - Does pressure outlier change case classification?

3. **Compare with ground truth**:
   - Detection rate vs. actual injected issues
   - False positive analysis

### 8.5 Phase 5: Explainability & Insights

1. **Generate explanations**:
   - Why was component X classified as degraded?
   - Which sensors contributed most?
   - How did quality issue Y affect outcome?

2. **Validate physical plausibility**:
   - Do explanations align with hydraulic physics?
   - Are sensor-component relationships correct?

3. **Produce insights**:
   - Recommendations for sensor maintenance
   - Quality tolerance thresholds
   - Sensor redundancy strategies

## 9. Expected Challenges

### 9.1 Technical Challenges

1. **High dimensionality**: 43,680 raw features per cycle requires dimensionality reduction
2. **Multi-resolution data**: Synchronizing 1Hz, 10Hz, 100Hz streams
3. **Large file sizes**: PS and EPS files >50MB (memory management needed)
4. **Event abstraction ambiguity**: No clear discrete events in continuous hydraulics

### 9.2 Domain-Specific Challenges

1. **Temporal dependencies**: Sensor values are not i.i.d., require sequence modeling
2. **Physical constraints**: Not all sensor combinations are physically plausible
3. **Component interactions**: Multiple components affect same sensors (confounding)
4. **Cyclic stationarity**: Patterns repeat but with variations

### 9.3 Explainability Challenges

1. **Attribution complexity**: Which sensor and which time window matters?
2. **Multi-output problem**: Four simultaneous classification targets
3. **Feature interpretability**: Statistical features less interpretable than raw signals
4. **Causal reasoning**: Correlation vs. causation in sensor relationships

## 10. Key Differences from Event Log Data

| Aspect | Traditional Event Logs | This Hydraulic Dataset |
|--------|----------------------|------------------------|
| **Data structure** | Discrete events with timestamps | Continuous time-series signals |
| **Activities** | Explicit activity labels | Must be inferred from patterns |
| **Cases** | Natural trace boundaries | Cycles are artificial groupings |
| **Temporal resolution** | Event-driven (irregular) | Fixed sampling rates |
| **Attributes** | Categorical/discrete | Continuous sensor readings |
| **Process complexity** | Multiple process variants | Fixed process, varying conditions |
| **Quality issues** | Log incompleteness, noise | Sensor faults, drift, noise |

## 11. Summary and Recommendations

### 11.1 Dataset Strengths

✅ **High-quality ground truth**: Known component conditions  
✅ **Rich sensor coverage**: 17 sensors, multiple physical domains  
✅ **Realistic degradation**: Experimentally controlled failure modes  
✅ **Sufficient scale**: 2,205 cycles for ML training  
✅ **No missing values**: Complete dataset  
✅ **Well-documented**: Published research, clear metadata  
✅ **Multi-fault scenarios**: Complex condition combinations  

### 11.2 Dataset Limitations

⚠️ **Not event log format**: Requires significant transformation  
⚠️ **Limited process variety**: Single fixed load cycle  
⚠️ **Large files**: Computational resources needed for 100Hz sensors  
⚠️ **Controlled environment**: May not reflect real-world variability  
⚠️ **Domain complexity**: Requires hydraulic system knowledge  

### 11.3 Suitability for Explainability Pipeline

**Overall Assessment**: ⭐⭐⭐⭐☆ (Highly Suitable with Modifications)

**Best suited for**:
- Time-series to process mining transformation methodology development
- Sensor-level quality issue detection and propagation
- Physics-informed explainability validation
- Multi-sensor fusion under degraded conditions

**Requires**:
- Custom event abstraction layer
- Domain expertise for validation
- Computational resources for high-frequency data

### 11.4 Recommended Next Steps

1. **Prototype event abstraction** on small subset (100 cycles)
2. **Validate event extraction** against manual inspection
3. **Define quality issue injection scenarios** (priority: drift, outliers, missing data)
4. **Implement cycle-to-trace conversion** module
5. **Test framework components** on transformed data
6. **Document transformation methodology** for reproducibility

## 12. Conclusion

The **Condition Monitoring of Hydraulic Systems** dataset provides an excellent testbed for developing and validating the explainability pipeline, particularly for:
- **Signal-to-event abstraction** in IoT-enhanced process mining
- **Multi-sensor data quality** detection and propagation
- **Physical grounding** of explainability methods

While not in event log format, the dataset's rich sensor coverage, clear degradation states, and published benchmark results make it ideal for demonstrating how data quality issues propagate through transformation pipelines and affect final process mining outcomes. The challenge of abstracting events from continuous signals directly addresses the core problem of IoT-enhanced process mining.

---

**Dataset Citation**:  
Helwig, N., Pignanelli, E., & Schütze, A. (2015). Condition Monitoring of a Complex Hydraulic System Using Multivariate Statistics. *IEEE International Instrumentation and Measurement Technology Conference (I2MTC-2015)*, Pisa, Italy. DOI: 10.1109/I2MTC.2015.7151267

**Analysis Date**: November 19, 2025  
**Analysis Version**: 1.0
