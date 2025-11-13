## Dataset Description

This dataset captures a 30-hour continuous manufacturing process executed in the Future Factories Testbed at the University of South Carolina. The system assembles and disassembles a four-piece rocket prototype using four industrial robots, a conveyor loop, PLC-controlled I/O, and multiple sensors. Both normal and defective assemblies were produced, with anomalies introduced manually (e.g., missing rocket components).

Two synchronized datasets were collected:

- **Analog Dataset**: 10 Hz tabular sensor data over 325 filtered production cycles, stored in CSV files.
- **Multi-Modal Dataset**: 2–3 Hz sensor data synchronized with dual-camera image streams, stored as batches of image folders with corresponding JSON metadata (166k records total).

Use cases include industrial machine learning research, anomaly detection, process monitoring, and cyber-physical manufacturing applications.

---

## Appendix — Sensor Table

| Asset | Sensor | Data Type | Multi-Modal Dataset | Analog Dataset | Description |
|---|---|---|:---:|:---:|---|
| Conveyors | Q_VFD1_Temperature | Float | ✓ |  | Temperature of conveyor 1 (°F) |
|  | Q_VFD2_Temperature | Float | ✓ |  | Temperature of conveyor 2 (°F) |
|  | Q_VFD3_Temperature | Float | ✓ |  | Temperature of conveyor 3 (°F) |
|  | Q_VFD4_Temperature | Float | ✓ |  | Temperature of conveyor 4 (°F) |
|  | M_Conv1_Speed_mmps | Integer |  | ✓ | Conveyor 1 speed (mm/s) |
|  | M_Conv2_Speed_mmps | Integer |  | ✓ | Conveyor 2 speed (mm/s) |
|  | M_Conv3_Speed_mmps | Integer |  | ✓ | Conveyor 3 speed (mm/s) |
|  | M_Conv4_Speed_mmps | Integer |  | ✓ | Conveyor 4 speed (mm/s) |
| Grippers | I_R01_Gripper_Pot | Integer | ✓ |  | Potentiometer value, Robot 1 gripper |
|  | I_R02_Gripper_Pot | Integer | ✓ |  | Potentiometer value, Robot 2 gripper |
|  | I_R03_Gripper_Pot | Integer | ✓ |  | Potentiometer value, Robot 3 gripper |
|  | I_R04_Gripper_Pot | Integer | ✓ |  | Potentiometer value, Robot 4 gripper |
|  | I_R01_Gripper_Load | Integer | ✓ |  | Load cell value, Robot 1 gripper |
|  | I_R02_Gripper_Load | Integer | ✓ |  | Load cell value, Robot 2 gripper |
|  | I_R03_Gripper_Load | Integer | ✓ |  | Load cell value, Robot 3 gripper |
|  | I_R04_Gripper_Load | Integer | ✓ |  | Load cell value, Robot 4 gripper |
| Robot 1 | M_R01_SJointAngle_Degree | Float | ✓ |  | Joint S angle (°) |
|  | M_R01_LJointAngle_Degree | Float | ✓ |  | Joint L angle (°) |
|  | M_R01_UJointAngle_Degree | Float | ✓ |  | Joint U angle (°) |
|  | M_R01_RJointAngle_Degree | Float | ✓ |  | Joint R angle (°) |
|  | M_R01_BJointAngle_Degree | Float | ✓ |  | Joint B angle (°) |
|  | M_R01_TJointAngle_Degree | Float | ✓ |  | Joint T angle (°) |
| Robot 2 | M_R02_SJointAngle_Degree | Float | ✓ |  | Joint S angle (°) |
|  | M_R02_LJointAngle_Degree | Float | ✓ |  | Joint L angle (°) |
|  | M_R02_UJointAngle_Degree | Float | ✓ |  | Joint U angle (°) |
|  | M_R02_RJointAngle_Degree | Float | ✓ |  | Joint R angle (°) |
|  | M_R02_BJointAngle_Degree | Float | ✓ |  | Joint B angle (°) |
|  | M_R02_TJointAngle_Degree | Float | ✓ |  | Joint T angle (°) |
| Robot 3 | M_R03_SJointAngle_Degree | Float | ✓ |  | Joint S angle (°) |
|  | M_R03_LJointAngle_Degree | Float | ✓ |  | Joint L angle (°) |
|  | M_R03_UJointAngle_Degree | Float | ✓ |  | Joint U angle (°) |
|  | M_R03_RJointAngle_Degree | Float | ✓ |  | Joint R angle (°) |
|  | M_R03_BJointAngle_Degree | Float | ✓ |  | Joint B angle (°) |
|  | M_R03_TJointAngle_Degree | Float | ✓ |  | Joint T angle (°) |
| Robot 4 | M_R04_SJointAngle_Degree | Float | ✓ |  | Joint S angle (°) |
|  | M_R04_LJointAngle_Degree | Float | ✓ |  | Joint L angle (°) |
|  | M_R04_UJointAngle_Degree | Float | ✓ |  | Joint U angle (°) |
|  | M_R04_RJointAngle_Degree | Float | ✓ |  | Joint R angle (°) |
|  | M_R04_BJointAngle_Degree | Float | ✓ |  | Joint B angle (°) |
|  | M_R04_TJointAngle_Degree | Float | ✓ |  | Joint T angle (°) |
| Safety | I_SafetyDoor1_Status | Bool | ✓ |  | True if Safety Door 1 is open |
|  | I_SafetyDoor2_Status | Bool | ✓ |  | True if Safety Door 2 is open |
|  | I_HMI_EStop_Status | Bool | ✓ |  | True if emergency stop pressed |
| Cycle Management | Q_Cell_CycleCount | Integer | ✓ |  | Cycle counter (resets on interruptions) |
| Material Handling | I_MHS_GreenRocketTray | Bool | ✓ |  | Tray detected at handling station |
| Stopper | I_Stopper1_Status | Bool | ✓ |  | Stopper 1 status |
|  | I_Stopper2_Status | Bool | ✓ |  | Stopper 2 status |
|  | I_Stopper3_Status | Bool | ✓ |  | Stopper 3 status |
|  | I_Stopper4_Status | Bool | ✓ |  | Stopper 4 status |
|  | I_Stopper5_Status | Bool | ✓ |  | Stopper 5 status |
| Cameras | Path1 | String | ✓ |  | Camera 1 image path |
|  | Path2 | String | ✓ |  | Camera 2 image path |


## Data Quality Considerations

The dataset is released without preprocessing to preserve raw industrial characteristics. While this increases realism, it also introduces potential data quality issues that should be considered when applying downstream analytics or machine learning workflows:

### Possible Issues

| Category | Description | Potential Sources |
|---|---|---|
| Sensor Noise | Fluctuations and non-smooth readings | Analog sensors (potentiometers, load cells), electrical noise |
| Missing Data | Gaps in time series records | Communication delays, sensor unavailability, PLC buffering |
| Desynchronization | Misalignment between sensor streams and images | Variable acquisition rates (10 Hz vs. 2–3 Hz), delayed frames |
| Downtime Records | Idle periods with non-informative values | Conveyor stops, robot idle periods, manual anomaly insertions |
| Drift and Calibration Effects | Gradual change in sensor baselines over long run | Load cell tension changes, temperature effects on robot joints |
| Boolean State Flicker | Short, inconsistent state oscillations | Mechanical jitter, contact bounce, PLC sampling artifacts |
| Cycle Counter Reset | Counter resets on interruptions | Mid-cycle stops, operational restarts |
| Human-Induced Variability | Manual removal of rocket parts introduces non-uniform anomaly patterns | Human intervention, timing variation |

### Implications for Use

- Preprocessing may be required for alignment, filtering, and segmentation.
- Anomaly labels correspond to cycles only, not individual timestamps.
- Image-sensor synchronization requires JSON metadata resolution.
- Idle system states should be excluded for cycle-level learning tasks.
- Raw values may require normalization per robot/joint to ensure model stability.

This dataset is well-suited for work in anomaly detection, industrial time-series modeling, multi-modal fusion, and robustness research where realistic noise and imperfect industrial conditions are desirable.
