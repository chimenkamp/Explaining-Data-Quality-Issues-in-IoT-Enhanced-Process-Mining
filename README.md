# IoT Data Quality Pipeline

A comprehensive system for detecting, classifying, propagating, and interpreting data quality issues in IoT environments, based on the research framework for explainable process analytics.

## Overview

Instead of treating data quality issues as problems to be filtered out, this system **propagates them through the entire data pipeline** and transforms them into **valuable insights** about the IoT environment. The core philosophy is that data quality issues carry meaningful information about system behavior, sensor infrastructure, and organizational processes.

## Key Features

### 🔍 **Quality Issue Detection & Classification**
- **C1: Inadequate Sampling Rate** - Detects when sensors sample too slowly for process dynamics
- **C2: Poor Sensor Placement** - Identifies overlapping or inconsistent sensor readings
- **C3: Sensor Noise & Outliers** - Finds high variance and erroneous readings
- **C4: Sensor Range Too Small** - Detects blind spots and value clipping
- **C5: High Data Volume** - Identifies processing bottlenecks and data loss

### 📊 **Quality-Aware Pipeline**
- **Preprocessing** - Gentle data cleaning while preserving quality signatures
- **Event Abstraction** - Converts sensor data to structured events with quality annotations
- **Case Correlation** - Groups events into process instances with quality assessment
- **Process Mining** - Discovers process models using quality-weighted algorithms
- **Enhanced Visualization** - Creates quality-annotated process visualizations

### 💡 **Explainable Insights**
- **Root Cause Analysis** - Links observed effects to underlying causes
- **Information Gain Metrics** - Quantifies interpretability and actionability
- **Actionable Recommendations** - Provides specific remediation strategies
- **Causal Chain Detection** - Identifies how issues propagate through stages

### 🏭 **Synthetic IoT Environment**
- **Realistic Sensor Models** - Power, temperature, vibration, position sensors
- **Manufacturing Processes** - Welding, inspection, packaging stations
- **Configurable Quality Issues** - Controllable issue injection for testing
- **Multi-Station Workflows** - Complex process instances across machines

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd iot-data-quality-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py
```

## Quick Start

```python
from src.synthetic_environment.iot_environment import IoTEnvironment
from src.pipeline.pipeline_manager import PipelineManager
from src.explainability.insights import InsightGenerator

# Create synthetic IoT environment
env = IoTEnvironment(name="Manufacturing Line", duration_hours=8)
env.add_welding_station()
env.add_inspection_station()
env.add_packaging_station()

# Generate data with quality issues
data = env.generate_data()

# Run quality-aware pipeline
pipeline = PipelineManager()
results = pipeline.run(data, env)

# Generate explainable insights
insight_generator = InsightGenerator()
insights = insight_generator.generate_insights(results)

# Display results
for insight in insights[:5]:
    print(f"• {insight['message']}")
    print(f"  Confidence: {insight['confidence']:.2f}")
    print(f"  Actionable: {insight['actionable']}")
```

## Architecture

```
📁 src/
├── 📁 synthetic_environment/     # IoT environment simulation
│   ├── iot_environment.py       # Main environment class
│   ├── sensor_models.py         # Sensor implementations
│   └── data_generator.py        # Synthetic data generation
├── 📁 data_quality/             # Quality issue detection
│   ├── detectors.py             # Issue detection algorithms
│   ├── classifiers.py           # Probabilistic classification
│   └── propagation.py           # Cross-stage propagation
├── 📁 pipeline/                 # Data processing pipeline
│   ├── pipeline_manager.py      # Main pipeline orchestrator
│   ├── preprocessing.py         # Quality-aware preprocessing
│   ├── event_abstraction.py     # Sensor data → events
│   ├── case_correlation.py      # Events → process instances
│   ├── process_mining.py        # Quality-weighted process discovery
│   └── visualization.py         # Enhanced visualizations
├── 📁 explainability/           # Insights and explanations
│   ├── insights.py              # Insight generation
│   └── explanations.py          # Detailed explanations
└── 📁 config/                   # Configuration
    └── settings.py              # System parameters
```

## Example Results

### Quality Issues Detected
```
📊 QUALITY ISSUES DETECTED: 12
  C1_inadequate_sampling: 3 instances
    • Low sampling rate: 0.45 Hz (Confidence: 0.85)
    • Detected 15 large sampling gaps (Confidence: 0.78)
  C3_sensor_noise: 4 instances  
    • 156 outliers (18.2% of readings) (Confidence: 0.92)
    • High noise level: 0.134 (Confidence: 0.71)
```

### Process Model Metrics
```
🔄 PROCESS MODEL METRICS:
  Fitness Score: 0.734
  Precision Score: 0.892
  Complexity Score: 0.456
  Quality-Weighted Fitness: 0.681
  Activities Discovered: 11
  Causality Relations: 18
```

### Generated Insights
```
💡 GENERATED INSIGHTS: 8
  1. Most prevalent quality issue: C3_sensor_noise (4 occurrences)
     Confidence: 0.90 | Actionable: True
     Recommendations: Check sensor calibration, Review electrical interference
  
  2. Sensor WS_01_PWR has multiple quality issues: C1_inadequate_sampling, C3_sensor_noise
     Confidence: 0.85 | Actionable: True
     Recommendations: Prioritize maintenance, Update sensor sampling configuration
```

## Key Concepts

### Information Gain Framework
Quality issues are reframed as sources of information:
- **Interpretability Gain** - How much insight the issue provides
- **Actionability Gain** - How clearly it points to specific actions
- **Explainability Gain** - How well it explains observed phenomena

### Probabilistic Reasoning
Uses Bayesian inference to handle uncertainty:
```python
P(C1_inadequate_sampling) = 0.85  # High likelihood
P(C2_poor_placement) = 0.23       # Lower likelihood  
P(C3_sensor_noise) = 0.67         # Moderate likelihood
```

### Quality Propagation
Issues propagate through pipeline stages with evolving signatures:
- **Raw Data** → irregular_sampling, high_variance
- **Events** → missing_events, false_events  
- **Process Model** → incomplete_paths, spaghetti_model

## Research Background

This implementation is based on the research concept of **propagating data quality issues through IoT process analytics pipelines** rather than filtering them out. The key insight is that quality issues contain valuable information about:

- **IoT Infrastructure** - Sensor placement, configuration, maintenance needs
- **Process Understanding** - Hidden process variants, timing dependencies
- **System Health** - Degradation patterns, failure prediction
- **Organizational Factors** - Process changes, equipment updates

## Use Cases

### Manufacturing
- **Predictive Maintenance** - Quality issues indicate sensor degradation
- **Process Optimization** - Missing events reveal hidden process steps
- **Equipment Monitoring** - Noise patterns suggest mechanical issues

### Smart Buildings
- **HVAC Optimization** - Sensor placement issues affect control accuracy
- **Energy Management** - Data gaps indicate measurement blind spots
- **Occupancy Analysis** - Quality patterns reveal usage patterns

### Healthcare IoT
- **Patient Monitoring** - Sensor issues affect care quality
- **Equipment Management** - Quality trends predict device failures
- **Workflow Analysis** - Data issues reveal process inefficiencies

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit: `git commit -am 'Add feature'`
5. Push: `git push origin feature-name`
6. Create a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{iot_data_quality_pipeline,
  title={IoT Data Quality Pipeline: Explainable Process Analytics},
  author={[Author Name]},
  year={2024},
  url={[Repository URL]}
}
```

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact [contact information].

---

**Note**: This system transforms the traditional view of data quality from "problems to be solved" to "insights to be leveraged" for better IoT system understanding and management.