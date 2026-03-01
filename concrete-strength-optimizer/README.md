# NirmaanAI_demo
Functional prototype of NirmaanAI for createch 2026
# NirmaanAI Demo

## AI-Powered Concrete Curing Optimization for Precast Construction

---

## Overview

NirmaanAI is an intelligent decision support system that optimizes concrete curing cycles using machine learning and real-time weather data. The system predicts concrete strength development and recommends optimal demoulding times, reducing energy consumption and maximizing production efficiency.

Built for the L&T CreaTech Hackathon, this prototype demonstrates how AI can transform traditional construction processes through data-driven insights and automation.

---

## Key Features

### 1. Machine Learning Core
- Random Forest regression model trained on 164 real curing data samples
- 95% prediction accuracy (R² score)
- Feature importance analysis: curing time (59%), humidity (18%), temperature (13%), curing method (10%)

### 2. Weather Integration
- Real-time weather fetch via Open-Meteo API
- Automatic detection of ambient curing conditions
- Location-based optimization for major Indian cities

### 3. Energy Arbitrage Intelligence
- Automatic steam boiler disable when natural conditions suffice
- Conditions: Temperature ≥30°C and Humidity ≥70% create natural steam-room effect
- Daily diesel savings up to ₹84,000 per curing cycle

### 4. Cost Optimization
- Dynamic cost calculation based on curing method
- Comparison across multiple curing strategies
- ROI analysis with per-element cost breakdown

### 5. Interactive Dashboard
- Professional, corporate-grade UI
- Real-time parameter adjustment
- Visual strength development curves
- Sensitivity analysis for key parameters

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Backend | Python HTTP Server |
| Machine Learning | scikit-learn (Random Forest Regressor) |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Weather API | Open-Meteo (free, no API key required) |
| Version Control | Git, GitHub |

---

## System Architecture

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ │ │ │ │ │
│ HTML/CSS UI │────▶│ Python Server │────▶│ ML Model │
│ (index.html) │ │ (server.py) │ │ (Random Forest)│
│ │◀────│ │◀────│ │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│ │ │
│ │ │
▼ ▼ ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Weather API │ │ Cost Engine │ │ Training Data │
│ Open-Meteo │ │ ₹/hour rates │ │ dataset.csv │
└─────────────────┘ └─────────────────┘ └─────────────────┘


---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Modern web browser (Chrome, Firefox, Edge)

### Step 1: Clone Repository

git clone https://github.com/BhargavMalakonda/NirmaanAI_demo.git
cd NirmaanAI_demo

## Usage Guide
1. Input Parameters
Project Details: Select project type, region, concrete grade

Mix Design: Choose cement type, w/c ratio, admixtures

Curing Method: Select from Steam, Water, Membrane, Heat

Climatic Conditions: Enter city for live weather or manual values

2. Run Analysis
Click "Run Analysis" to trigger ML prediction and optimization

3. Interpret Results
Optimal Cycle Time: Recommended demoulding hours

Predicted Strength: Expected concrete strength at demoulding

Cost Analysis: Total cost with method comparison

Energy Arbitrage: Automatic recommendation for steam disable

4. Test Scenarios
Scenario	          Temperature	Humidity	Expected Result
Natural Steam Room	35°C	      85%	      Steam    Disabled
Partial Ambient	    28°C	      68%	      Steam    Optional
Steam Required	    18°C	      45%	      Steam    Active

## model performance

=== Model Metrics ===
Train R² Score:    0.9852
Test R² Score:     0.9556
Test RMSE:         2.24 MPa
Test MAE:          1.68 MPa
Cross-val R²:      0.9450 (±0.0340)

=== Feature Importance ===
hours:      0.5926 (59.3%)
humidity:   0.1823 (18.2%)
temperature: 0.1303 (13.0%)
curing_type: 0.0948 (9.5%)

