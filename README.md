# Pune Air Quality Sentinel

A Python-based air quality monitoring and forecasting platform for Pune, India, featuring live environmental data, interactive visualisations, and experimental machine learning models.

## Overview

Pune Air Quality Sentinel (PAQS) is a project focused on monitoring and analysing air quality across different locations in Pune.

It combines environmental data from public APIs with data processing, visualisation, and machine learning techniques to explore air pollution patterns and forecasting.

The project consists of two main components:

- **Live Monitoring:** Retrieves air quality and weather forecasts using Open-Meteo and Copernicus CAMS, presenting the information through an interactive Streamlit dashboard and REST API.
- **Machine Learning:** An experimental research pipeline implementing ARIMA, LSTM, and Spatio-Temporal Graph Neural Networks (ST-GNN) for air quality forecasting.

## Features

- Air quality and weather forecasts for 10 predefined Pune locations.
- Hourly air quality and meteorological information.
- Interactive Streamlit dashboard.
- FastAPI backend for accessing forecast and prediction services.
- Historical air quality and weather data processing.
- Synthetic IoT sensor data generation.
- Experimental time-series and spatio-temporal forecasting models.
- Model evaluation and visualisation utilities.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Backend | FastAPI |
| Data Processing | Pandas, NumPy |
| Machine Learning | PyTorch, Scikit-learn |
| Forecasting | ARIMA, LSTM, ST-GNN |
| Visualisation | Matplotlib, Folium |
| Data Storage | CSV, Parquet, JSON |

## Data Sources

The project uses the following sources:

- [Open-Meteo](https://open-meteo.com/) – Air quality and weather forecasts, as well as historical weather data.
- [Copernicus CAMS](https://atmosphere.copernicus.eu/) – Atmospheric composition forecast data accessed through Open-Meteo.
- [OpenAQ](https://openaq.org/) – Historical air quality observations.
- [Kaggle](https://www.kaggle.com/) – Historical air quality datasets.

## Getting Started

### Requirements

- Python 3.11
- Git
- Internet connection for retrieving live environmental data.

### 1. Clone the repository

```bash
git clone https://github.com/xt67/pune-air-quality-sentinel.git

cd pune-air-quality-sentinel
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

For the dashboard and API:

```bash
pip install -r requirements.txt
```

For the additional machine learning and research dependencies:

```bash
pip install -r requirements-dev.txt
```

### 4. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard will be available at:

`http://localhost:8501`

### 5. Run the API

In a separate terminal:

```bash
uvicorn app.api.main:app --reload
```

API documentation:

`http://127.0.0.1:8000/docs`

## Machine Learning

The research pipeline includes three forecasting approaches:

- **ARIMA:** Statistical time-series forecasting and baseline comparison.
- **LSTM:** Deep learning for capturing temporal dependencies.
- **ST-GNN:** Graph-based modelling of spatial and temporal relationships.

The repository includes data preprocessing, model training, evaluation, and inference utilities.

The historical datasets and trained model checkpoints are not included in the default repository checkout. Additional data preparation and dependencies are required to reproduce the experiments.

## Project Structure

```text
pune-air-quality-sentinel/
│
├── app/
│   ├── api/                 # FastAPI application
│   └── streamlit_app.py     # Dashboard
│
├── configs/                 # Model and data configurations
├── data/                    # Data and graph metadata
├── docs/                    # Documentation and results
├── src/
│   ├── data/                # Data ingestion and preprocessing
│   ├── models/              # ML models and training
│   ├── evaluate/            # Model evaluation
│   ├── utils/               # Utilities
│   └── viz/                 # Visualisations
│
├── tests/                   # Automated tests
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Important Notes

- Live air quality values are external forecast estimates, not measurements from locally deployed sensors.
- The live application uses the US EPA AQI standard. The historical research pipeline also contains separate CPCB AQI calculation utilities.
- CAMS global forecast data has approximately 45 km spatial resolution, so nearby Pune locations may share similar readings.
- The live dashboard uses external provider forecasts independently of the experimental machine learning models.
- This project is intended for educational and research purposes, not as a certified environmental monitoring or health advisory service.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Rayan Rahman**

GitHub: [@xt67](https://github.com/xt67)
