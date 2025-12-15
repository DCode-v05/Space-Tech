# GNSS Error Prediction System

## Project Description
A machine learning system for predicting GNSS (Global Navigation Satellite System) errors. This dashboard provides visualization of GNSS error predictions to improve positioning accuracy in space technology applications. The solution includes a FastAPI backend for predictions and a Streamlit dashboard for real-time visualization.

---

## Project Details

### Problem Statement
Accurate positioning is critical for space technology applications. However, GNSS signals are subject to various errors such as ionospheric delays, clock errors, and orbit inaccuracies. Predicting these errors helps improve the precision of navigation systems.

### Data Preprocessing
- **Missing Values:** Handled during the data loading pipeline.
- **Feature Engineering:**
  - Timestamp conversion and normalization.
  - Sequence creation for time-series models (LSTM/Transformer).
  - Feature normalization to ensure stable model training.
- **Processing:** Raw GNSS data is processed into sequences suitable for deep learning models.

### Model Architecture
- **Models Used:**
  - **LSTM (Long Short-Term Memory):** Captures temporal dependencies in the error sequences.
  - **Transformer:** Handles long-range dependencies and complex patterns.
  - **Ensemble:** A weighted combination of LSTM and Transformer predictions for optimal performance.
- **Training:** Dynamic weight adjustment during training.

### Web Application
The Streamlit app provides:
- **Real-time Predictions:** Visualizes the predicted errors against actual data.
- **Model Performance:** Displays metrics like RMSE and MAE.
- **Interactive Plots:** Users can explore error distributions and confidence intervals.
- **Responsive Design:** Optimized for both desktop and various screen sizes.

---

## Tech Stack
- **Languages:** Python 3.8+
- **Machine Learning:** PyTorch, NumPy, Scikit-learn
- **Web Frameworks:** FastAPI (Backend), Streamlit (Frontend)
- **Containerization:** Docker, Docker Compose
- **Data Processing:** Pandas

---

## Getting Started

### 1. Clone the repository
```
git clone https://github.com/DCode-v05/Space-Tech.git
cd Space-Tech
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the App or Training
To train models:
```
python -m src.models.train
```
To launch the Streamlit app:
```
streamlit run src/dashboard/app.py
```

---

## Usage
- **Dashboard:** Open `http://localhost:8501` to view the interactive visualizations.
- **API:** Send POST requests to `http://localhost:8000/predict` with sequence data to get error predictions.
- **Training:** Run the training script to retrain the models on new data. Configuration can be adjusted in `config/config.yaml`.

---

## Project Structure
```
SpaceTech/
│
├── config/                    # Configuration files
│   └── config.yaml           # Main configuration
├── data/                     # Data directory
│   ├── raw/                  # Raw GNSS data files
│   └── processed/            # Processed data files
├── models/                   # Model files
│   └── saved/                # Saved model checkpoints
├── src/                      # Source code
│   ├── api/                  # FastAPI service
│   ├── dashboard/            # Streamlit dashboard
│   ├── data/                 # Data processing scripts
│   ├── models/               # Model implementations (LSTM, Transformer)
│   ├── training/             # Training pipeline
│   ├── evaluation/           # Metrics and visualization tools
│   └── utils/                # Utility functions
├── tests/                    # Test suite
├── docker-compose.yml        # Docker composition
├── Dockerfile                # API Dockerfile
├── Dockerfile.dashboard      # Dashboard Dockerfile
└── README.md                 # Project documentation
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com