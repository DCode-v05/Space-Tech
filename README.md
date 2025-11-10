# GNSS Error Prediction System

A machine learning system for predicting GNSS (Global Navigation Satellite System) errors. This dashboard provides visualization of GNSS error predictions to improve positioning accuracy in space technology applications.

## Dashboard Setup and Usage

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository (if not already done):
   ```bash
   git clone <repository-url>
   cd SpaceTech
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r dashboard_requirements.txt
   ```

### Running the Dashboard

1. Start the Streamlit dashboard:
   ```bash
   streamlit run src/dashboard/app.py
   ```

2. The dashboard should automatically open in your default web browser. If not, navigate to:
   ```
   http://localhost:8501
   ```

### Features

- **Interactive Visualizations**: View GNSS error predictions and metrics
- **Real-time Updates**: See predictions update in real-time
- **Multiple Views**: Explore different aspects of the GNSS error data
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
SpaceTech/
├── config/                    # Configuration files
│   └── config.yaml           # Main configuration
├── data/                     # Data directory
│   ├── raw/                  # Raw GNSS data files
│   └── processed/            # Processed data files
├── models/                   # Model files
│   └── saved/                # Saved model checkpoints
├── src/                      # Source code
│   ├── api/                  # FastAPI service
│   │   ├── endpoints.py      # API endpoints
│   │   ├── main.py           # FastAPI app
│   │   └── service.py        # Model service
│   ├── data/                 # Data processing
│   │   ├── data_processor.py # Basic processor
│   │   └── advanced_processor.py # Advanced processor
│   ├── models/               # Model implementations
│   │   ├── model.py          # Basic models
│   │   ├── advanced_model.py # Advanced models
│   │   └── train.py          # Training scripts
│   ├── training/             # Training pipeline
│   │   └── trainer.py        # Training utilities
│   ├── evaluation/           # Model evaluation
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── visualization.py  # Visualization tools
│   └── utils/                # Utility functions
│       └── data_utils.py     # Data utilities
├── tests/                    # Test suite
├── dashboard.py              # Streamlit dashboard
├── train.py                  # Training script
├── Dockerfile                # Dockerfile for API service
├── Dockerfile.dashboard      # Dockerfile for dashboard
├── docker-compose.yml        # Docker Compose configuration
└── requirements.txt          # Python dependencies
```

## Installation

### Prerequisites

- Python 3.9+
- pip
- Docker and Docker Compose (for containerized deployment)

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SpaceTech.git
cd SpaceTech
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Installation

1. Build and start the services:
```bash
docker-compose up --build
```

This will start:
- API service at http://localhost:8000
- Streamlit dashboard at http://localhost:8501

## Usage

### Running the API

Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

The API documentation will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Running the Dashboard

Start the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

### Making Predictions

You can make predictions using the API:

```python
import requests

url = "http://localhost:8000/api/v1/predict"
payload = {
    "satellites": ["G01", "G02"],
    "horizon_hours": 24,
    "return_confidence": True
}

response = requests.post(url, json=payload)
predictions = response.json()
```

### Training the Model

To train a new model:

```bash
python train.py --config config/config.yaml
```

## API Endpoints

- `POST /api/v1/predict`: Get predictions for one or more satellites
- `GET /api/v1/health`: Check API health status
- `GET /api/v1/info`: Get information about available models and satellites

## Configuration

Edit `config/config.yaml` to configure:
- Model parameters
- Training settings
- Data paths
- API settings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the open-source community for the amazing libraries used in this project.
- Special thanks to the contributors who helped improve this project.

### Training the Model

Train the ensemble model using:

```bash
python -m src.models.train
```

This will:
- Load and preprocess the GNSS data
- Train the LSTM and Transformer models
- Save the trained model and configuration

### Starting the Prediction Service

Start the FastAPI service:

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000` with endpoints:
- `POST /predict`: Make predictions
- `GET /health`: Check service health

### Running the Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run src.dashboard.app
```

Visit `http://localhost:8501` to access the dashboard features:
- Real-time predictions
- Model performance metrics
- Error distribution visualization

### Running Tests

Execute the test suite:

```bash
python -m pytest
```

This will run all tests and generate a coverage report.

## API Documentation

### Prediction Endpoint

```python
POST /predict

Request Body:
{
    "sequence": List[List[float]],  # Input sequence
    "prediction_horizon": int        # Number of steps to predict
}

Response:
{
    "predictions": List[float],
    "confidence_intervals": {
        "lower_95": List[float],
        "upper_95": List[float]
    }
}
```

## Model Architecture

The system uses an ensemble of:
1. **LSTM Model**: For capturing temporal dependencies
2. **Transformer Model**: For handling long-range dependencies

The ensemble weights are dynamically adjusted during training to optimize performance.

## Data Processing

The data processing pipeline includes:
- Missing value handling
- Timestamp conversion
- Feature normalization
- Sequence creation for time series prediction

## Testing

The test suite covers:
- Data processing functionality
- Model components and training
- API endpoints and error handling
- Dashboard visualization components

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GNSS data provided by [Your Data Source]
- Built with PyTorch, FastAPI, and Streamlit