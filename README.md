# Credit Card Fraud Detection API
ML-powered fraud detection system with real-time predictions via REST API and web interface.

## Features
- **Machine Learning Models**: Seven models trained on synthetic credit card transaction data.
- **REST API**: FastAPI backend with automatic documentation
- **Web Interface**: Interactive UI for testing predictions
- **Real-time Monitoring**: Track prediction statistics and model performance
- **Containerized Deployment**: Docker-ready for easy deployment

## Tech Stack
- **Backend**: FastAPI, Python 3.10
- **ML Libraries**: SKLearn, Pandas, NumPy, MatplotLib
- **Deployment**: Docker, Render
- **Frontend**: HTML, CSS, JavaScript

## Model Performance
- **Accuracy**: 0.XXXX
- **Recall**: 0.XXXX
- **Precision**: 0.XXXX
- **F1**: 0.XXXX
- **ROC_AUC**: 0.XXXX
- **True Positives**: XXXX
- **True Negatives**: XXXX
- **False Positives**: XXXX
- **False Negatives**: XXXX

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/channan1/fraud-detection-api.git
cd fraud-detection-api
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate (On Windows: venv\Scripts\activate)
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn main:app --reload
```

5. Open your browser to `http://localhost:8000`

### Docker Deployment
```bash
docker build -t fraud-detection-api
docker run -p 8000:8000 fraud-detection-api
```
**Note** First startup trains the models automatically (~1 minute).

Access the application at `http://localhost:8000` or `http://127.0.0.1:8000`

## API Documentation
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Demo
![Fraud Detection Demo](static/images/demo.gif)

## API Endpoints
- `GET /` - View web interface dashboard
- `POST /metrics/` - View model performance metrics
- `POST /obs_predict` - Predict fraud for a transaction
- `GET /health` - Returns health status of API.

## Use Cases
- Proof of concept for real-time fraud detection for payment processors.
- Risk assessment for financial institutions
- Transaction monitoring systems
- Educational demonstration of ML deployment

## Contact
**Carlos Hannan**
- LinkedIn: [linkedin.com/in/carlos-hannan](https://www.linkedin.com/in/carlos-hannan)
- GitHub: [@channan1](https://github.com/channan1)
- Email: carloshannan96@gmail.com
   
