# ECG-Pomodoro

A Pomodoro timer web app with an ECG feature extraction service and an AI suggestion service.

**Group:** 2026-Chen-Chen-Liu  
**Authors:** Chen GuoZhu, Chen KunYu, Liu YungHsin
**License:** Apache-2.0, CC-BY-4.0

## Description

This project provides a Pomodoro timer UI and a demo pipeline that sends an ECG segment to an ECG service for feature extraction, then sends the extracted features to an AI service for generating suggestions/predictions.

Components:

- React frontend (UI + Demo Pipeline)
- ECG service (FastAPI): `POST /ecg/features`
- AI service (FastAPI): `POST /ai/predict`

## Requirements

- Node.js + npm (for frontend)
- Python 3.9+ + pip (for backend services)
- `uvicorn` (to run FastAPI services)

## API Keys

Gemini API (ecg-pomodoro/ai-service/.env)

## Installation

### 1) Frontend dependencies

In the directory that contains `package.json`:

```bash
npm install
```

### 2) Backend dependencies

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
```

Install required Python packages for the services.
(Use the `requirements.txt` files provided in the service folders if available.)

## Usage

Run all three services concurrently.

### 1) Start the frontend (port 3000)

In the directory that contains `package.json`:

```bash
npm start
```

Open: http://localhost:3000

### 2) Start ECG service (port 8001)

In a new terminal:

```bash
cd ecg-service
uvicorn main:app --reload --port 8001
```

ECG API base URL: http://127.0.0.1:8001  
Swagger UI: http://127.0.0.1:8001/docs

### 3) Start AI service (port 8002)

In a new terminal:

```bash
cd ai-service
uvicorn main:app --reload --port 8002
```

AI API base URL: http://127.0.0.1:8002  
Swagger UI: http://127.0.0.1:8002/docs

## Architecture

```
┌───────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  React Frontend   │──────>│   ECG Service    │──────>│    AI Service     │
│ (localhost:3000)  │       │ (127.0.0.1:8001) │       │ (127.0.0.1:8002)  │
└───────────────────┘       └──────────────────┘       └──────────────────┘
```

Frontend uses:

- `ECG_API = http://127.0.0.1:8001`
- `AI_API  = http://127.0.0.1:8002`

## Testing

### Manual end-to-end test (recommended)

1. Start ECG service (8001), AI service (8002), and frontend (3000).
2. In the web UI, open the **Demo Pipeline** tab.
3. Click **Run Demo**.
4. Verify the page shows:
    - A non-empty "ECG Features" JSON returned by the ECG service.
    - A non-empty "AI Prediction" JSON returned by the AI service.

### API sanity checks (optional)

- ECG service health:
    - `GET http://127.0.0.1:8001/health`
- AI service health:
    - `GET http://127.0.0.1:8002/health`

## Known Issues

- The system includes demo/testing flows; output quality depends on the input ECG segment length and signal quality.
- The project focuses on API integration and inter-service communication.

## License

Apache-2.0. See the `LICENSE` file.

## Disclaimer

This project is for educational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment.
