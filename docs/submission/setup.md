# Voice RAG — Setup & Reproducibility Guide

**HH Goa 2026 — Task 2 | Quickstart & Deployment**

---

## 1. Prerequisites

- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Node.js 18+ & npm
- Git

---

## 2. Quickstart (Local Environment)

### Step 1: Clone Repository
```bash
git clone <REPO_URL>
cd "GOA TASK 2"
```

### Step 2: Configure Environment Variables
```bash
cp .env.example .env
```
*(Optional: Add your `SARVAM_API_KEY` or `LLM_API_KEY` if testing live external APIs. For offline testing/demo, placeholders are sufficient as the system defaults to deterministic offline test fixtures).*

### Step 3: Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
cd ..
```

### Step 4: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 5: Start Backend Server
```bash
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at: `http://localhost:8000/docs`

### Step 6: Start Frontend Development Server
```bash
cd frontend
npm run dev
```
Interactive UI will be accessible at: `http://localhost:5173`

---

## 3. Running Automated Tests & Benchmarks

### Run Full Test Suite (127 Tests):
```bash
python -m pytest
```

### Run Production Benchmark Suite (141 Queries):
```bash
python -m benchmark.harness
```

---

## 4. Docker Deployment

To launch both backend and frontend via Docker:
```bash
docker compose up --build
```
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`
