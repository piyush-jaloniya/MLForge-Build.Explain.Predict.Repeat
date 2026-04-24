# ⚡ MLForge

Build. Explain. Predict. Repeat.

A full-stack no-code machine learning platform: upload data, preprocess, train models, evaluate, explain, and deploy — all via a browser.

---

## 🗂️ Project Structure

```
MLForge/
├── backend/                  FastAPI application
│   ├── main.py               App factory, CORS, middleware
│   ├── session_store.py      In-memory session store (TTL eviction)
│   ├── db/database.py        SQLAlchemy ORM + SQLite
│   ├── models/schemas.py     Pydantic request/response schemas
│   └── routers/
│       ├── data.py           Upload, preview, quality, sessions
│       ├── preprocess.py     14 preprocessing steps + undo/reset
│       ├── train.py          Model training + RL advisor
│       ├── predict.py        Single/batch prediction + eval metrics
│       ├── export.py         .pkl export, PowerBI CSV
│       ├── hyperopt.py       Optuna hyperparameter search
│       ├── xai.py            SHAP + permutation importance
│       ├── viz.py            15 Plotly chart endpoints
│       └── ai.py             9 Gemini AI endpoints
│
├── ml_engine/                ML core (framework-agnostic)
│   ├── models/
│   │   ├── classical.py      10 sklearn/XGBoost/LightGBM models
│   │   ├── trainer.py        Training pipeline + CV + MLflow
│   │   ├── registry.py       In-memory run registry (LRU eviction)
│   │   └── hyperopt.py       Optuna TPE search spaces
│   ├── preprocessing/        Cleaner, encoder, scaler, outlier, undo
│   ├── evaluation/           Metrics, SHAP explainability, data quality
│   └── visualizations/       18 Plotly chart builders
│
├── ai_services/
│   ├── gemini/service.py     9 Gemini 2.5 Flash AI functions
│   └── rl_advisor/bandit.py  Thompson Sampling model advisor
│
├── frontend/                 React 19 + Vite + Zustand
│   └── src/
│       ├── pages/            13 pages (Upload → Report)
│       ├── components/       PlotlyChart, ErrorBoundary
│       ├── store/            Zustand: session, model, UI stores
│       └── api/client.ts     Typed axios wrappers (25 functions)
│
├── data/samples/             iris.csv, titanic.csv, housing.csv
├── infra/docker/             Dockerfile.backend
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### 1. Backend

```bash
# from project root (this folder)

# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set GEMINI_API_KEY if you want AI features

# Start API server
uvicorn backend.main:app --reload --port 8000

# API docs at: http://localhost:8000/docs
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# App at: http://localhost:5173
```

### 3. Docker Compose (full stack)

```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:5173
# MLflow:   http://localhost:5001
```

---

## 🔌 API Overview (66 routes)

| Module | Routes | Description |
|--------|--------|-------------|
| `/api/data` | 7 | Upload, preview, quality, sessions, samples |
| `/api/preprocess` | 7 | Apply steps, undo, reset, feature select |
| `/api/train` | 7 | Start, status, cancel, runs, RL advisor |
| `/api/eval` | 3 | Metrics, compare, feature importance |
| `/api/predict` | 2 | Single prediction, batch CSV |
| `/api/export` | 2 | .pkl download, PowerBI CSV |
| `/api/hyperopt` | 4 | Optuna start/status/results/jobs |
| `/api/xai` | 4 | SHAP global, permutation, beeswarm, waterfall |
| `/api/viz` | 15 | All Plotly chart endpoints |
| `/api/ai` | 9 | Gemini schema, chat, report, explain, narrate |

---

## 🖥️ Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| Upload | `/` | Drag-drop + 3 sample datasets |
| Data Quality | `/quality` | Completeness/uniqueness/consistency scores |
| Visualize | `/viz` | 6 EDA chart types (Plotly) |
| Preprocess | `/preprocess` | 14 steps, undo stack, column selector |
| RL Advisor | `/rl-advisor` | Thompson Sampling model recommendations |
| Train | `/train` | 10 models, live progress polling |
| Hyperopt | `/hyperopt` | Optuna trial visualization |
| Compare | `/compare` | Model leaderboard |
| Metrics | `/metrics` | CV scores, confusion matrix, ROC curve |
| Explain (XAI) | `/xai` | SHAP beeswarm/waterfall, permutation |
| Predict | `/predict` | Form-based single prediction |
| AI Assistant | `/ai` | Gemini chat + insight cards |
| Report | `/report` | AI-generated markdown experiment report |

---

## ⚙️ Environment Variables

See `.env.example` for all options. Key variables:

| Variable | Default | Required |
|----------|---------|----------|
| `GEMINI_API_KEY` | *(empty)* | No (AI features gracefully disabled) |
| `SECRET_KEY` | `dev_secret_key...` | **Yes in production** |
| `ENV` | `development` | Set to `production` for prod |
| `UPLOAD_MAX_MB` | `500` | No |
| `CORS_ORIGINS` | `http://localhost:5173,...` | No |

---

## 🤖 Supported Models

**Classification:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, Decision Tree, KNN, Naive Bayes

**Regression:** Random Forest, Gradient Boosting, XGBoost, LightGBM, Ridge, SVM, Decision Tree, KNN

---

## 🔐 Production Checklist

- [ ] Set `SECRET_KEY` to a random 32+ char string
- [ ] Set `ENV=production`
- [ ] Restrict `CORS_ORIGINS` to your actual domain
- [ ] Set `GEMINI_API_KEY` if using AI features
- [ ] Use PostgreSQL instead of SQLite (`DATABASE_URL=postgresql://...`)
- [ ] Put behind a reverse proxy (nginx) with TLS
- [ ] Set `UPLOAD_MAX_MB` appropriate for your server RAM
