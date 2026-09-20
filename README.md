# DermAI — Dermatology Chatbot with Skin Cancer Classification

> Published — **IJCACI 2025, Washington**

DermAI combines a skin-lesion image classifier with a grounded dermatology chatbot:

- **CNN classifier** over the **HAM10000** dataset across **7 lesion classes**
  (~97% classification accuracy).
- **Grad-CAM explainability** that surfaces the lesion regions driving each prediction.
- **Retrieval-Augmented Generation** pipeline (**LangChain + FAISS**) for grounded
  dermatology Q&A over curated sources, generated with **Groq** (free tier).

It ships two ways:

1. **Streamlit app** (`app.py`) — the original end-to-end app (upload, inference, chat), run locally.
2. **Web + API split** for deployment — a **Next.js** frontend on **Vercel** and a
   **FastAPI** backend on **Render**, both driven by the same `dermai` engine.

## Repo layout

```
dermai/                  Shared engine (imported by both the Streamlit app and the API)
  classifier/            7-class HAM10000 CNN, Grad-CAM, inference, training
  rag/                   LangChain + FAISS RAG, Groq LLM
app.py                   Streamlit app (local, end-to-end)
apps/
  api/                   FastAPI backend  -> Render   (wraps dermai)
  web/                   Next.js frontend -> Vercel   (calls the API)
packages/shared/         Shared TypeScript types
models/best_model.h5     Trained CNN weights
data/sources/            Curated dermatology knowledge base
render.yaml              Render deployment for the API
```

## The 7 HAM10000 classes

`akiec` (actinic keratosis), `bcc` (basal cell carcinoma), `bkl` (benign keratosis),
`df` (dermatofibroma), `nv` (melanocytic nevi), `vasc` (vascular / pyogenic granuloma),
`mel` (melanoma).

## Run locally

### Option A — Streamlit (single app)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add GROQ_API_KEY for live chat (optional)
streamlit run app.py      # http://localhost:8501
```

### Option B — Web + API (mirrors production)

Backend (FastAPI):
```bash
source .venv/bin/activate
pip install -r apps/api/requirements.txt
export GROQ_API_KEY=your_key            # optional; falls back to extractive
uvicorn app.main:app --app-dir apps/api --reload --port 8000
```

Frontend (Next.js):
```bash
npm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev:web   # http://localhost:3000
```

## Deployment

Same split as before:

- **Web → Vercel.** Set the project **Root Directory** to `apps/web` and add env var
  `NEXT_PUBLIC_API_BASE_URL` = your Render API URL. Push to `main` and Vercel rebuilds.
- **API → Render.** `render.yaml` defines a Docker web service (`apps/api/Dockerfile`).
  Set `GROQ_API_KEY` in the Render dashboard. TensorFlow + FAISS need memory — use at
  least a Standard instance.

> Note: the API (TensorFlow/FAISS) cannot run on Vercel — that's why it lives on Render.
> Vercel only serves the Next.js frontend.

## API endpoints

- `GET /health` — status, model availability, LLM backend
- `POST /chat` — `{ message, history }` → grounded answer + sources
- `POST /classify` — multipart image → 7-class prediction, confidence, Grad-CAM overlay, probabilities

## Retraining the classifier

`models/best_model.h5` ships pre-trained. To reproduce on HAM10000:

```bash
pip install -r requirements.txt -r requirements-train.txt
python -m dermai.classifier.train            # downloads HAM10000 via kagglehub
```

## Notes

- DermAI is **informational only** and does not provide a medical diagnosis.
- Chat uses **Groq** when `GROQ_API_KEY` is set, otherwise an offline extractive fallback.
