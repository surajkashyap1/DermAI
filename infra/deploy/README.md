# Deploy Notes

DermAI is deployable today with this split:

- web: Vercel
- api: Render

## Render API

The repository includes a root `render.yaml` blueprint for the FastAPI service.

Important details:

- the API Docker image targets Python 3.11
- the image rebuilds the local seed corpus during deploy
- `DERMAI_GROQ_API_KEY` must be added in Render as a secret environment variable
- Render should use `/health` as the health check path

Recommended custom domain:

- `api.your-domain.com`

## Vercel Web

Recommended project settings:

- framework: Next.js
- root directory: `apps/web`
- install command: `npm install`
- build command: `npm run build:web`

Required environment variable:

- `NEXT_PUBLIC_API_BASE_URL=https://api.your-domain.com`

Recommended custom domain:

- `your-domain.com`

## Current Limits

- chat is real but still uses a seed corpus rather than a mature production corpus
- vision is a heuristic demo pipeline, not a trained diagnostic model
- no persistent database or object storage is required yet for the current demo deploy
