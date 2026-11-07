# Root-Cause Monitor (dashboard)

React + TypeScript + Vite + Recharts. Talks to the FastAPI backend in `../backend`.

## Run

    # terminal 1 (from backend/, venv active)
    uvicorn api.main:app --port 8000

    # terminal 2 (from frontend/)
    npm install
    copy .env.example .env        # only needed if the API is not on http://localhost:8000
    npm run dev                   # http://localhost:5173

## Check

    npm test                      # 24 tests, no browser and no backend needed
    npm run build                 # type-checks and bundles

## What is on screen

- **Fleet**: every machine with a status light (a rule check of its latest reading, not a prediction).
- **Machine > Monitor**: latest reading, the documented limit checks, six sensor charts with the limits drawn in.
- **Machine > Investigate**: start an investigation (failure reported, routine check, or replay a recorded incident),
  watch the timeline fill in live, then read the report: conclusion, hypotheses with evidence for and against,
  observed facts, ML prediction, and what was considered and rejected.
- **Machine > History**: recorded failures and (random demo) maintenance records.

## Recorded demo (free hosting, no card, no backend)

Free hosts no longer run a Python backend without a payment card, so the dashboard has a **recorded mode**: it reads
JSON files instead of calling the API, and "Investigate" replays real, saved investigations step by step. Nothing here
calls an AI model, so it cannot run out of tokens.

    # 1. from backend/ (venv active): export the data and the 37 recorded test-run investigations
    python evaluation/09_export_demo.py

    # 2. from frontend/: build the static site
    npm run build:demo            # output in dist/

`dist/` is a plain static site with relative paths, so it works from any folder. Preview it with `npx vite preview`.

### Hosting options

- **Hugging Face static Space** (free for everyone; creating Docker or Gradio Spaces now needs a paid plan): create a Space
  with SDK "Static", then upload the *contents* of `dist/` (including its `README.md`) to it.
- **GitHub Pages**: free for public repositories. To keep the main repository private, push `dist/` to a separate public
  repository and enable Pages for its branch.
- Netlify or Cloudflare Pages also host static folders; check their current sign-up requirements.

Only the exported demo data is public: sensor readings, recorded failures and the saved reports. Never commit `.env`.
