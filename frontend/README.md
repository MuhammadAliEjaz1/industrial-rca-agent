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
