# PLAN — crop-yield-forecast

A weekend portfolio project: forecast next year's crop yield by country and crop, with **honest evaluation** against a naive baseline, served with FastAPI + Streamlit in Docker, with a live Streamlit demo on Streamlit Community Cloud.

**Audience:** hiring managers for agritech / data science IC roles.
**What it should prove:** sound ML judgment (no leakage, a baseline to beat), MLOps basics (MLflow, Docker), and deployment (a public URL).

---

## 1. Working rules for Claude Code

- Read this file at the start of every session. Work on **one milestone per session**, and tick its checkboxes when it's done.
- **Stay in scope.** Anything in "Out of scope" needs my explicit OK.
- Ask before adding a dependency that isn't listed in §3.
- **Never commit** secrets (`kaggle.json`, any `.streamlit/secrets.toml`), raw data (`data/raw/`), MLflow runs, or `.venv/`.
- **Reported metrics come only from the time-based split.** The random split exists only to show the leakage.
- Keep code in `src/`, not in notebooks. Notebooks are for EDA and narrative only.
- Before calling a milestone done, run `uv run ruff check .` and `uv run pytest`.

### Teaching mode (always on)
I'm learning Docker, uv, MLflow and Streamlit Community Cloud in this project, so explain as you go:
- **Before each command or file change:** 1–2 lines on *what* it does and *why* it's needed. Keep it short, no lectures.
- **Go deeper on tools that are new to me:** `uv`, `Makefile`, `Dockerfile`, MLflow, FastAPI, the Streamlit Cloud deploy. Keep explanations minimal for pandas, scikit-learn and LightGBM (I already know them).
- **Pause and ask before:** creating the GitHub repo or the Streamlit Cloud app, any `git push`, installing anything outside the project, or anything that can't be easily undone.
- **At the end of each milestone:** a 3–5 bullet recap of what was built, plus the commands I'd run to reproduce it myself.
- If something fails, explain what the error means before fixing it.

---

## 2. Environment

| Item | Status |
|---|---|
| Mac (macOS) | ✅ |
| GitHub account | ✅ |
| Kaggle account + `~/.kaggle/kaggle.json` | ✅ |
| Claude Code | ✅ |
| Git, VS Code | ✅ |
| Python 3.13 via `uv` | ✅ |
| Docker Desktop for Mac | ✅ |
| Homebrew + `libomp` (LightGBM needs it on macOS) | ✅ |
| Python packages (§3) | ✅ installed in M0 via `uv` (Python 3.13) |

**Python version:** use 3.13. If any package fails to install on 3.13, switch with `uv python pin 3.12` and keep going. Don't spend time debugging it.

---

## 3. Dependencies (managed by `uv`, in `pyproject.toml`)

- **Core:** pandas, numpy, scikit-learn, lightgbm, mlflow, pyarrow
- **Serving:** fastapi, uvicorn, pydantic, streamlit, plotly
- **Tooling:** kaggle
- **Dev:** pytest, ruff, jupyter / ipykernel, httpx + httpx2 (for FastAPI's `TestClient`)
- **Stretch only:** shap

---

## 4. Data

- **Source:** Kaggle `patelris/crop-yield-prediction-dataset` (FAO yield and pesticides, World Bank rainfall and temperature)
- **Main file:** `yield_df.csv`. The other four files are its raw inputs.
- **Target:** `hg/ha_yield` (hectograms per hectare), by country (`Area`), crop (`Item`) and `Year`
- **Coverage:** about 100 countries, 10 crops, 1990–2013
- **Download:** `make data`, which runs `kaggle datasets download -d patelris/crop-yield-prediction-dataset -p data/raw --unzip`
- **Licensing:** raw CSVs are not committed. The small processed file the app needs (`app/data/panel.parquet`) is committed, with attribution to FAO and the World Bank in the README. Check the dataset's license terms before publishing.

---

## 5. Repo structure

```
crop-yield-forecast/
├── PLAN.md
├── CLAUDE.md            # one line: @PLAN.md
├── README.md            # demo link, results, model card
├── LICENSE              # MIT
├── .gitignore
├── pyproject.toml / uv.lock
├── Makefile             # data, eda, train, serve, app, docker-build, docker-run
├── Dockerfile
├── data/raw/            # gitignored
├── data/processed/      # gitignored
├── notebooks/01_eda.ipynb
├── src/cropyield/
│   ├── data.py          # load, dedupe, clean, build panel
│   ├── features.py      # lag features
│   ├── baseline.py      # last-year-yield baseline
│   ├── train.py         # LightGBM + MLflow logging
│   └── evaluate.py      # metrics, per-crop breakdown
├── api/main.py          # FastAPI: /health, /predict
├── app/streamlit_app.py # UI
├── app/requirements.txt # slim runtime deps for Streamlit Cloud (overrides root uv.lock there)
├── app/data/panel.parquet
├── models/model.txt     # small LightGBM model, committed for the live app
└── tests/
```

---

## 6. Milestones

### M0 — Setup (Friday night or Saturday, about 1 h)
- [x] Install Docker Desktop and `libomp`
- [x] Sanity check: `docker run --rm hello-world` works and Docker Desktop is running
- [x] Create the public GitHub repo `crop-yield-forecast` with the MIT license
- [x] `uv init`, add the dependencies from §3, commit `uv.lock`
- [x] Add `.gitignore` (§9), `CLAUDE.md` (containing `@PLAN.md`), `Makefile`
- [x] `make data` downloads the CSVs into `data/raw/`
- [x] First commit and push

### M1 — EDA (Saturday morning, 2–3 h, timeboxed)
Notebook: `notebooks/01_eda.ipynb`. It answers these questions, and each finding goes into the README:
- [x] **Duplicates:** are there repeated (Area, Item, Year) rows after the merge? How many, and how are they resolved?
- [x] **Coverage:** which country-crop pairs have a full 1990–2013 history, and where are the gaps? What minimum history is needed for lags?
- [x] **Scale:** how does yield range across crops (e.g. potatoes vs rice)? Decide between a log target and per-crop handling.
- [x] **Feature quality:** does rainfall change year to year, or is it a constant per country? How do pesticides and temperature correlate with yield?
- [x] **Trend:** how strong is the upward trend over time, and how well does last year's yield alone predict this year's?
- [x] **Thailand rice:** plot the full series (the app's showcase)
- [x] Record the decisions for M2 at the end of the notebook

### M2 — Baseline, model, MLflow (Saturday afternoon)
- [x] `data.py` builds a clean panel with one row per (Area, Item, Year)
- [x] Task: **one-year-ahead forecast.** Features must be known before the target year: yield lags 1–3, prior-year weather and pesticides, crop, country. Same-year weather is left out unless the EDA shows it adds something, and if it's used, that's documented as an assumption.
- [x] **Time split:** train ≤ 2008, test 2009–2013
- [x] **Baseline:** predict last year's yield
- [x] **Model:** LightGBM
- [x] **Leakage demo:** the same model on a random 80/20 split
- [x] **Metrics:** MAPE and RMSE on the log target, overall and per crop
- [x] MLflow logs three runs (`baseline`, `lgbm_time_split`, `lgbm_random_split`) to `sqlite:///mlflow.db`
- [x] Save `models/model.txt` and `app/data/panel.parquet`
- [x] Tests: no duplicate keys, no test years in the training data, lags never read future rows

### M3 — API and Docker (Sunday morning)
- [x] `api/main.py`: `GET /health`, and `POST /predict` taking `{area, item, year}` and returning the prediction plus the baseline
- [x] Pydantic validation (unknown country or crop → 422)
- [x] `Dockerfile` based on `python:3.13-slim` (or 3.12 if pinned), installing with `uv`
- [x] `make docker-build && make docker-run` works locally
- [x] Test for the API using FastAPI's `TestClient`

### M4 — Streamlit app and Streamlit Community Cloud deploy (Sunday midday)
- [x] `app/streamlit_app.py`: pick a country and crop (default: Thailand, Rice, paddy); chart of actual vs model vs baseline; a small table of test-period metrics
- [x] Add `app/requirements.txt` with only the app's runtime packages, and check the app runs from a clean venv built from it
- [x] Deploy to Streamlit Community Cloud from the GitHub repo (steps in §10)
- [x] The public URL loads from a cold start (the app sleeps when idle; the first load takes about a minute)
- [x] Docker runs the app locally on port 8501 (`make docker-run-app`), alongside the FastAPI container from M3

### M5 — README and model card (Sunday afternoon)
- [x] Top of the README: the live demo link, a screenshot, and one sentence on what it does
- [x] "What I found in the data" (from M1)
- [x] Results table: baseline vs LightGBM on the time split, plus the random split with an explanation of why it's inflated
- [x] Model card: intended use, limits (national-level, annual, data ends in 2013, not for farm decisions), data sources
- [x] Credits: FAO (FAOSTAT) and the World Bank, plus the Kaggle dataset author
- [x] How to run: `make data && make train && make app`

### Stretch (only if everything above is done)
- [ ] A SHAP "why this prediction" panel in the app
- [ ] A GitHub Action that runs ruff and pytest on push

---

## 7. Out of scope

- PlantVillage or any image data
- Synthetic datasets
- An LLM or RAG layer (the SHAP stretch goal only)
- Spark (the data is too small)
- Satellite (NDVI) data or live weather APIs (ideas for version 2)
- Hyperparameter tuning beyond a small grid
- Separate hosting for the API. FastAPI runs locally and in Docker; Streamlit Community Cloud serves only the Streamlit app.

---

## 8. Definition of done

- [x] Public GitHub repo with a clean README, MIT license, and passing tests
- [x] A live Streamlit Community Cloud URL
- [x] LightGBM beats the baseline on the time split. If it doesn't, the README says so honestly and explains why.
- [x] No secrets or raw data in the git history

---

## 9. `.gitignore`

Claude Code creates this in M0. Contents:

```
# Python / uv
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/

# Data (downloaded via `make data`)
data/raw/
data/processed/

# MLflow
mlruns/
mlflow.db
mlartifacts/

# Secrets — never commit
kaggle.json
.env
*.token

# macOS / editors
.DS_Store
.vscode/
```

`.gitignore` is a plain text file at the repo root. Git skips any file that matches these patterns, so `git add .` can't accidentally pick up your data or secrets.

---

## 10. Streamlit Community Cloud for a first-time user (M4)

1. Sign in at share.streamlit.io with GitHub.
2. Create app → choose the repo, branch `main`, main file `app/streamlit_app.py`.
3. Advanced settings → choose Python 3.12 or 3.13 to match the project.
4. **Dependencies:** Streamlit Cloud uses only **one** dependency file, and it looks in the app's folder before the repo root. Put a slim `app/requirements.txt` next to `streamlit_app.py` (streamlit, pandas, pyarrow, lightgbm, plotly, with versions matching `uv.lock`). It then wins over the root `uv.lock`, which would otherwise install everything, including MLflow and Jupyter, and slow down or break the build.
5. If LightGBM fails to import with an OpenMP error, add `packages.txt` at the repo root containing `libgomp1` (it's for apt packages, one per line).
6. Keep `models/model.txt` and `app/data/panel.parquet` committed, since the app reads them from the repo. Load them with paths relative to the script file, not the working directory.
7. Every `git push` to `main` redeploys the app automatically.

---

## 11. Risks

| Risk | Fallback |
|---|---|
| A package won't install on Python 3.13 | `uv python pin 3.12` |
| LightGBM import error on Mac | `brew install libomp` |
| The model doesn't beat the baseline | Report it honestly. That's still a strong portfolio story. |
| The Streamlit Cloud build fails on Sunday | Ship the GitHub repo plus a Docker run guide, and fix the deploy on Monday |
| The app sleeps after inactivity | Expected on the free tier; note in the README that the first load can take about a minute |
| EDA runs over time | Stop at 3 h and write down open questions instead |
