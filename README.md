# Pokémon Card Fair-Value Model

AIPI 510 Module Project 3: a reproducible ML pipeline that estimates the fair market price of
a Pokémon trading card from its attributes, so a buyer can tell whether an asking price is a
**deal**, **fair**, or **overpriced**.

The model needs details like rarity and collector number, so the app starts with a **semantic
card search**. You type "charizard ex 151" or "fiery flying dragon", pick the card, and its
details go straight to the model.

> **Live Web Application:** [https://data-sourcing-ml-pipeline-cmfyvuectyqmuewhadcq6b.streamlit.app/](https://data-sourcing-ml-pipeline-cmfyvuectyqmuewhadcq6b.streamlit.app/)  
> **Production API (Cloud Run):** [https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app](https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app)

## Links

| Resource | URL / Location |
|---|---|
| **Front-End Web App (Streamlit)** | [https://data-sourcing-ml-pipeline-cmfyvuectyqmuewhadcq6b.streamlit.app/](https://data-sourcing-ml-pipeline-cmfyvuectyqmuewhadcq6b.streamlit.app/) |
| **API Backend (Cloud Run)** | https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app |
| **API Documentation (Swagger UI)** | https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app/docs |
| **API Health Check** | https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app/health |
| **Data Snapshot (GCS)** | `gs://data-sourcing-ml-pipeline-data/snapshots/cards_2026-11-23.parquet` ([direct download](https://storage.googleapis.com/data-sourcing-ml-pipeline-data/snapshots/cards_2026-11-23.parquet)) |
| **Data Source Mirror** | [TCGCSV](https://tcgcsv.com) (TCGplayer catalog & prices, Category 3) |
| **Semantic Search Foundation** | [colbyrobbins/Semantic-Search-Engine](https://github.com/colbyrobbins/Semantic-Search-Engine) |

## Problem

- **Task:** supervised regression.
- **Target:** a card's TCGplayer **market price** in US dollars. The model predicts
  `log(1 + price)` because prices are heavily skewed, from $0.01 to $9,200.
- **Inputs:** attributes a buyer can see on the card or listing:
  - rarity and printing (normal, holo, reverse holo, 1st edition)
  - card type, stage, HP and retreat cost
  - collector number and set release date
  - whether the card has a second attack
- **Main metric:** RMSE of log price. It treats a $1 miss on a $2 card like a $50 miss on a
  $100 card.
- **Also reported:** R² on log price, plus the median absolute error in dollars and in percent.

## Data

| | |
|---|---|
| Source | [TCGCSV](https://tcgcsv.com), a free nightly mirror of TCGplayer's catalog and prices (category 3 = Pokémon) |
| Snapshot | `2026-11-23`, one day's prices |
| After cleaning | **42,352** single cards (product × printing) with a market price, from 28,331 products in 213 sets |
| Cleaning | drop sealed products (no rarity), code cards, unconfirmed rarities and rows without a price; parse HP, retreat cost and dates; strip HTML from attack and card text |
| Storage | the cleaned snapshot is a single parquet file, uploaded to Google Cloud Storage (see [Cloud deployment](#cloud-deployment-gcp)) |

TCGCSV prices change every day, so the pipeline never trains on live data. `pipeline.ingest`
freezes a dated snapshot, and `config.yaml` pins its date and SHA-256 checksum. Training
refuses a file that doesn't match, so every run on the same config sees identical data.

The listing prices (low, mid and high) are kept in the snapshot but **excluded from the
features**. They track the market price almost exactly (correlation 0.97) and would leak the answer.

Prices are TCGplayer data mirrored by TCGCSV. This project uses them for non-commercial
coursework only.

## Pipeline

```
TCGCSV API ──► ingest.py ──► cleaned snapshot (parquet) ──► GCS bucket
                                                               │
                         train.py ◄── data.py (download + verify SHA-256)
                            │
           train / validation / test split (grouped by product)
                            │
         fit candidates on train ─► compare on validation ─► refit best on train+val
                            │
               score once on test ─► MLflow (params, metrics, plots, model)
                            │
                artifacts/model.joblib + model_metadata.json ─► API /predict

              snapshot ──► search.py build ──► ChromaDB index ─► API /search
```

| Module | Role |
|---|---|
| `pipeline/ingest.py` | Fetch every Pokémon set's products and prices, flatten and clean them, save a dated snapshot, and optionally upload it to GCS |
| `pipeline/storage.py` | GCS upload/download. Reading falls back to anonymous access for public buckets |
| `pipeline/data.py` | Load the configured snapshot (from GCS or a verified local copy), then split it into train/validation/test, keeping all printings of a card in the same split |
| `pipeline/features.py` | `CardFeatures`: turn raw card fields into model features (set age, vintage era, secret-rare numbering, name flags such as `ex`/`VMAX`/`Charizard`/eeveelutions) |
| `pipeline/model.py` | Candidate models (gradient boosting, random forest, ridge, two-stage hurdle), each a full sklearn pipeline from raw fields to dollars |
| `pipeline/train.py` | Run the pipeline and log everything to MLflow |
| `pipeline/evaluate.py` | Metrics and diagnostic plots |
| `pipeline/predict.py` | Load the model and predict fair values / deal verdicts. This is what the API calls |
| `pipeline/search.py` | Build a ChromaDB index of every card (sentence-transformer embeddings) and search it. Results come back ready to send to the model |
| `config.yaml` | Every setting: data source, snapshot, storage, features, split sizes, model hyperparameters, tracking |

Feature engineering lives inside the saved model, so the API only has to pass raw card fields.

## Results (snapshot 2026-11-23)

Validation scores for each candidate (6,416 cards):

| Model | RMSE (log) | R² (log) | Notes |
|---|---|---|---|
| **Histogram gradient boosting** | **0.621** | **0.827** | **Selected** |
| Random forest | 0.638 | 0.818 | |
| Two-stage hurdle | 0.644 | 0.814 | Classifier gate for cards ≥ $25, plus separate bulk and chase regressors blended by the gate's probability |
| Ridge regression | 0.920 | 0.621 | |

The best model, refit on train + validation and scored once on the held-out test set
(6,309 cards):

| RMSE (log) | R² (log) | Median abs. error | Median abs. % error | Mean abs. error |
|---|---|---|---|---|
| 0.627 | 0.820 | $0.73 | 49% | $12.71 |

**The hurdle model didn't help.** We added it to fix the under-valuation of expensive chase
cards, but it scored worse on validation than a single gradient-boosting model, so the
pipeline kept gradient boosting. The cut at $25 leaves the chase regressor few examples, and
a wrong gate decision costs more than the blend saves. What did help was the new features
(vintage era, eeveelution and chase-legendary name flags): test RMSE went from 0.657 to 0.627
compared with the previous version. The two runs use different daily snapshots, so that
comparison is approximate.

The typical card is cheap, so the median error is under a dollar. The mean error is pulled up
by a few very expensive cards that the attributes can't explain, such as vintage 1st editions
and chase cards of popular Pokémon. The model tends to over-value bulk cards and under-value
the most expensive ones. Both plots are logged to MLflow with each run.

## Setup

Requires Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Get the data

Training reads the snapshot pinned in `config.yaml`.

- **If the snapshot is in GCS:** set `storage.backend: gcs` and `storage.gcs_bucket`, and
  training downloads it automatically.
- **To build a snapshot yourself:**

```bash
python -m pipeline.ingest                  # fetch today's prices (~2 min), clean, save to data/processed/
python -m pipeline.ingest --upload         # ...and upload to gs://<bucket>/snapshots/
python -m pipeline.ingest --skip-fetch --date 2026-11-23   # rebuild from raw files already downloaded
```

The command prints the snapshot's date and SHA-256. Copy them into `data.snapshot_date` and
`data.snapshot_sha256` to train on it.

### 2. Train

```bash
python -m pipeline.train                   # or: --config path/to/other.yaml
```

This writes `artifacts/model.joblib` and `artifacts/model_metadata.json`, and logs a run to MLflow.

### 3. Look at experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open <http://127.0.0.1:5000>. Each training run has one child run per candidate model,
and the parent run records:
- the config file, the snapshot date and hash, and the git commit
- validation and test metrics
- the plots and the candidate comparison table
- the logged model

### 4. Predict from the command line

```bash
python -m pipeline.predict '{"name": "Charizard ex - 199/165", "rarity": "Special Illustration Rare",
  "printing": "Holofoil", "card_type": "Fire", "stage": "Stage 2", "hp": 330, "retreat_cost": 2,
  "number": "199/165", "set_published": "2023-09-22", "is_supplemental": false, "has_attack2": true}' --asking 120
```

```json
{ "fair_value": 288.81, "asking_price": 120.0, "verdict": "deal" }
```

A price more than 15% below fair value is a deal, and more than 15% above is overpriced.

### 5. Search for a card

```bash
python -m pipeline.search build                           # embed all 28,331 cards (~75 s, downloads the model once)
python -m pipeline.search query "charizard ex 151"
```

```
1.142  Charizard ex - 199/165  [SV: Scarlet & Violet 151, Special Illustration Rare]  Holofoil $277.65
1.120  Charizard ex - 183/165  [SV: Scarlet & Violet 151, Ultra Rare]  Holofoil $30.57
1.106  Charizard ex - 006/165  [SV: Scarlet & Violet 151, Double Rare]  Holofoil $7.57
```

**How search works:** this builds on Colby's
[semantic search activity](https://github.com/colbyrobbins/Semantic-Search-Engine).
- **Index:** each card becomes one document: name, set, year, rarity, type, HP, printings,
  attacks and card text. Documents are embedded with
  `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB under `artifacts/search_index/`.
- **Two kinds of query:** embeddings alone handle descriptions well ("fiery flying dragon",
  "discard your hand and draw 7 cards"). They often miss exact lookups such as
  "charizard ex 199/165", where a person types a name and number.
- **Hybrid ranking:** candidates come from both the nearest embeddings and the best keyword
  matches on name, set, number, rarity and printing. The score is cosine similarity plus
  `keyword_weight` times the keyword match. Rare words count more (IDF weighting), so
  "charizard" matters more than "set".
- **Speed:** a search takes about 30 ms once the model is loaded.
- **Using results:** `model_input(card, printing)` turns a result into the model's input fields.

Settings are under `search:` in `config.yaml`.

### Tests

```bash
python -m pytest
```

The tests run offline on synthetic TCGCSV-shaped data and cover:
- cleaning, and that the same input always produces the same snapshot checksum
- the checksum check and GCS download (mocked)
- feature engineering, and that the split keeps each card's printings together
- an end-to-end training run with MLflow logging, and prediction
- building and querying the search index (with a stand-in embedder, so no model download)
- the API's endpoints and input validation (with a fake model and search)

## API

The service in `api/main.py` is built with FastAPI. Interactive docs are served at `/docs`.

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | | model name, snapshot date, test metrics, whether search is loaded |
| `POST /predict` | card fields as JSON (`name` and `rarity` required) plus optional `asking_price` | `fair_value`, and `verdict` (`deal` / `fair` / `overpriced`) when an asking price is given |
| `GET /search?q=...&k=5` | free-text query, 1–25 results | matching cards with images, market price per printing, and a ready-to-send `card` body for `/predict` |

Run it locally (after training and building the search index):

```bash
uvicorn api.main:app --port 8080
```

```bash
curl -s http://127.0.0.1:8080/health
```

```bash
curl -s -G http://127.0.0.1:8080/search --data-urlencode "q=umbreon vmax alternate art" -d k=3
```

```bash
curl -s -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" -d '{
  "name": "Charizard ex - 199/165", "rarity": "Special Illustration Rare", "printing": "Holofoil",
  "card_type": "Fire", "stage": "Stage 2", "hp": 330, "retreat_cost": 2, "number": "199/165",
  "set_published": "2023-09-22", "has_attack2": true, "asking_price": 120}'
```

```json
{"fair_value": 217.78, "asking_price": 120.0, "verdict": "deal", "model_name": "hist_gbm", "snapshot_date": "2026-11-23"}
```

**Settings (environment variables)**
- `MODEL_DIR`: folder with the model files. Defaults to `artifacts/`.
- `CONFIG_PATH`: config file to read. Defaults to `config.yaml`.
- `ENABLE_SEARCH=0`: serve `/predict` only, without loading the search index or embedding model.

## Front-End Application (Streamlit)

A web interface is included in `frontend/app.py` for end users to search cards, view card art, and evaluate asking prices against the deployed model API.

- **Live Deployed Web Application**: [https://data-sourcing-ml-pipeline-cmfyvuectyqmuewhadcq6b.streamlit.app/](https://data-sourcing-ml-pipeline-cmfyvuectyqmuewhadcq6b.streamlit.app/)

### Run locally:
```bash
streamlit run frontend/app.py
```
By default, the front end calls the live Cloud Run API (`https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app`). To use a local server instead, set `API_URL=http://127.0.0.1:8080` or paste the URL into the sidebar settings.

### Cloud Deployment:
The application is deployed on **Streamlit Community Cloud** tracking the `main` branch. It automatically installs dependencies from `frontend/requirements.txt` to keep the front-end container lightweight and responsive.

## Docker

One image runs every part of the project. The first argument picks the command, and with no
argument it serves the API.

**1. Build the image**

```bash
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t pokemon-fair-value .
```

`GIT_COMMIT` is optional. It lets MLflow runs inside the container record which commit they came from.

**2. Train and build the search index inside the container**

Mount `data/` to read the snapshot and `artifacts/` to keep the outputs. MLflow runs from the
container go to `mlruns-docker/` because the file paths MLflow records differ between your
machine and the container.

```bash
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" -v "$PWD/mlruns-docker:/app/mlruns" pokemon-fair-value train
```

**3. Rebuild and serve**

The image includes `artifacts/` when it exists, so rebuild after training and the server
needs no mounts.

If the build context has no `artifacts/model.joblib`, the build trains the model and builds
the search index itself. This happens with `gcloud builds submit`, which skips git-ignored
files. That build step downloads the snapshot from the GCS bucket, so it needs network access
and adds about 5 minutes.

```bash
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t pokemon-fair-value . && docker run --rm -p 8080:8080 pokemon-fair-value
```

**Other commands**

```bash
docker run --rm pokemon-fair-value test                       # run the test suite in the container
docker run --rm -v "$PWD/data:/app/data" pokemon-fair-value ingest   # fetch a new snapshot
docker run --rm -p 5000:5000 -v "$PWD/mlruns-docker:/app/mlruns" pokemon-fair-value mlflow-ui
```

The image installs the CPU-only build of PyTorch, and it includes the embedding model so the
server doesn't download it at startup. The Docker image is approximately 1.1 GB, the server starts up
within seconds, and containerized training produces identical test metrics to local execution.

## Cloud Deployment (GCP)

Automated scripts are provided under `scripts/` to set up cloud storage and deploy the API to Google Cloud Run.

### 1. Cloud Storage Setup
Create your GCS bucket, set public read access for reproducible training, and upload the snapshot:

```bash
# Bash (Linux/macOS)
./scripts/setup_gcs.sh <YOUR_PROJECT_ID> <YOUR_BUCKET_NAME> [REGION]

# PowerShell (Windows)
.\scripts\setup_gcs.ps1 -ProjectId "<YOUR_PROJECT_ID>" -BucketName "<YOUR_BUCKET_NAME>"
```

### 2. Deploy API to Google Cloud Run
Build the container image using Google Cloud Build and deploy to Cloud Run:

```bash
# Bash (Linux/macOS)
./scripts/deploy_cloud_run.sh <YOUR_PROJECT_ID> [SERVICE_NAME] [REGION]

# PowerShell (Windows)
.\scripts\deploy_cloud_run.ps1 -ProjectId "<YOUR_PROJECT_ID>"
```
The script outputs the live Cloud Run endpoint (e.g. `https://pokemon-fair-value-api-xxxx.a.run.app`).

### Live Deployed Endpoints
- **Streamlit Web UI**: [https://data-sourcing-ml-pipeline-cmfyvuectyqmuewhadcq6b.streamlit.app/](https://data-sourcing-ml-pipeline-cmfyvuectyqmuewhadcq6b.streamlit.app/)
- **Cloud Run API**: [https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app](https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app)
- **Health Check**: [https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app/health](https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app/health)
- **Interactive OpenAPI Docs**: [https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app/docs](https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app/docs)

## Roadmap

- [x] Data sourcing, cleaning and dated snapshots (TCGCSV)
- [x] `config.yaml`-driven, reproducible training with a train/validation/test split
- [x] MLflow experiment tracking
- [x] Semantic card search index (ChromaDB + sentence-transformers, hybrid ranking)
- [x] Two-stage hurdle model for high-value cards (evaluated; compared with histogram gradient boosting)
- [x] GCS setup script and public snapshot hosting
- [x] FastAPI `/predict` and `/search` endpoints
- [x] Dockerfile with ingest, train, serve, test and MLflow UI entry points
- [x] Google Cloud Run live deployment
- [x] Streamlit front-end application (`frontend/app.py`)
- [x] Streamlit Community Cloud public deployment ([Live App](https://data-sourcing-ml-pipeline-cmfyvuectyqmuewhadcq6b.streamlit.app/))

## Limitations and ethics

- **Unpriced factors:** the model doesn't know a Pokémon's popularity beyond name flags,
  a card's condition or physical grading (PSA/CGC), or recent tournament hype.
- **Expensive cards:** chase-card prices follow collector sentiment more than anything printed
  on the card, so the model still under-values them. The Umbreon VMAX Alt Art from Evolving
  Skies sells for $2,149, and the model estimates $233. A separate model for expensive cards
  (the two-stage hurdle) didn't fix this. The front end shows the snapshot market price next to
  the estimate for this reason.
- **Snapshot in time:** market prices move, and a model trained on one day's prices goes
  stale. Re-ingest and retrain regularly.
- **Who benefits:** a "deal finder" mostly helps resellers buy underpriced cards. That can
  push prices up for players and kids who just want the cards. We present the output as an
  estimate, not financial advice.
- **Data terms:** prices are TCGplayer's data via TCGCSV. Use is limited to coursework, and
  requests are rate-limited.

## Project layout

```
config.yaml           pipeline configuration
requirements.txt      pinned dependencies
Dockerfile            one image: ingest / train / serve / test / mlflow-ui
docker-entrypoint.sh
frontend/
  app.py              Streamlit web app: search, preview, deal calculator
  requirements.txt    Streamlit cloud dependencies
scripts/
  setup_gcs.sh / ps1        create GCS bucket & upload snapshot
  deploy_cloud_run.sh / ps1 build and deploy API to Cloud Run
api/
  main.py             FastAPI app: /health, /predict, /search
pipeline/
  ingest.py           TCGCSV -> cleaned snapshot (-> GCS)
  storage.py          GCS helpers
  data.py             load + verify snapshot, train/val/test split
  features.py         CardFeatures transformer, vintage & chase flags
  model.py            candidate models (hist_gbm, hurdle_outlier, random_forest, ridge)
  evaluate.py         metrics and plots
  train.py            training + MLflow tracking
  predict.py          fair value + verdict (used by the API)
  search.py           semantic card search (used by the API)
tests/
  test_pipeline.py
  test_search.py
  test_api.py
```

`data/`, `artifacts/` and `mlruns/` (which holds `mlflow.db`) are git-ignored and recreated by the pipeline.
