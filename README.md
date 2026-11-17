# 3D-Petrophysical-Prediction-ML

**By: Jackson R. Tomski**

This project predicts petrophysical properties (total porosity and TOC) across a 3D seismic volume using a Bayesian Neural Network trained on well logs and inverted P-impedance. Instead of a single prediction at each location, the BNN gives a mean plus P10/P90 so you can see where the model is confident and where it isn't.

It started as my UT Austin Master's thesis and AAPG Bulletin paper on the Bakken Formation. The original work used real well and seismic data that I can't share, so this repo runs on synthetic data built to look like it: a layered earth model, a P-impedance volume, horizons, and well logs, all generated from inputs you control. Because the synthetic data comes with the true porosity and TOC behind it, you can check whether the model actually recovers the truth and whether its P10/P90 bracket it.

---

## What's Inside

### Synthetic Dataset Generator

`dataPrep/syntheticData.py` builds a full dataset from an input JSON. Nothing about the survey is hard coded.

1. **Horizons** from a shared structure (dip plus smooth relief) with thickness changes between each pair. Horizons are forced to never cross.
2. **Zones** between horizons, each with its own porosity, TOC and matrix velocity ranges. Properties vary laterally and vertically using smooth random fields.
3. **Rock physics** to get impedance: TOC converts to kerogen volume, density comes from mixing matrix, kerogen and fluid, and velocity drops with porosity and kerogen. Zp = density x velocity.
4. **Inverted-volume look** by blurring the true Zp and adding noise, so it reads like an inversion result and not a perfect model.
5. **Wells** at any inline/xline. Logs are converted from time to depth with the true velocity, sampled in MD, and include high and low resolution curves, measurement noise, and -999.25 nulls. A seismic-scaled sheet averages the logs into each seismic sample.

| File | What it is |
|------|------------|
| `seismic.sgy` | P-impedance volume (SEG-Y, inline/xline header bytes set from the inputs) |
| `horizons.xlsx` | One row per trace: Inline, Xline, X, Y, and a time for each horizon |
| `well_logs.xlsx` | Per well: logs in MD and a seismic-scaled sheet in time |
| `truth.npz` | True porosity, TOC, Zp and zone cubes |
| `manifest.json` | Geometry, file names, ranges and wells, read by the data checks |

The example config (120 x 80 traces x 221 samples) is about 11 MB and builds in a few seconds. There's a size guard so you don't accidentally write a multi-GB SEG-Y.

### Data Checks

Same idea as the data check at the start of Production-Prediction-ML: catch bad inputs before anything touches a model, and report every problem at once.

- **`checkSyntheticInputs`** validates generator inputs: types, ranges, horizons shallow to deep, one more zone than horizons, wells inside the survey, file size.
- **`checkDataset`** validates any dataset through its manifest, synthetic or real: SEG-Y geometry and sample rate, Zp range, horizon rows and crossings, well sheet columns, null fractions, and MD/TWT increasing.

### Bayesian Neural Network

`models/bnn.py` and `models/mcmc.py` are the thesis BNN rebuilt as a package. It samples the posterior over network weights with MCMC using Langevin dynamics, which uses the gradient to propose better moves than a plain random walk. Predictions come back as mean, P10 and P90.

### Calculator

`Calculator.py` follows the same flow as Production-Prediction-ML:

```
input JSON → checkData → build or load dataset → checkDataset → MLflow logging → output wrapper
```

`dataset_mode` is either `synthetic` (build one from the inputs) or `existing` (point `manifest_path` at a dataset you already have).

---

## Getting Started

**You need:** Docker Desktop and VS Code with the Dev Containers extension.

```bash
git clone https://github.com/tomskija/3D-Petrophysical-Prediction-ML.git
cd 3D-Petrophysical-Prediction-ML
```

Open the folder in VS Code and select **Reopen in Container**. Docker Compose starts three services: the calculator, its FastAPI server, and an MLflow tracking server.

**Run the calculator:**
```bash
python rockPropCalculator/Calculator.py
```

**Only build a synthetic dataset:**
```bash
python -m dataPrep.syntheticData tests/in_jsons/synthetic_small.json
```

**Run the test suite:**
```bash
pytest tests -v
```

**Without VS Code:**
```bash
docker compose up -d --build
docker compose exec calculator pytest tests -v
docker compose exec calculator python rockPropCalculator/Calculator.py
```

| Service   | URL                   |
|-----------|-----------------------|
| API       | http://localhost:8001  |
| API Docs  | http://localhost:8001/docs |
| MLflow UI | http://localhost:5001 |

MLflow is on 5001 so it can run next to Production-Prediction-ML on 5000.

---

## API Endpoints

| Method | Endpoint                | Description                                        |
|--------|-------------------------|----------------------------------------------------|
| `GET`  | `/health`               | Check if the API is running                        |
| `GET`  | `/ready`                | Check the calculator and its schema load           |
| `GET`  | `/schema`               | Input form definition (fields, types, defaults, groups) used by the UI |
| `POST` | `/calculate`            | Submit a run with `{"inJson": {...}}`, returns `job_id` immediately |
| `GET`  | `/runs/{job_id}/status` | Poll run status: pending / queued / running / complete / failed |
| `GET`  | `/jobs/{job_id}/output` | Output params and tables once the run is complete |
| `GET`  | `/jobs`                 | List every job the API has seen                    |

Runs go one at a time per calculator in a background thread, and each API run writes to its own folder under `Data/Synthetic/` so runs don't overwrite each other. The full interactive docs are at `http://localhost:8001/docs`. This is the same contract as Production-Prediction-ML, which is what lets [ML-Platform-UI](https://github.com/tomskija/ML-Platform-UI) drive every calculator the same way.

---

## Configuration

Inputs use the same `params` / `tables` JSON format as Production-Prediction-ML. Anything you leave out falls back to a generic default. The main ones:

| Parameter | What it does |
|-----------|--------------|
| `dataset_mode` | `synthetic` or `existing` |
| `manifest_path` | Dataset to load when `dataset_mode` is `existing` |
| `n_inlines`, `n_xlines` | Survey size |
| `t_start_ms`, `t_end_ms`, `dt_ms` | Time window and sample rate |
| `horizon_names`, `horizon_mean_time_ms` | Horizons, shallow to deep |
| `zone_phi_mean`, `zone_toc_mean`, `zone_vp_matrix_fts` | Rock properties per zone (one more value than horizons) |
| `well_names`, `well_inlines`, `well_xlines` | Where the wells go |
| `target_horizons` | Top and base of the logged interval |
| `zp_min`, `zp_max` | Impedance range, used for clipping and checks |
| `random_seed` | Same seed, same dataset |

See `tests/in_jsons/testCase01.json` for a full example. Generated data goes to `Data/Synthetic/`, which is gitignored.

---

## Project Structure

```
3D-Petrophysical-Prediction-ML/
├── .devcontainer/devcontainer.json     # Dev container config
├── .github/workflows/ci.yml            # Builds the Docker stack, runs pytest and the calculator
├── rockPropCalculator/
│   ├── Calculator.py                   # Orchestrator
│   ├── dataPrep/
│   │   ├── syntheticData.py            # Synthetic seismic, horizons and well logs
│   │   ├── schema.py                   # Input form definition served at /schema
│   │   ├── dataChecks.py               # Input and dataset checks
│   │   ├── inputWrapper.py             # JSON input parser
│   │   ├── outputWrapper.py            # Output wrapper
│   │   └── mlFlowConfig.py             # MLflow (Docker server or local SQLite)
│   ├── models/
│   │   ├── bnn.py                      # Bayesian Neural Network
│   │   ├── mcmc.py                     # Langevin dynamics MCMC sampler
│   │   └── config.py                   # BNN and MCMC settings
│   ├── DataFiles/                      # Well, horizon and seismic loaders
│   ├── utils/                          # Metrics and plotting
│   └── main.py                         # Original thesis workflow
├── tests/
│   ├── in_jsons/                       # pytest input configs
│   ├── test_synthetic_data.py
│   ├── test_api.py
│   ├── test_calculator.py
│   └── test_bnn.py
├── api.py                              # FastAPI, same calculator contract as PPML
├── conftest.py                         # pytest path config
├── Dockerfile                          # Python 3.11
├── Dockerfile.mlflow                   # MLflow server
├── docker-compose.yml                  # calculator, api, mlflow
└── requirements.txt
```

---

## Citation

This repo builds on the following work:

**Journal Publication**
```
Tomski, J.R., Sen, M.K., Hess, T.E., Pyrcz, M.J. (2022). Unconventional reservoir
characterization by seismic inversion and machine learning of the Bakken Formation.
AAPG Bulletin, 106(11): 2203-2223. https://doi.org/10.1306/12162121035
```

**Master's Thesis**
```
Tomski, J.R. (2020). Unconventional reservoir parameter estimation by seismic inversion
and machine learning of the Bakken Formation, North Dakota.
Master's Thesis, University of Texas at Austin.
```

---

## Data

The original thesis used proprietary well and seismic data that is not included and never will be. Everything in this repo runs on synthetic data from `dataPrep/syntheticData.py`.

---

## Dependencies

Python 3.11 · numpy · pandas · scipy · scikit-learn · segyio · matplotlib · seaborn · openpyxl · mlflow · pytest · fastapi · uvicorn · pydantic
