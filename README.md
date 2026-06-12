#  Netflix Recommendation System (CULT Open Projects 2026)


A production-ready, highly optimized recommendation engine built for the **Netflix Prize Dataset**. This project features a dual-algorithm approach (Singular Value Decomposition & Alternating Least Squares) accelerated by CUDA, alongside a fully deployable FastAPI prediction service.

Developed by **Shivansh (2nd Year Undergrad in ECE at IIT Roorkee, India)** for the Cult Open Projects 2026 competition.

##  Key Features

* **Dual Algorithm Engine**: Combines standard Collaborative Filtering (Surprise SVD) and Deep Learning / Matrix Factorization (Implicit ALS).
* **Hardware Accelerated**: ALS training is optimized specifically for NVIDIA RTX 2050 (CUDA) for lightning-fast matrix decomposition on millions of data points.
* **Modular Pipeline**: Clean, 11-step local training pipeline covering everything from dataset extraction and EDA to model evaluation.
* **Production API**: Includes a lightweight, containerized FastAPI backend that serves `O(1)` precomputed predictions instantly.

##  Repository Structure

```text
 netflix-recsys
 ┣  DATASET/                    # Raw Netflix Prize txt/csv datasets
 ┣  kaggle/                     # Original Kaggle notebooks & experimental results
 ┣  local_training_pipeline/    # 11-step modular ML pipeline
 ┃ ┣  config.py                 # Hyperparameters & Paths
 ┃ ┣  step4_eda.py              # Exploratory Data Analysis
 ┃ ┣  step6_train_als.py        # GPU-accelerated ALS training
 ┃ ┗  run_pipeline.py           # Master execution script
 ┣  prediction_api/             # Production deployment folder
 ┃ ┣  data/                     # Pre-computed recommendations.json
 ┃ ┣  main.py                   # FastAPI service (CORS enabled)
 ┃ ┗  Dockerfile                # Deployment container blueprint
 ┣  README.md
 ┗  requirements.txt            # Global dependencies
```

##  Quick Start

### 1. Training the Models Locally
To run the entire machine learning pipeline from scratch (EDA -> SVD -> ALS -> Evaluation):

```bash
cd local_training_pipeline
python run_pipeline.py
```
*(Note: Ensure your `DATASET/` folder contains the Netflix Prize `.zip` files).*

### 2. Starting the API Server
The API serves pre-calculated `recommendations.json` data for blazing-fast `O(1)` response times.

```bash
cd prediction_api
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` to interact with the Swagger UI and test predictions!


##  Methodology & Metrics
* **Data Processing**: Filtered out inactive users (rating < 20 movies) to improve matrix density.
* **Surprise SVD**: 50 factors, 20 epochs. Excellent at capturing global user-movie biases.
* **Implicit ALS**: 50 factors, 15 iterations. Accelerated via CUDA (`use_gpu=True`). Excels at handling highly sparse implicit-feedback matrices.

---
*Built for Cult AI/ML 2026*
