import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'DATASET')
PROCESSED_DIR = os.path.join(BASE_DIR, 'local_training_pipeline', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'local_training_pipeline', 'results')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
