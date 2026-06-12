import os
import pandas as pd
import json
from config import PROCESSED_DIR, RESULTS_DIR
from utils import print_header, load_pickle

def main():
    print_header("Step 4: Exploratory Data Analysis")
    train_df = load_pickle(os.path.join(PROCESSED_DIR, "train.pkl"))
    
    stats = {
        "num_users": int(train_df['user_id'].nunique()),
        "num_movies": int(train_df['movie_id'].nunique()),
        "total_ratings": len(train_df),
        "mean_rating": float(train_df['rating'].mean())
    }
    
    with open(os.path.join(RESULTS_DIR, "eda_stats.json"), 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("✅ EDA Stats Generated:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
