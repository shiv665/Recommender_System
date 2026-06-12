import os
import pandas as pd
import numpy as np
from config import PROCESSED_DIR, SEED
from utils import print_header, load_pickle, save_pickle

def main():
    print_header("Step 3: Preprocessing & Splitting Data")
    np.random.seed(SEED)
    df = load_pickle(os.path.join(PROCESSED_DIR, "raw_ratings.pkl"))
    
    uc = df['user_id'].value_counts()
    active = uc[uc >= 20].index
    sampled = np.random.choice(active, size=int(len(active)*0.5), replace=False) # Use 50% for speed
    df = df[df['user_id'].isin(set(sampled))].copy().reset_index(drop=True)
    
    mask = np.random.rand(len(df)) < 0.8
    train_df = df[mask].copy().reset_index(drop=True)
    test_df = df[~mask].copy().reset_index(drop=True)
    
    save_pickle(train_df, os.path.join(PROCESSED_DIR, "train.pkl"))
    save_pickle(test_df, os.path.join(PROCESSED_DIR, "test.pkl"))
    print(f"✅ Data split: {len(train_df)} train, {len(test_df)} test.")

if __name__ == "__main__":
    main()
