import os
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from config import PROCESSED_DIR, SEED
from utils import print_header, load_pickle, save_pickle

def main():
    print_header("Step 6: Train ALS Model (CUDA)")
    train_df = load_pickle(os.path.join(PROCESSED_DIR, "train.pkl"))
    
    users_arr = train_df['user_id'].unique()
    items_arr = train_df['movie_id'].unique()
    u_map = {u: i for i, u in enumerate(users_arr)}
    i_map = {m: i for i, m in enumerate(items_arr)}
    
    save_pickle(u_map, os.path.join(PROCESSED_DIR, "u_map.pkl"))
    save_pickle(i_map, os.path.join(PROCESSED_DIR, "i_map.pkl"))
    
    rows = train_df['user_id'].map(u_map).values
    cols = train_df['movie_id'].map(i_map).values
    vals = train_df['rating'].values.astype(np.float32)
    sparse_mat = csr_matrix((vals, (rows, cols)), shape=(len(users_arr), len(items_arr)))
    
    print("Training ALS...")
    try:
        als = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15, random_state=SEED, use_gpu=True)
        als.fit(sparse_mat)
    except Exception:
        als = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15, random_state=SEED, use_gpu=False)
        als.fit(sparse_mat)
        
    save_pickle(als, os.path.join(PROCESSED_DIR, "als_model.pkl"))
    print("✅ ALS Model trained and saved.")

if __name__ == "__main__":
    main()
