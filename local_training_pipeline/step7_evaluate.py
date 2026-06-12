import os
import json
import numpy as np
from surprise import accuracy
from config import PROCESSED_DIR, RESULTS_DIR
from utils import print_header, load_pickle

def main():
    print_header("Step 7: Evaluate Models")
    test_df = load_pickle(os.path.join(PROCESSED_DIR, "test.pkl"))
    
    # Eval SVD
    svd = load_pickle(os.path.join(PROCESSED_DIR, "svd_model.pkl"))
    testset = list(zip(test_df['user_id'], test_df['movie_id'], test_df['rating']))
    svd_preds = svd.test(testset)
    svd_rmse = accuracy.rmse(svd_preds, verbose=False)
    
    # Eval ALS
    als = load_pickle(os.path.join(PROCESSED_DIR, "als_model.pkl"))
    u_map = load_pickle(os.path.join(PROCESSED_DIR, "u_map.pkl"))
    i_map = load_pickle(os.path.join(PROCESSED_DIR, "i_map.pkl"))
    
    _uf = als.user_factors.to_numpy() if hasattr(als.user_factors, 'to_numpy') else np.array(als.user_factors)
    _if = als.item_factors.to_numpy() if hasattr(als.item_factors, 'to_numpy') else np.array(als.item_factors)
    
    test_uids = test_df['user_id'].map(u_map)
    test_mids = test_df['movie_id'].map(i_map)
    known_mask = test_uids.notna() & test_mids.notna()
    als_est = np.full(len(test_df), 3.0)
    known_idx = np.where(known_mask)[0]
    u_indices = test_uids.iloc[known_idx].astype(int).values
    m_indices = test_mids.iloc[known_idx].astype(int).values
    
    # Vectorized computation
    als_est[known_idx] = np.clip(np.sum(_uf[u_indices] * _if[m_indices], axis=1), 1, 5)
    als_rmse = np.sqrt(np.mean((test_df['rating'].values - als_est)**2))
    
    results = {
        "SVD_RMSE": svd_rmse,
        "ALS_RMSE": als_rmse
    }
    with open(os.path.join(RESULTS_DIR, "eval_metrics.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("✅ Evaluation complete:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
