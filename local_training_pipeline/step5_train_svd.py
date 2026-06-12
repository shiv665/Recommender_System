import os
from surprise import SVD, Dataset, Reader, accuracy
from config import PROCESSED_DIR, SEED
from utils import print_header, load_pickle, save_pickle

def main():
    print_header("Step 5: Train SVD Model")
    train_df = load_pickle(os.path.join(PROCESSED_DIR, "train.pkl"))
    
    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(train_df[['user_id','movie_id','rating']], reader).build_full_trainset()
    
    svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=SEED)
    svd.fit(trainset)
    
    save_pickle(svd, os.path.join(PROCESSED_DIR, "svd_model.pkl"))
    print("✅ SVD Model trained and saved.")

if __name__ == "__main__":
    main()
