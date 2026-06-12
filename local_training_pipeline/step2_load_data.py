import os
import pandas as pd
import numpy as np
import zipfile
from config import DATA_DIR, PROCESSED_DIR
from utils import print_header, save_pickle

def main():
    print_header("Step 2: Loading Raw Data")
    
    # Check for txt or txt.zip
    files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('combined_data') and (f.endswith('.txt') or f.endswith('.txt.zip'))])
    if not files:
        print(f"❌ No combined_data files found in {DATA_DIR}.")
        return False
        
    tmp_csv = os.path.join(PROCESSED_DIR, "tmp_ratings.csv")
    if not os.path.exists(tmp_csv):
        print(f"Parsing raw text into CSV from {files[0]}...")
        f_path = os.path.join(DATA_DIR, files[0])
        
        try:
            with open(tmp_csv, 'w') as f_out:
                f_out.write("user_id,movie_id,rating,date\n")
                movie_id = -1
                
                # Handle both zipped and unzipped variants
                if f_path.endswith('.zip'):
                    with zipfile.ZipFile(f_path, 'r') as z:
                        inner_name = files[0].replace('.zip', '')
                        with z.open(inner_name, 'r') as f_in:
                            for line in f_in:
                                line = line.decode('utf-8').strip()
                                if line.endswith(':'):
                                    movie_id = line[:-1]
                                else:
                                    p = line.split(',')
                                    f_out.write(f"{p[0]},{movie_id},{p[1]},{p[2]}\n")
                else:
                    with open(f_path, 'r') as f_in:
                        for line in f_in:
                            line = line.strip()
                            if line.endswith(':'):
                                movie_id = line[:-1]
                            else:
                                p = line.split(',')
                                f_out.write(f"{p[0]},{movie_id},{p[1]},{p[2]}\n")
        except Exception as e:
            print(f"❌ Error during parsing: {e}")
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)
            return False
            
    print("Loading CSV into DataFrame...")
    df = pd.read_csv(tmp_csv, dtype={'user_id': np.int32, 'movie_id': np.int16, 'rating': np.float32})
    save_pickle(df, os.path.join(PROCESSED_DIR, "raw_ratings.pkl"))
    print("✅ Raw ratings loaded and saved to pickle.")
    return True

if __name__ == "__main__":
    main()
