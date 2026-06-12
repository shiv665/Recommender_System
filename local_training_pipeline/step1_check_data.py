import os
from config import DATA_DIR
from utils import print_header

def main():
    print_header("Step 1: Check Data Dependencies")
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data folder missing at {DATA_DIR}")
        return False
    print("✅ Data directory verified.")
    return True

if __name__ == "__main__":
    main()
