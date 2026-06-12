from utils import print_header
import step1_check_data
import step2_load_data
import step3_preprocess
import step4_eda
import step5_train_svd
import step6_train_als
import step7_evaluate
import step8_generate_recs

def run_all():
    print_header("STARTING END-TO-END PIPELINE")
    
    if not step1_check_data.main():
        print("Pipeline aborted at Step 1.")
        return
    if not step2_load_data.main():
        print("Pipeline aborted at Step 2.")
        return
        
    step3_preprocess.main()
    step4_eda.main()
    step5_train_svd.main()
    step6_train_als.main()
    step7_evaluate.main()
    step8_generate_recs.main()
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all()
