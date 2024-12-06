#Remove the comments of these imports to use the sheduler
#import schedule 
#import time
from fine_tuning_system import LLMFinetuner

def periodic_fine_tuning():
    """
    Run fine-tuning process periodically
    """
    try:
        fine_tuner = LLMFinetuner()
        fine_tuner.run_fine_tuning_pipeline(
            days=7,   # Look back 7 days
            epochs=1,  # 2 training epochs
            batch_size=1  # Batch size for training
        )
    except Exception as e:
        print(f"Fine-tuning failed: {e}")

def main():
    # Run fine-tuning immediately on startup
    periodic_fine_tuning()
    
    # Schedule periodic fine-tuning(Turned of)
    # Run every week
    #schedule.every(7).days.do(periodic_fine_tuning)
    
    # Keep the script running (Turned off)
    #while True:
    #    schedule.run_pending()
    #    time.sleep(1)

if __name__ == "__main__":
    main()