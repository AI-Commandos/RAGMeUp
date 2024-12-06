import subprocess
import sys
import argparse

def run_server():
    subprocess.run([sys.executable, "server.py"])

def run_fine_tuning_scheduler():
    subprocess.run([sys.executable, "fine_tuning_scheduler.py"])

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Start server or fine-tuning scheduler")
    parser.add_argument("service", choices=["server", "fine-tuning"], 
                        help="Choose which service to run")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the selected service
    if args.service == "server":
        print("Starting server...")
        run_server()
    elif args.service == "fine-tuning":
        print("Starting fine-tuning scheduler...")
        run_fine_tuning_scheduler()

if __name__ == "__main__":
    main()