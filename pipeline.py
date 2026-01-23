import subprocess
import sys
import os

# CONFIGURATION
# Set to True if you want the PC to shut down after finishing (Windows only)
SHUTDOWN_ON_FINISH = False 

def run_command(command, step_name):
    print(f"\n{'='*60}")
    print(f"[PIPELINE] Starting Step: {step_name}")
    print(f"{'='*60}")
    
    try:
        # check=True stops the script immediately if an error occurs
        subprocess.run(command, check=True, shell=True)
        print(f"\n[SUCCESS] {step_name} completed successfully. \u2705")
    except subprocess.CalledProcessError:
        print(f"\n[FAILURE] {step_name} crashed! Pipeline stopped. \u274C")
        print("Check the error message above.")
        sys.exit(1) # Kills the script so nothing else runs

if __name__ == "__main__":
    # Get the current python interpreter (forces the 'holognn' environment)
    python_exe = sys.executable

    # --- STEP 1: TRAINING ---
    # This will take hours. If it crashes, the script dies here.
    run_command(f'"{python_exe}" train.py', "Phase 1: Model Training")

    # --- STEP 2: VERIFICATION ---
    # This runs only if Step 1 succeeds.
    run_command(f'"{python_exe}" predict.py', "Phase 2: Inference Test")

    print("\n" + "="*60)
    print("ALL TASKS COMPLETED. RESULTS READY.")
    print("="*60)

    # Optional: Shutdown computer to save power
    if SHUTDOWN_ON_FINISH:
        print("Shutting down in 60 seconds... (Ctrl+C to cancel)")
        os.system("shutdown /s /t 60")