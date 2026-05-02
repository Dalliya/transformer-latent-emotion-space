import subprocess
import sys
import time

def run_step(script_path: str, step_name: str):
    """
    Runs a python script as a separate sub-process.
    This ensures complete memory cleanup after each heavy ML step,
    which is critical for systems with limited RAM (e.g., 8GB).
    """
    print(f"\n{'='*50}")
    print(f"🚀 STARTING STEP: {step_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # Run the script and stream the output to the console
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: Step '{step_name}' failed. Pipeline aborted.")
        sys.exit(1)
        
    execution_time = time.time() - start_time
    print(f"\n✅ COMPLETED: {step_name} in {execution_time:.2f} seconds\n")

if __name__ == "__main__":
    print("🧠 INITIALIZING EMOTION CLASSIFICATION PIPELINE 🧠")
    
    # Step 1: Data Ingestion & Subsampling
    run_step("src/data_loader.py", "Data Loading & Stratification")
    
    # Step 2: BERT Inference (Heavy RAM/GPU usage)
    run_step("src/inference.py", "Model Inference (BERT on MPS)")
    
    # Step 3: Domain Transfer & Profiling (28 -> 2 mapping)
    run_step("src/mapping.py", "Zero-Shot Mapping & Performance Profiling")
    
    # Step 4: Dimensionality Reduction & Visualization
    run_step("src/visualization.py", "UMAP Latent Space Visualization")
    
    print("\n🎉 PIPELINE EXECUTED SUCCESSFULLY. Check 'data/processed/' for outputs.")