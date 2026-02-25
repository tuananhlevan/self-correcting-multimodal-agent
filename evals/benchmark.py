import json
import time
import torch
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Add root to path so we can import our graph
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.graph import build_graph

def run_benchmark(dataset_path: str):
    print("Initializing Benchmark Suite...")
    
    # Load the evaluation dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        
    app = build_graph()
    
    results = {
        "total_tested": len(dataset),
        "baseline_correct": 0,       # Got it right on the very first try (0 loops)
        "self_corrected_correct": 0, # Got it right ONLY after looping and fixing an error
        "failed": 0,                 # Never got it right, even after max retries
        "total_loops_triggered": 0,
        "average_time_seconds": 0.0
    }

    total_time = 0

    print(f"Running evaluation on {len(dataset)} charts...\n")

    for i, item in enumerate(dataset):
        print(f"--- Evaluating Item {i+1}/{len(dataset)}: {item['image_path']} ---")
        
        initial_state = {
            "image_path": item["image_path"],
            "user_query": item["query"],
            "extracted_data": {},
            "generated_code": "",
            "execution_result": "",
            "error_history": [],
            "loop_count": 0,
            "final_answer": ""
        }

        # Reset PyTorch memory stats for this specific run
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        
        # Run the agent synchronously for the benchmark
        final_state = app.invoke(initial_state)
        
        execution_time = time.time() - start_time
        total_time += execution_time
        
        # Hardware Profiling
        peak_vram_gb = 0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        actual_answer = final_state.get("final_answer", "").strip()
        expected = item["expected_answer"].strip()
        loops = final_state.get("loop_count", 0)
        
        results["total_loops_triggered"] += (loops - 1) if loops > 0 else 0

        # Evaluation Logic
        # (For a real project, you might use an LLM-as-a-judge here instead of strict string matching)
        is_correct = expected in actual_answer
        
        if is_correct and loops <= 1:
            print(f"✅ PASS (Baseline): Answered correctly on first attempt.")
            results["baseline_correct"] += 1
        elif is_correct and loops > 1:
            print(f"🔄 PASS (Self-Corrected): Answered correctly after {loops - 1} correction loops.")
            results["self_corrected_correct"] += 1
        else:
            print(f"❌ FAIL: Expected '{expected}', got '{actual_answer}'.")
            results["failed"] += 1
            
        print(f"Time: {execution_time:.2f}s | Peak VRAM: {peak_vram_gb:.2f} GB\n")

    # --- Final Report Generation ---
    results["average_time_seconds"] = total_time / len(dataset)
    baseline_accuracy = (results["baseline_correct"] / len(dataset)) * 100
    final_accuracy = ((results["baseline_correct"] + results["self_corrected_correct"]) / len(dataset)) * 100
    improvement = final_accuracy - baseline_accuracy

    print("=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Baseline Accuracy (No Loops):    {baseline_accuracy:.1f}%")
    print(f"Final Accuracy (Self-Correcting): {final_accuracy:.1f}%")
    print(f"Net Accuracy Improvement:        +{improvement:.1f}%")
    print(f"Total Correction Loops Fired:    {results['total_loops_triggered']}")
    print(f"Average Inference Time:          {results['average_time_seconds']:.2f}s")
    print("=" * 50)

if __name__ == "__main__":
    # Point this to your JSON dataset
    run_benchmark("evals/test_dataset.json")