import os
import json
import random
from datasets import load_dataset

def generate_benchmark_dataset(full: bool = False, num_samples: int = 20):
    print("Downloading ChartQA dataset from Hugging Face...")
    
    # Use the validation split because it's meant for testing and is smaller to download
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="val")
    
    # Create the directory for the images if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "test_images")
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    
    if full:
        sampled_dataset = dataset.shuffle(seed=41)
    else:
        print(f"Randomly sampling {num_samples} charts...")
        # Shuffle the dataset and select the requested number of samples
        sampled_dataset = dataset.shuffle(seed=41).select(range(num_samples))
    
    benchmark_data = []
    
    for i, item in enumerate(sampled_dataset):
        # 1. Extract the image and save it locally
        image = item["image"]
        image_filename = f"chartqa_sample_{i}.png"
        image_filepath = os.path.join(output_dir, image_filename)
        
        # Convert to RGB to avoid issues with saving weird formats, then save
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_filepath)
        
        # 2. Extract the query and the answer
        query = item["query"]
        
        # ChartQA stores answers in a list under the 'label' key (e.g., ["49"])
        # Grab the first item in the list
        expected_answer = item["label"][0] if isinstance(item["label"], list) else item["label"]
        
        # 3. Format it for our benchmark script
        benchmark_data.append({
            # Save the relative path from the root directory so main.py can find it
            "image_path": f"evals/test_images/{image_filename}",
            "query": query,
            "expected_answer": expected_answer
        })
        
        print(f"Saved Sample {i+1}: {query[:50]}... -> {expected_answer}")

    # 4. Save the JSON ledger
    with open(json_path, 'w') as f:
        json.dump(benchmark_data, f, indent=4)
        
    print("\nDataset generation complete!")
    print(f"   Images saved to: {output_dir}")
    print(f"   JSON saved to:   {json_path}")

if __name__ == "__main__":
    generate_benchmark_dataset(True, 0)