#!/usr/bin/env python3

"""
Script to download and transform the MSC (Multi-Session Chat) dataset from HuggingFace.
"""

import os
from pathlib import Path
from data_processing.msc.transform import transform_msc_dataset


def setup_msc_dataset():
    """Download and transform the MSC dataset from HuggingFace"""
    
    # Create datasets directory if it doesn't exist
    datasets_dir = Path("datasets")
    msc_dir = datasets_dir / "msc"
    processed_dir = msc_dir / "processed"
    
    # Create directories
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Setting up MSC dataset...")
    print(f"Output directory: {processed_dir}")
    
    # Download and transform the dataset
    try:
        output_file = processed_dir / "msc_memfuse_mc10_transformed.json"
        
        dataset = transform_msc_dataset(
            input_source="Percena/msc-memfuse-mc10",
            output_file=output_file,
            source_type="huggingface"
        )
        
        print(f"\nMSC dataset successfully set up!")
        print(f"Dataset saved to: {output_file}")
        print(f"Total entries: {len(dataset.entries)}")
        print(f"Average choices per question: {dataset.get_average_choices_per_question():.1f}")
        print(f"Average sessions per entry: {dataset.get_average_sessions_per_entry():.1f}")
        print(f"Average messages per entry: {dataset.get_average_messages_per_entry():.1f}")
        
        print(f"\nYou can now run evaluations with:")
        print(f"python main.py -d {output_file} -p ollama -m llama3.2:latest -n 10")
        
    except Exception as e:
        print(f"Error setting up MSC dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    setup_msc_dataset()
