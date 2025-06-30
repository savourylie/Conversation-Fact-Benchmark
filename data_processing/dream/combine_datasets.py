import json
import os
from pathlib import Path
from typing import List, Dict, Any


def load_json_file(file_path: str) -> List[Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        List[Any]: The loaded JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} items from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        raise


def add_split_info(data: List[Any], split_name: str) -> List[Dict[str, Any]]:
    """
    Add split information to each data item.
    
    Args:
        data (List[Any]): The original data
        split_name (str): Name of the data split (train, dev, test)
        
    Returns:
        List[Dict[str, Any]]: Data with added split information
    """
    enhanced_data = []
    for item in data:
        enhanced_item = {
            "conversation": item[0],
            "questions": item[1], 
            "id": item[2],
            "split": split_name
        }
        enhanced_data.append(enhanced_item)
    
    return enhanced_data


def combine_dream_datasets(
    train_path: str = "datasets/dream/train.json",
    dev_path: str = "datasets/dream/dev.json", 
    test_path: str = "datasets/dream/test.json",
    output_path: str = "datasets/dream/full.json"
) -> Dict[str, int]:
    """
    Combine the three DREAM dataset JSON files into one full.json file.
    
    Args:
        train_path (str): Path to train.json
        dev_path (str): Path to dev.json  
        test_path (str): Path to test.json
        output_path (str): Path for the combined full.json output
        
    Returns:
        Dict[str, int]: Statistics about the combination process
    """
    print("Starting to combine DREAM datasets...")
    
    # Load all three datasets
    try:
        train_data = load_json_file(train_path)
        dev_data = load_json_file(dev_path)
        test_data = load_json_file(test_path)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Failed to load one or more dataset files")
        return {}
    
    # Add split information to each dataset
    train_enhanced = add_split_info(train_data, "train")
    dev_enhanced = add_split_info(dev_data, "dev") 
    test_enhanced = add_split_info(test_data, "test")
    
    # Combine all datasets
    combined_data = train_enhanced + dev_enhanced + test_enhanced
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save combined dataset
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved combined dataset to {output_path}")
        
        # Return statistics
        stats = {
            "train_samples": len(train_data),
            "dev_samples": len(dev_data),
            "test_samples": len(test_data), 
            "total_samples": len(combined_data)
        }
        
        print(f"Dataset combination statistics:")
        print(f"  Train samples: {stats['train_samples']}")
        print(f"  Dev samples: {stats['dev_samples']}")
        print(f"  Test samples: {stats['test_samples']}")
        print(f"  Total samples: {stats['total_samples']}")
        
        return stats
        
    except Exception as e:
        print(f"Error saving combined dataset: {e}")
        return {}


def verify_combined_dataset(file_path: str = "datasets/dream/full.json") -> bool:
    """
    Verify the structure and content of the combined dataset.
    
    Args:
        file_path (str): Path to the combined dataset file
        
    Returns:
        bool: True if verification passes, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Verifying combined dataset with {len(data)} items...")
        
        # Check if data is not empty
        if not data:
            print("Error: Combined dataset is empty")
            return False
        
        # Check structure of first few items
        required_keys = {"conversation", "questions", "id", "split"}
        
        for i, item in enumerate(data[:5]):  # Check first 5 items
            if not isinstance(item, dict):
                print(f"Error: Item {i} is not a dictionary")
                return False
                
            if not required_keys.issubset(item.keys()):
                print(f"Error: Item {i} missing required keys. Has: {set(item.keys())}, Required: {required_keys}")
                return False
        
        # Check split distribution
        split_counts = {}
        for item in data:
            split = item.get("split", "unknown")
            split_counts[split] = split_counts.get(split, 0) + 1
        
        print("Split distribution:")
        for split, count in split_counts.items():
            print(f"  {split}: {count}")
        
        print("Verification passed!")
        return True
        
    except Exception as e:
        print(f"Error during verification: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("Combining DREAM datasets...")
    stats = combine_dream_datasets()
    
    if stats:
        print("\nVerifying combined dataset...")
        verify_combined_dataset()
    else:
        print("Failed to combine datasets")
