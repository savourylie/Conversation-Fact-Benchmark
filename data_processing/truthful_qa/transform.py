import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from data_models.truthful_qa import (
    TruthfulQAEntry,
    TruthfulQADataset
)


def extract_choices_and_correct_index(targets_dict: Dict[str, int]) -> Tuple[List[str], int]:
    """
    Extract choices and correct choice index from the original targets format.
    
    Args:
        targets_dict: Dictionary with choice text as keys and correctness (0 or 1) as values
    
    Returns:
        Tuple[List[str], int]: List of choices and the index of the correct choice
    """
    choices = list(targets_dict.keys())
    correct_choice_index = -1
    
    # Find the correct choice
    for i, (choice_text, is_correct) in enumerate(targets_dict.items()):
        if is_correct == 1:
            if correct_choice_index == -1:
                correct_choice_index = i
            else:
                # Multiple correct answers found - this shouldn't happen in MC1
                raise ValueError(f"Multiple correct answers found in MC1 targets: {targets_dict}")
    
    if correct_choice_index == -1:
        raise ValueError(f"No correct answer found in targets: {targets_dict}")
    
    return choices, correct_choice_index


def transform_entry(entry_data: Dict[str, Any]) -> TruthfulQAEntry:
    """
    Transform a single TruthfulQA entry from the original format to the new format.
    
    Args:
        entry_data: Dictionary containing the original entry data
    
    Returns:
        TruthfulQAEntry: Transformed entry
    """
    question = entry_data["question"]
    
    # Extract choices and correct index from MC1 targets only
    choices, correct_choice_index = extract_choices_and_correct_index(entry_data["mc1_targets"])
    
    return TruthfulQAEntry(
        question=question,
        choices=choices,
        correct_choice_index=correct_choice_index
    )


def validate_entry(entry: TruthfulQAEntry) -> List[str]:
    """
    Validate a TruthfulQA entry and return any validation errors.
    
    Args:
        entry: The TruthfulQA entry to validate
    
    Returns:
        List[str]: List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check that there are choices
    if len(entry.choices) == 0:
        errors.append("Entry should have at least 1 choice")
    
    # Check that correct_choice_index is valid
    if not (0 <= entry.correct_choice_index < len(entry.choices)):
        errors.append(f"Invalid correct_choice_index: {entry.correct_choice_index}, should be between 0 and {len(entry.choices) - 1}")
    
    # Check for duplicate choices
    if len(entry.choices) != len(set(entry.choices)):
        errors.append("Duplicate choices found")
    
    return errors


def load_and_transform_truthful_qa_dataset(input_file: Path) -> TruthfulQADataset:
    """
    Load the original TruthfulQA dataset JSON file and transform it to the new format.
    
    Args:
        input_file: Path to the input JSON file
    
    Returns:
        TruthfulQADataset: Transformed dataset
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    entries = []
    failed_entries = []
    validation_warnings = []
    
    for i, entry_data in enumerate(original_data):
        try:
            transformed_entry = transform_entry(entry_data)
            
            # Validate the transformed entry
            validation_errors = validate_entry(transformed_entry)
            if validation_errors:
                validation_warnings.append({
                    'entry_index': i,
                    'question': entry_data.get('question', 'Unknown question')[:100] + '...',
                    'validation_errors': validation_errors
                })
                print(f"Warning: Validation issues for entry {i}: {', '.join(validation_errors)}")
            
            entries.append(transformed_entry)
        except Exception as e:
            failed_entries.append({
                'index': i,
                'question': entry_data.get('question', 'Unknown question')[:100] + '...',
                'error': str(e)
            })
            print(f"Error: Failed to transform entry {i}: {e}")
    
    # Calculate statistics
    total_choices = sum(len(entry.choices) for entry in entries)
    avg_choices = total_choices / len(entries) if entries else 0
    
    # Analyze choice distribution
    choice_counts = {}
    for entry in entries:
        num_choices = len(entry.choices)
        choice_counts[num_choices] = choice_counts.get(num_choices, 0) + 1
    
    metadata = {
        'source_file': str(input_file),
        'total_entries': len(original_data),
        'successfully_transformed': len(entries),
        'failed_entries': len(failed_entries),
        'validation_warnings': len(validation_warnings),
        'average_choices_per_question': round(avg_choices, 2),
        'choice_distribution': choice_counts,
        'failed_entry_details': failed_entries,
        'validation_warning_details': validation_warnings
    }
    
    return TruthfulQADataset(entries=entries, metadata=metadata)


def save_transformed_dataset_jsonl(dataset: TruthfulQADataset, output_file: Path) -> None:
    """
    Save the transformed dataset to a JSONL file (one entry per line).
    
    Args:
        dataset: The transformed TruthfulQA dataset
        output_file: Path where to save the transformed dataset
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset.entries:
            json.dump(entry.model_dump(), f, ensure_ascii=False)
            f.write('\n')


def load_transformed_dataset_jsonl(input_file: Path) -> TruthfulQADataset:
    """
    Load a transformed TruthfulQA dataset from a JSONL file.
    
    Args:
        input_file: Path to the input JSONL file
    
    Returns:
        TruthfulQADataset: Loaded dataset
    """
    entries = []
    failed_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry_data = json.loads(line)
                entry = TruthfulQAEntry(**entry_data)
                entries.append(entry)
            except Exception as e:
                failed_entries.append({
                    'line_number': line_num,
                    'error': str(e)
                })
                print(f"Warning: Failed to parse line {line_num}: {e}")
    
    # Calculate statistics for loaded data
    total_choices = sum(len(entry.choices) for entry in entries)
    avg_choices = total_choices / len(entries) if entries else 0
    
    choice_counts = {}
    for entry in entries:
        num_choices = len(entry.choices)
        choice_counts[num_choices] = choice_counts.get(num_choices, 0) + 1
    
    metadata = {
        'source_file': str(input_file),
        'format': 'jsonl',
        'total_entries': len(entries),
        'failed_entries': len(failed_entries),
        'average_choices_per_question': round(avg_choices, 2),
        'choice_distribution': choice_counts,
        'failed_entry_details': failed_entries
    }
    
    return TruthfulQADataset(entries=entries, metadata=metadata)


def transform_truthful_qa_dataset(
    input_file: Path = None,
    output_file: Path = None
) -> TruthfulQADataset:
    """
    Main function to transform the TruthfulQA dataset to JSONL format.
    
    Args:
        input_file: Path to the input JSON file (defaults to datasets/truthful_qa/raw/mc_task.json)
        output_file: Path to save the transformed dataset (optional)
    
    Returns:
        TruthfulQADataset: The transformed dataset
    """
    if input_file is None:
        input_file = Path("datasets/truthful_qa/raw/mc_task.json")
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading and transforming TruthfulQA dataset from: {input_file}")
    dataset = load_and_transform_truthful_qa_dataset(input_file)
    
    print(f"Transformation complete:")
    print(f"  - Total entries: {dataset.metadata['total_entries']}")
    print(f"  - Successfully transformed: {dataset.metadata['successfully_transformed']}")
    print(f"  - Failed entries: {dataset.metadata['failed_entries']}")
    print(f"  - Validation warnings: {dataset.metadata['validation_warnings']}")
    print(f"  - Average choices per question: {dataset.metadata['average_choices_per_question']}")
    
    # Show choice distribution
    choice_dist = dataset.metadata['choice_distribution']
    print(f"  - Choice distribution: {dict(sorted(choice_dist.items()))}")
    
    if output_file:
        print(f"Saving transformed dataset to: {output_file} (JSONL format)")
        save_transformed_dataset_jsonl(dataset, output_file)
        print("Save complete!")
    
    return dataset


if __name__ == "__main__":
    # Transform to JSONL format (but with .json extension for HuggingFace compatibility)
    transformed_dataset = transform_truthful_qa_dataset(
        output_file=Path("datasets/truthful_qa/processed/mc_task_transformed.json")
    )
