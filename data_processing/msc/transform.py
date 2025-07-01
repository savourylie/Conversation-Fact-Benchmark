import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datasets import load_dataset

from data_models.msc import (
    MSCEntry,
    MSCDataset,
    ConversationMessage,
    ConversationSession
)


def validate_entry(entry: MSCEntry) -> List[str]:
    """
    Validate an MSC entry and return any validation errors.
    
    Args:
        entry: The MSC entry to validate
    
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
    
    # Check that there are conversation sessions
    if len(entry.haystack_sessions) == 0:
        errors.append("Entry should have at least 1 conversation session")
    
    # Check that session IDs match session count
    if len(entry.haystack_session_ids) != len(entry.haystack_sessions):
        errors.append(f"Mismatch between session IDs ({len(entry.haystack_session_ids)}) and sessions ({len(entry.haystack_sessions)})")
    
    # Check conversation format
    for i, session in enumerate(entry.haystack_sessions):
        if not isinstance(session, list):
            errors.append(f"Session {i} should be a list of messages")
            continue
        
        for j, message in enumerate(session):
            if not isinstance(message, dict):
                errors.append(f"Session {i}, message {j} should be a dictionary")
                continue
            
            if 'role' not in message or 'content' not in message:
                errors.append(f"Session {i}, message {j} missing 'role' or 'content' field")
                continue
            
            if message['role'] not in ['user', 'assistant']:
                errors.append(f"Session {i}, message {j} has invalid role: {message['role']}")
    
    return errors


def transform_entry_from_huggingface(entry_data: Dict[str, Any]) -> MSCEntry:
    """
    Transform a single MSC entry from HuggingFace format to our format.
    
    Args:
        entry_data: Dictionary containing the original entry data from HuggingFace
    
    Returns:
        MSCEntry: Transformed entry
    """
    return MSCEntry(
        question_id=entry_data["question_id"],
        question=entry_data["question"],
        answer=entry_data["answer"],
        choices=entry_data["choices"],
        correct_choice_index=entry_data["correct_choice_index"],
        haystack_session_ids=entry_data["haystack_session_ids"],
        haystack_sessions=entry_data["haystack_sessions"]
    )


def load_and_transform_msc_dataset_from_huggingface(dataset_id: str = "Percena/msc-memfuse-mc10") -> MSCDataset:
    """
    Load the MSC dataset from HuggingFace and transform it to our format.
    
    Args:
        dataset_id: HuggingFace dataset identifier
    
    Returns:
        MSCDataset: Transformed dataset
    """
    print(f"Loading MSC dataset from HuggingFace: {dataset_id}")
    
    # Load dataset from HuggingFace
    try:
        dataset = load_dataset(dataset_id)
        # Assuming we want the 'train' split
        data = dataset['train']
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from HuggingFace: {e}")
    
    entries = []
    failed_entries = []
    validation_warnings = []
    
    for i, entry_data in enumerate(data):
        try:
            transformed_entry = transform_entry_from_huggingface(entry_data)
            
            # Validate the transformed entry
            validation_errors = validate_entry(transformed_entry)
            if validation_errors:
                validation_warnings.append({
                    'entry_index': i,
                    'question_id': entry_data.get('question_id', 'Unknown'),
                    'validation_errors': validation_errors
                })
                print(f"Warning: Validation issues for entry {i}: {', '.join(validation_errors)}")
            
            entries.append(transformed_entry)
        except Exception as e:
            failed_entries.append({
                'index': i,
                'question_id': entry_data.get('question_id', 'Unknown'),
                'error': str(e)
            })
            print(f"Error: Failed to transform entry {i}: {e}")
    
    # Calculate statistics
    if entries:
        total_choices = sum(len(entry.choices) for entry in entries)
        avg_choices = total_choices / len(entries)
        
        total_sessions = sum(len(entry.haystack_sessions) for entry in entries)
        avg_sessions = total_sessions / len(entries)
        
        total_messages = sum(entry.get_total_messages() for entry in entries)
        avg_messages = total_messages / len(entries)
        
        # Analyze choice distribution
        choice_counts = {}
        for entry in entries:
            num_choices = len(entry.choices)
            choice_counts[num_choices] = choice_counts.get(num_choices, 0) + 1
        
        # Analyze session distribution
        session_counts = {}
        for entry in entries:
            num_sessions = len(entry.haystack_sessions)
            session_counts[num_sessions] = session_counts.get(num_sessions, 0) + 1
    else:
        avg_choices = avg_sessions = avg_messages = 0
        choice_counts = session_counts = {}
    
    metadata = {
        'source': 'HuggingFace',
        'dataset_id': dataset_id,
        'total_entries': len(data),
        'successfully_transformed': len(entries),
        'failed_entries': len(failed_entries),
        'validation_warnings': len(validation_warnings),
        'average_choices_per_question': round(avg_choices, 2),
        'average_sessions_per_entry': round(avg_sessions, 2),
        'average_messages_per_entry': round(avg_messages, 2),
        'choice_distribution': choice_counts,
        'session_distribution': session_counts,
        'failed_entry_details': failed_entries,
        'validation_warning_details': validation_warnings
    }
    
    return MSCDataset(entries=entries, metadata=metadata)


def transform_entry_from_jsonl(entry_data: Dict[str, Any]) -> MSCEntry:
    """
    Transform a single MSC entry from JSONL format to our format.
    
    Args:
        entry_data: Dictionary containing the entry data from JSONL
    
    Returns:
        MSCEntry: Transformed entry
    """
    return MSCEntry(**entry_data)


def save_transformed_dataset_jsonl(dataset: MSCDataset, output_file: Path) -> None:
    """
    Save the transformed dataset to a JSONL file (one entry per line).
    
    Args:
        dataset: The transformed MSC dataset
        output_file: Path where to save the transformed dataset
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset.entries:
            json.dump(entry.model_dump(), f, ensure_ascii=False)
            f.write('\n')


def load_transformed_dataset_jsonl(input_file: Path) -> MSCDataset:
    """
    Load a transformed MSC dataset from a JSONL file.
    
    Args:
        input_file: Path to the input JSONL file
    
    Returns:
        MSCDataset: Loaded dataset
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
                entry = transform_entry_from_jsonl(entry_data)
                entries.append(entry)
            except Exception as e:
                failed_entries.append({
                    'line_number': line_num,
                    'error': str(e)
                })
                print(f"Warning: Failed to parse line {line_num}: {e}")
    
    # Calculate statistics for loaded data
    if entries:
        total_choices = sum(len(entry.choices) for entry in entries)
        avg_choices = total_choices / len(entries)
        
        total_sessions = sum(len(entry.haystack_sessions) for entry in entries)
        avg_sessions = total_sessions / len(entries)
        
        total_messages = sum(entry.get_total_messages() for entry in entries)
        avg_messages = total_messages / len(entries)
        
        choice_counts = {}
        for entry in entries:
            num_choices = len(entry.choices)
            choice_counts[num_choices] = choice_counts.get(num_choices, 0) + 1
    else:
        avg_choices = avg_sessions = avg_messages = 0
        choice_counts = {}
    
    metadata = {
        'source_file': str(input_file),
        'total_entries_loaded': len(entries),
        'failed_entries': len(failed_entries),
        'average_choices_per_question': round(avg_choices, 2),
        'average_sessions_per_entry': round(avg_sessions, 2),
        'average_messages_per_entry': round(avg_messages, 2),
        'choice_distribution': choice_counts,
        'failed_entry_details': failed_entries
    }
    
    return MSCDataset(entries=entries, metadata=metadata)


def transform_msc_dataset(
    input_source: str = "Percena/msc-memfuse-mc10",
    output_file: Path = None,
    source_type: str = "huggingface"
) -> MSCDataset:
    """
    Transform the MSC dataset from various sources to our standardized format.
    
    Args:
        input_source: Source of the dataset (HuggingFace dataset ID or file path)
        output_file: Optional path to save the transformed dataset
        source_type: Type of source ("huggingface" or "jsonl")
    
    Returns:
        MSCDataset: Transformed dataset
    """
    if source_type == "huggingface":
        dataset = load_and_transform_msc_dataset_from_huggingface(input_source)
    elif source_type == "jsonl":
        dataset = load_transformed_dataset_jsonl(Path(input_source))
    else:
        raise ValueError(f"Unsupported source type: {source_type}. Use 'huggingface' or 'jsonl'")
    
    # Save to file if specified
    if output_file:
        save_transformed_dataset_jsonl(dataset, output_file)
        print(f"Transformed dataset saved to: {output_file}")
    
    # Print summary
    print(f"\nMSC Dataset Transformation Summary:")
    print(f"  Source: {dataset.metadata.get('source', input_source)}")
    print(f"  Total entries: {dataset.metadata.get('successfully_transformed', len(dataset.entries))}")
    print(f"  Failed entries: {dataset.metadata.get('failed_entries', 0)}")
    print(f"  Validation warnings: {dataset.metadata.get('validation_warnings', 0)}")
    print(f"  Average choices per question: {dataset.metadata.get('average_choices_per_question', 0)}")
    print(f"  Average sessions per entry: {dataset.metadata.get('average_sessions_per_entry', 0)}")
    print(f"  Average messages per entry: {dataset.metadata.get('average_messages_per_entry', 0)}")
    
    return dataset 