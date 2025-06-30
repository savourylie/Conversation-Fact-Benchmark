import json
from pathlib import Path
from typing import List, Dict, Any
import re

from data_models.dream import (
    DialogueTurn, 
    Question, 
    DreamEntry, 
    DreamDataset
)


def parse_dialogue_turn(turn_text: str) -> DialogueTurn:
    """
    Parse a dialogue turn from the original format.
    Expected format: "Speaker: dialogue text"
    """
    # Use regex to split on the first colon
    match = re.match(r'^([^:]+):\s*(.*)$', turn_text.strip())
    if match:
        speaker = match.group(1).strip()
        text = match.group(2).strip()
    else:
        # Fallback if format doesn't match expected pattern
        speaker = "Unknown"
        text = turn_text.strip()
    
    return DialogueTurn(speaker=speaker, text=text)


def transform_question(question_data: Dict[str, Any]) -> Question:
    """
    Transform a question from the original format to the new format.
    """
    question_text = question_data["question"]
    answer_text = question_data["answer"]
    choices = question_data["choice"]
    
    # Find the index of the correct answer
    correct_choice_index = -1
    for i, choice in enumerate(choices):
        if choice.strip() == answer_text.strip():
            correct_choice_index = i
            break
    
    if correct_choice_index == -1:
        # If exact match not found, try to find partial match
        for i, choice in enumerate(choices):
            if choice.strip().lower() == answer_text.strip().lower():
                correct_choice_index = i
                break
    
    if correct_choice_index == -1:
        raise ValueError(f"Could not find correct answer '{answer_text}' in choices: {choices}")
    
    return Question(
        question_text=question_text,
        answer_text=answer_text,
        choices=choices,
        correct_choice_index=correct_choice_index
    )


def transform_entry(entry_data: Dict[str, Any]) -> DreamEntry:
    """
    Transform a single DREAM entry from the original format to the new format.
    """
    # Parse dialogue turns
    dialogue_turns = []
    for turn_text in entry_data["conversation"]:
        dialogue_turn = parse_dialogue_turn(turn_text)
        dialogue_turns.append(dialogue_turn)
    
    # Transform questions
    questions = []
    for question_data in entry_data["questions"]:
        question = transform_question(question_data)
        questions.append(question)
    
    return DreamEntry(
        id=entry_data["id"],
        dialogue_turns=dialogue_turns,
        questions=questions
    )


def load_and_transform_dream_dataset(input_file: Path) -> DreamDataset:
    """
    Load the original DREAM dataset JSON file and transform it to the new format.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    entries = []
    failed_entries = []
    
    for i, entry_data in enumerate(original_data):
        try:
            transformed_entry = transform_entry(entry_data)
            entries.append(transformed_entry)
        except Exception as e:
            failed_entries.append({
                'index': i,
                'id': entry_data.get('id', 'unknown'),
                'error': str(e)
            })
            print(f"Warning: Failed to transform entry {i} (id: {entry_data.get('id', 'unknown')}): {e}")
    
    metadata = {
        'source_file': str(input_file),
        'total_entries': len(original_data),
        'successfully_transformed': len(entries),
        'failed_entries': len(failed_entries),
        'failed_entry_details': failed_entries
    }
    
    return DreamDataset(entries=entries, metadata=metadata)


def save_transformed_dataset_jsonl(dataset: DreamDataset, output_file: Path) -> None:
    """
    Save the transformed dataset to a JSONL file (one entry per line).
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset.entries:
            json.dump(entry.model_dump(), f, ensure_ascii=False)
            f.write('\n')


def load_transformed_dataset_jsonl(input_file: Path) -> DreamDataset:
    """
    Load a transformed DREAM dataset from a JSONL file.
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
                entry = DreamEntry(**entry_data)
                entries.append(entry)
            except Exception as e:
                failed_entries.append({
                    'line_number': line_num,
                    'error': str(e)
                })
                print(f"Warning: Failed to parse line {line_num}: {e}")
    
    metadata = {
        'source_file': str(input_file),
        'format': 'jsonl',
        'total_entries': len(entries),
        'failed_entries': len(failed_entries),
        'failed_entry_details': failed_entries
    }
    
    return DreamDataset(entries=entries, metadata=metadata)


def transform_dream_full_dataset(
    input_file: Path = None, 
    output_file: Path = None
) -> DreamDataset:
    """
    Main function to transform the full DREAM dataset to JSONL format.
    
    Args:
        input_file: Path to the input JSON file (defaults to datasets/dream/raw/full.json)
        output_file: Path to save the transformed dataset (optional)
    
    Returns:
        DreamDataset: The transformed dataset
    """
    if input_file is None:
        input_file = Path("datasets/dream/raw/full.json")
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading and transforming dataset from: {input_file}")
    dataset = load_and_transform_dream_dataset(input_file)
    
    print(f"Transformation complete:")
    print(f"  - Total entries: {dataset.metadata['total_entries']}")
    print(f"  - Successfully transformed: {dataset.metadata['successfully_transformed']}")
    print(f"  - Failed entries: {dataset.metadata['failed_entries']}")
    
    if output_file:
        print(f"Saving transformed dataset to: {output_file} (JSONL format)")
        save_transformed_dataset_jsonl(dataset, output_file)
        print("Save complete!")
    
    return dataset


if __name__ == "__main__":
    # Transform to JSONL format (but with .json extension for HuggingFace compatibility)
    transformed_dataset = transform_dream_full_dataset(
        output_file=Path("datasets/dream/processed/full_transformed.json")
    )
