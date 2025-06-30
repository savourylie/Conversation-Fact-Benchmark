#!/usr/bin/env python3

import json
import time
import logging
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union
from pydantic import BaseModel

from data_models.dream import DreamDataset, DreamEntry
from data_models.truthful_qa import TruthfulQADataset, TruthfulQAEntry
from data_models.response import Choice
from llm import LLMFactory, LLMProvider


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestResult(BaseModel):
    """Results for a single test case"""
    entry_id: str
    question_index: int
    correct_answer: int
    predicted_answer: int
    is_correct: bool
    confidence: float = 0.0
    response_time: float = 0.0
    raw_response: str = ""
    error: str = ""


class TestSuite:
    """Main test suite for evaluating LLMs on DREAM and TruthfulQA datasets"""
    
    def __init__(self, provider_type: str, model_name: str):
        self.provider_type = provider_type
        self.model_name = model_name
        self.results: List[TestResult] = []
        self.llm_provider = LLMFactory.create_provider(provider_type, model_name, logger)
        self.dataset_type = None
        
    def load_dataset(self, dataset_path: str) -> Union[DreamDataset, TruthfulQADataset]:
        """Load dataset from JSONL format (expects .json extension for HuggingFace compatibility)"""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        print(f"Loading JSONL dataset from {dataset_path}...")
        
        entries = []
        with open(dataset_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry_data = json.loads(line)
                    entries.append(entry_data)
                except Exception as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
        
        if not entries:
            raise ValueError(f"No valid entries found in {dataset_path}")
        
        # Auto-detect dataset type based on structure of first entry
        first_entry = entries[0]
        if 'dialogue_turns' in first_entry and 'questions' in first_entry:
            self.dataset_type = 'dream'
            # Import the load function from the transform module
            from data_processing.dream.transform import load_transformed_dataset_jsonl
            return load_transformed_dataset_jsonl(Path(dataset_path))
        elif 'question' in first_entry and 'choices' in first_entry and 'correct_choice_index' in first_entry:
            self.dataset_type = 'truthful_qa'
            # Import the load function from the transform module
            from data_processing.truthful_qa.transform import load_transformed_dataset_jsonl
            return load_transformed_dataset_jsonl(Path(dataset_path))
        else:
            raise ValueError(f"Unable to determine dataset type from JSONL file {dataset_path}")
    
    def test_single_entry(self, entry: Union[DreamEntry, TruthfulQAEntry], question_idx: int = 0) -> TestResult:
        """Test a single question from a dataset entry"""
        # Query model using the provider
        result = self.llm_provider.query_model(entry, question_idx)
        
        # Get correct answer and entry ID based on dataset type
        if isinstance(entry, DreamEntry):
            question = entry.questions[question_idx]
            correct_answer = question.correct_choice_index
            entry_id = entry.id
        elif isinstance(entry, TruthfulQAEntry):
            correct_answer = entry.correct_choice_index
            entry_id = str(hash(entry.question))  # Use question hash as ID for TruthfulQA
        else:
            raise ValueError(f"Unsupported entry type: {type(entry)}")
        
        # Create test result
        test_result = TestResult(
            entry_id=entry_id,
            question_index=question_idx,
            correct_answer=correct_answer,
            predicted_answer=-1,  # Default to invalid
            is_correct=False,
            response_time=result['response_time'],
            raw_response=result['raw_content']
        )
        
        if result['success']:
            model_response = result['response']
            test_result.predicted_answer = model_response.index
            test_result.is_correct = (model_response.index == correct_answer)
        else:
            test_result.error = result['error']
            
        return test_result
    
    def run_evaluation(self, dataset_path: str, max_samples: int = None) -> Dict[str, Any]:
        """Run full evaluation on the dataset"""
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        print(f"Dataset loaded: {len(dataset.entries)} entries")
        print(f"Dataset type: {self.dataset_type.upper()}")
        print(f"Provider: {self.provider_type}")
        print(f"Model: {self.model_name}")
        print("=" * 50)
        
        # Calculate total questions based on dataset type
        if self.dataset_type == 'dream':
            total_questions = sum(len(entry.questions) for entry in dataset.entries)
        else:  # truthful_qa
            total_questions = len(dataset.entries)  # One question per entry
        
        # Determine sample size
        if max_samples:
            print(f"Running evaluation on first {max_samples} questions (out of {total_questions} total)")
        else:
            print(f"Running full evaluation on all {total_questions} questions")
            max_samples = total_questions
        
        # Run tests
        questions_tested = 0
        start_time = time.time()
        
        for entry_idx, entry in enumerate(dataset.entries):
            if questions_tested >= max_samples:
                break
                
            print(f"\nProcessing entry {entry_idx + 1}/{len(dataset.entries)}")
            
            if self.dataset_type == 'dream':
                # DREAM: Multiple questions per entry
                for question_idx in range(len(entry.questions)):
                    if questions_tested >= max_samples:
                        break
                        
                    print(f"  Question {question_idx + 1}/{len(entry.questions)}: ", end="", flush=True)
                    
                    # Test this question
                    result = self.test_single_entry(entry, question_idx)
                    self.results.append(result)
                    
                    # Print result
                    if result.error:
                        print(f"ERROR - {result.error}")
                    else:
                        status = "✓ CORRECT" if result.is_correct else "✗ INCORRECT"
                        print(f"{status} (predicted: {result.predicted_answer}, actual: {result.correct_answer})")
                    
                    questions_tested += 1
            else:
                # TruthfulQA: One question per entry
                if questions_tested >= max_samples:
                    break
                    
                print(f"  Question: ", end="", flush=True)
                
                # Test this question
                result = self.test_single_entry(entry, 0)
                self.results.append(result)
                
                # Print result
                if result.error:
                    print(f"ERROR - {result.error}")
                else:
                    status = "✓ CORRECT" if result.is_correct else "✗ INCORRECT"
                    print(f"{status} (predicted: {result.predicted_answer}, actual: {result.correct_answer})")
                
                questions_tested += 1
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        stats = self.calculate_statistics(total_time)
        
        # Print summary
        self.print_summary(stats)
        
        return stats
    
    def calculate_statistics(self, total_time: float) -> Dict[str, Any]:
        """Calculate evaluation statistics"""
        total_questions = len(self.results)
        correct_answers = sum(1 for r in self.results if r.is_correct)
        errors = sum(1 for r in self.results if r.error)
        
        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        error_rate = (errors / total_questions * 100) if total_questions > 0 else 0
        
        response_times = [r.response_time for r in self.results if not r.error]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'dataset_type': self.dataset_type,
            'provider_type': self.provider_type,
            'model_name': self.model_name,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'incorrect_answers': total_questions - correct_answers - errors,
            'errors': errors,
            'accuracy': accuracy,
            'error_rate': error_rate,
            'total_time': total_time,
            'avg_response_time': avg_response_time,
            'questions_per_minute': (total_questions / total_time * 60) if total_time > 0 else 0
        }
    
    def print_summary(self, stats: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Dataset: {stats['dataset_type'].upper()}")
        print(f"Provider: {stats['provider_type']}")
        print(f"Model: {stats['model_name']}")
        print(f"Total Questions: {stats['total_questions']}")
        print(f"Correct Answers: {stats['correct_answers']}")
        print(f"Incorrect Answers: {stats['incorrect_answers']}")
        print(f"Errors: {stats['errors']}")
        print(f"Accuracy: {stats['accuracy']:.2f}%")
        print(f"Error Rate: {stats['error_rate']:.2f}%")
        print(f"Total Time: {stats['total_time']:.2f} seconds")
        print(f"Avg Response Time: {stats['avg_response_time']:.2f} seconds")
        print(f"Questions per Minute: {stats['questions_per_minute']:.1f}")
        print("=" * 60)
    
    def save_results(self, output_path: str):
        """Save detailed results to JSON"""
        results_data = {
            'metadata': {
                'dataset_type': self.dataset_type,
                'provider_type': self.provider_type,
                'model_name': self.model_name,
                'timestamp': time.time(),
                'total_questions': len(self.results)
            },
            'results': [result.model_dump() for result in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and fall back to environment variables"""
    parser = argparse.ArgumentParser(
        description="Run dataset benchmark with selectable LLM provider and model. Supports DREAM and TruthfulQA datasets."
    )

    parser.add_argument(
        "-p", "--provider",
        default=os.getenv("PROVIDER_TYPE", "ollama"),
        choices=["ollama", "openrouter"],
        help="LLM provider to use (ollama or openrouter). Defaults to env PROVIDER_TYPE or 'ollama'."
    )

    parser.add_argument(
        "-m", "--model",
        default=os.getenv("MODEL_NAME", "gemma3n:latest"),
        help="Model name to evaluate. Defaults to env MODEL_NAME or 'gemma3n:latest'."
    )

    parser.add_argument(
        "-d", "--dataset",
        default=os.getenv("DATASET_PATH", "datasets/dream/processed/full_transformed.json"),
        help="Path to dataset file (DREAM or TruthfulQA, JSONL format with .json extension). Defaults to env DATASET_PATH or DREAM dataset."
    )

    parser.add_argument(
        "-n", "--max-samples",
        type=int,
        default=int(os.getenv("MAX_SAMPLES", "20")),
        help="Maximum number of questions to evaluate (0 = full dataset). Defaults to env MAX_SAMPLES or 20."
    )

    return parser.parse_args()


def main():
    """Main entry point: parse args, validate, and run evaluation."""
    args = parse_args()

    DATASET_PATH = args.dataset
    PROVIDER_TYPE = args.provider.lower()
    MODEL_NAME = args.model
    MAX_SAMPLES = None if args.max_samples == 0 else args.max_samples
    
    print("=" * 60)
    print("DATASET EVALUATION BENCHMARK")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Provider: {PROVIDER_TYPE}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max Samples: {MAX_SAMPLES or 'Full dataset'}")
    print("=" * 60)
    
    # Validate provider type (argparse already restricts, but double check)
    if PROVIDER_TYPE not in ["ollama", "openrouter"]:
        print(f"Error: Invalid provider type '{PROVIDER_TYPE}'. Must be 'ollama' or 'openrouter'")
        return
    
    # Check for OpenRouter API key if using OpenRouter
    if PROVIDER_TYPE == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            print("Error: OPENROUTER_API_KEY environment variable is required for OpenRouter provider")
            print("Please set it with: export OPENROUTER_API_KEY=your_api_key_here")
            return
    
    # Initialize test suite
    test_suite = TestSuite(provider_type=PROVIDER_TYPE, model_name=MODEL_NAME)
    
    # Run evaluation
    try:
        stats = test_suite.run_evaluation(DATASET_PATH, max_samples=MAX_SAMPLES)
        
        # Save results
        timestamp = int(time.time())
        provider_safe = PROVIDER_TYPE.replace('/', '_')
        model_safe = MODEL_NAME.replace(':', '_').replace('/', '_')
        dataset_name = stats['dataset_type']
        results_file = f"evaluation_results_{dataset_name}_{provider_safe}_{model_safe}_{timestamp}.json"
        test_suite.save_results(results_file)
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        if test_suite.results:
            stats = test_suite.calculate_statistics(0)
            test_suite.print_summary(stats)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
