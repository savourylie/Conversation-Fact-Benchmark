from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
from pathlib import Path


class TruthfulQAEntry(BaseModel):
    """Represents a single TruthfulQA entry with question and multiple choice options"""
    question: str = Field(..., description="The question text")
    choices: List[str] = Field(..., description="The list of answer choices")
    correct_choice_index: int = Field(..., description="Index of the correct answer (0-based)")
    
    def get_correct_choice(self) -> str:
        """Get the correct choice text"""
        if 0 <= self.correct_choice_index < len(self.choices):
            return self.choices[self.correct_choice_index]
        raise ValueError(f"Invalid correct_choice_index: {self.correct_choice_index}")
    
    def get_incorrect_choices(self) -> List[str]:
        """Get all incorrect choice texts"""
        return [choice for i, choice in enumerate(self.choices) if i != self.correct_choice_index]


class TruthfulQADataset(BaseModel):
    """Collection of TruthfulQA entries"""
    entries: List[TruthfulQAEntry] = Field(..., description="List of TruthfulQA entries")
    metadata: Dict = Field(default_factory=dict, description="Dataset metadata including source file and statistics")
    
    def get_total_questions(self) -> int:
        """Get the total number of questions in the dataset"""
        return len(self.entries)
    
    def get_average_choices_per_question(self) -> float:
        """Get the average number of choices per question"""
        if not self.entries:
            return 0.0
        total_choices = sum(len(entry.choices) for entry in self.entries)
        return total_choices / len(self.entries)
