from pydantic import BaseModel, Field
from typing import List, Optional
import json
from pathlib import Path


class DialogueTurn(BaseModel):
    """Represents a single turn in the dialogue"""
    speaker: str  # 'M', 'W', 'Speaker', etc.
    text: str


class Question(BaseModel):
    """Represents a question with multiple choice answers"""
    question_text: str = Field(..., description="The question that was asked")
    answer_text: str = Field(..., description="The original correct answer text for reference")
    choices: List[str] = Field(..., description="The list of answer choices")
    correct_choice_index: int = Field(..., description="Index of the correct answer (0-based)")


class DreamEntry(BaseModel):
    """Improved DREAM dataset entry with index-based answers and dialogue summary"""
    id: str
    dialogue_turns: List[DialogueTurn] = Field(..., description="The list of dialogue turns")
    questions: List[Question] = Field(..., description="The list of questions")


class DreamDataset(BaseModel):
    """Collection of DREAM entries"""
    entries: List[DreamEntry]
    metadata: dict = Field(default_factory=dict)