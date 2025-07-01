from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
from pathlib import Path


class ConversationMessage(BaseModel):
    """Represents a single message in a conversation (OpenAI format)"""
    role: str = Field(..., description="The role of the message sender ('user' or 'assistant')")
    content: str = Field(..., description="The content of the message")


class ConversationSession(BaseModel):
    """Represents a single conversation session"""
    messages: List[ConversationMessage] = Field(..., description="List of messages in this session")


class MSCEntry(BaseModel):
    """Represents a single MSC (Multi-Session Chat) entry with conversational memory question"""
    question_id: str = Field(..., description="Unique identifier for the question")
    question: str = Field(..., description="The question about the conversation history")
    answer: str = Field(..., description="The ground truth answer")
    choices: List[str] = Field(..., description="The list of answer choices (typically 10 choices)")
    correct_choice_index: int = Field(..., description="Index of the correct answer (0-based)")
    haystack_session_ids: List[str] = Field(..., description="List of session identifiers")
    haystack_sessions: List[List[Dict[str, str]]] = Field(..., description="Multi-session dialogue history in OpenAI format")
    
    def get_correct_choice(self) -> str:
        """Get the correct choice text"""
        if 0 <= self.correct_choice_index < len(self.choices):
            return self.choices[self.correct_choice_index]
        raise ValueError(f"Invalid correct_choice_index: {self.correct_choice_index}")
    
    def get_incorrect_choices(self) -> List[str]:
        """Get all incorrect choice texts"""
        return [choice for i, choice in enumerate(self.choices) if i != self.correct_choice_index]
    
    def get_conversation_sessions(self) -> List[ConversationSession]:
        """Get structured conversation sessions"""
        sessions = []
        for session_messages in self.haystack_sessions:
            messages = [ConversationMessage(**msg) for msg in session_messages]
            sessions.append(ConversationSession(messages=messages))
        return sessions
    
    def get_total_messages(self) -> int:
        """Get total number of messages across all sessions"""
        return sum(len(session) for session in self.haystack_sessions)
    
    def get_conversation_text(self) -> str:
        """Get formatted conversation text for display purposes"""
        conversation_parts = []
        for i, session in enumerate(self.haystack_sessions, 1):
            conversation_parts.append(f"=== Session {i} ===")
            for message in session:
                role = message['role'].title()
                content = message['content']
                conversation_parts.append(f"{role}: {content}")
            conversation_parts.append("")  # Empty line between sessions
        return "\n".join(conversation_parts)


class MSCDataset(BaseModel):
    """Collection of MSC entries"""
    entries: List[MSCEntry] = Field(..., description="List of MSC entries")
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
    
    def get_average_sessions_per_entry(self) -> float:
        """Get the average number of conversation sessions per entry"""
        if not self.entries:
            return 0.0
        total_sessions = sum(len(entry.haystack_sessions) for entry in self.entries)
        return total_sessions / len(self.entries)
    
    def get_average_messages_per_entry(self) -> float:
        """Get the average number of messages per entry"""
        if not self.entries:
            return 0.0
        total_messages = sum(entry.get_total_messages() for entry in self.entries)
        return total_messages / len(self.entries) 