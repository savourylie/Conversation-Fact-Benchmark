from pydantic import BaseModel, Field


class Choice(BaseModel):
    """Represents a single answer choice"""
    description: str = Field(..., description="A short description of what was discussed in the conversation")
    reasoning: str = Field(..., description="The reasoning for the answer choice")
    index: int = Field(..., description="The index of the answer choice")