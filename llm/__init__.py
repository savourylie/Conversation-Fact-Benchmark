from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import logging
from data_models.dream import DreamEntry
from data_models.truthful_qa import TruthfulQAEntry
from data_models.response import Choice
from prompts.utils import Prompt
from prompts.prompt_manager import PromptManager


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, model_name: str, logger: logging.Logger = None):
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def query_model(self, entry: Union[DreamEntry, TruthfulQAEntry], question_idx: int = 0) -> Dict[str, Any]:
        """Query the model with a dataset entry and question index"""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider"""
    
    def __init__(self, model_name: str, logger: logging.Logger = None):
        super().__init__(model_name, logger)
        from llm.ollama import OllamaClient
        self.ollama_client = OllamaClient(model_name=model_name)
    
    def query_model(self, entry: Union[DreamEntry, TruthfulQAEntry], question_idx: int = 0) -> Dict[str, Any]:
        """Query Ollama model with combined prompt"""
        # Create combined prompt based on entry type
        combined_prompt = self._create_combined_prompt(entry, question_idx)
        
        # Query model
        result = self.ollama_client.query_with_structured_output(combined_prompt, Choice)
        
        return result
    
    def _create_combined_prompt(self, entry: Union[DreamEntry, TruthfulQAEntry], question_idx: int) -> str:
        """Create combined system + user prompt for Ollama based on entry type"""
        if isinstance(entry, DreamEntry):
            return self._create_dream_combined_prompt(entry, question_idx)
        elif isinstance(entry, TruthfulQAEntry):
            return self._create_truthful_qa_combined_prompt(entry)
        else:
            raise ValueError(f"Unsupported entry type: {type(entry)}")

    def _create_dream_combined_prompt(self, entry: DreamEntry, question_idx: int) -> str:
        """Create combined prompt for DREAM dataset"""
        # Get system prompt
        system_prompt = PromptManager.get_prompt("dialog_qa_system")
        
        # Get user prompt
        question = entry.questions[question_idx]
        user_prompt = PromptManager.get_prompt(
            "dialog_qa_user",
            dialogue_turns=[
                {"speaker": turn.speaker, "text": turn.text} 
                for turn in entry.dialogue_turns
            ],
            question_text=question.question_text,
            choices=question.choices
        )
        
        # Combine prompts (system first, then user)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        return combined_prompt

    def _create_truthful_qa_combined_prompt(self, entry: TruthfulQAEntry) -> str:
        """Create combined prompt for TruthfulQA dataset"""
        # Get system prompt
        system_prompt = PromptManager.get_prompt("truthful_qa_system")
        
        # Get user prompt
        user_prompt = PromptManager.get_prompt(
            "truthful_qa_user",
            question_text=entry.question,
            choices=entry.choices
        )
        
        # Combine prompts (system first, then user)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        return combined_prompt


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider"""
    
    def __init__(self, model_name: str, logger: logging.Logger = None):
        super().__init__(model_name, logger)
    
    def query_model(self, entry: Union[DreamEntry, TruthfulQAEntry], question_idx: int = 0) -> Dict[str, Any]:
        """Query OpenRouter model with separate system/user prompts"""
        from llm.openrouter import call_openrouter
        import time
        
        # Create separate prompts based on entry type
        prompt = self._create_prompt(entry, question_idx)
        choices_length = self._get_choices_length(entry, question_idx)
        
        try:
            start_time = time.time()
            response = call_openrouter(
                prompt=prompt,
                model=self.model_name,
                choices_length=choices_length,
                logger=self.logger
            )
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'response': response,
                'response_time': response_time,
                'raw_content': f"Description: {response.description}, Reasoning: {response.reasoning}",
                'error': None
            }
        except Exception as e:
            self.logger.error(f"Error querying OpenRouter: {e}")
            return {
                'success': False,
                'response': None,
                'response_time': 0.0,
                'raw_content': "",
                'error': str(e)
            }
    
    def _create_prompt(self, entry: Union[DreamEntry, TruthfulQAEntry], question_idx: int) -> Prompt:
        """Create separate system and user prompts based on entry type"""
        if isinstance(entry, DreamEntry):
            return self._create_dream_prompt(entry, question_idx)
        elif isinstance(entry, TruthfulQAEntry):
            return self._create_truthful_qa_prompt(entry)
        else:
            raise ValueError(f"Unsupported entry type: {type(entry)}")

    def _create_dream_prompt(self, entry: DreamEntry, question_idx: int) -> Prompt:
        """Create separate system and user prompts for DREAM dataset"""
        # Get system prompt
        system_prompt = PromptManager.get_prompt("dialog_qa_system")
        
        # Get user prompt
        question = entry.questions[question_idx]
        user_prompt = PromptManager.get_prompt(
            "dialog_qa_user",
            dialogue_turns=[
                {"speaker": turn.speaker, "text": turn.text} 
                for turn in entry.dialogue_turns
            ],
            question_text=question.question_text,
            choices=question.choices
        )
        
        return Prompt(system=system_prompt, user=user_prompt)

    def _create_truthful_qa_prompt(self, entry: TruthfulQAEntry) -> Prompt:
        """Create separate system and user prompts for TruthfulQA dataset"""
        # Get system prompt
        system_prompt = PromptManager.get_prompt("truthful_qa_system")
        
        # Get user prompt
        user_prompt = PromptManager.get_prompt(
            "truthful_qa_user",
            question_text=entry.question,
            choices=entry.choices
        )
        
        return Prompt(system=system_prompt, user=user_prompt)

    def _get_choices_length(self, entry: Union[DreamEntry, TruthfulQAEntry], question_idx: int) -> int:
        """Get the number of choices for the given entry and question"""
        if isinstance(entry, DreamEntry):
            return len(entry.questions[question_idx].choices)
        elif isinstance(entry, TruthfulQAEntry):
            return len(entry.choices)
        else:
            raise ValueError(f"Unsupported entry type: {type(entry)}")


class LLMFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(provider_type: str, model_name: str, logger: logging.Logger = None) -> LLMProvider:
        """Create an LLM provider based on the provider type"""
        provider_type = provider_type.lower()
        
        if provider_type == "ollama":
            return OllamaProvider(model_name, logger)
        elif provider_type == "openrouter":
            return OpenRouterProvider(model_name, logger)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}. Supported types: 'ollama', 'openrouter'") 