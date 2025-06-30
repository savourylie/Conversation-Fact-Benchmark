import time
from typing import Dict, Any
from ollama import chat
from pydantic import BaseModel


class OllamaClient:
    """Client for interacting with Ollama models"""
    
    def __init__(self, model_name: str = "gemma3n:latest"):
        self.model_name = model_name
    
    def query_with_structured_output(self, prompt: str, output_schema: BaseModel) -> Dict[str, Any]:
        """
        Query the model with structured output using a Pydantic schema
        
        Args:
            prompt: The input prompt for the model
            output_schema: Pydantic model class defining the expected output structure
            
        Returns:
            Dict containing success status, parsed response, timing, and raw content
        """
        start_time = time.time()
        
        try:
            response = chat(
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ],
                model=self.model_name,
                options={
                    "temperature": 0.1  # A low temperature for a factual answer
                },
                format=output_schema.model_json_schema(),
            )
            
            response_time = time.time() - start_time
            
            # Parse structured response
            model_response = output_schema.model_validate_json(response.message.content)
            
            return {
                'success': True,
                'response': model_response,
                'response_time': response_time,
                'raw_content': response.message.content
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time,
                'raw_content': ''
            }
    
    def query(self, prompt: str) -> Dict[str, Any]:
        """
        Query the model with plain text output
        
        Args:
            prompt: The input prompt for the model
            
        Returns:
            Dict containing success status, response text, timing, and error info
        """
        start_time = time.time()
        
        try:
            response = chat(
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ],
                model=self.model_name,
            )
            
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'response': response.message.content,
                'response_time': response_time,
                'raw_content': response.message.content
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time,
                'raw_content': ''
            }
