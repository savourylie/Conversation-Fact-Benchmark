from litellm import completion
from data_models.response import Choice
import os
import json
import logging
from prompts.utils import Prompt


def call_openrouter(
        prompt: Prompt, 
        model: str, 
        choices_length: int = None,
        logger: logging.Logger = logging.getLogger(__name__)
    ) -> Choice:
    """Call the OpenRouter LLM via LiteLLM with structured output."""
    
    system_prompt, user_prompt = prompt.system, prompt.user
    
    # Ensure model has openrouter/ prefix for LiteLLM
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=Choice,
            temperature=0.1
        )

        # Parse the response content as JSON
        response_content = response.choices[0].message.content
        parsed_response = Choice(**json.loads(response_content))
        
        if choices_length is not None and not (0 <= parsed_response.index < choices_length):
            logger.warning(f"Model returned choice index {parsed_response.index} which is out of range (0-{choices_length-1}). Defaulting to 0.")
            parsed_response.index = 0

        return parsed_response
    
    except Exception as e:
        logger.error(f"Error using structured completion with OpenRouter: {e}. Falling back to regular completion.")

        try:
            regular_completion = completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            response_text = regular_completion.choices[0].message.content
            logger.info(f"OpenRouter fallback response text: {response_text}")
            
            # Try to parse JSON from the response text
            try:
                # Extract JSON part if it's embedded
                import re
                json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    index = int(parsed_json.get("index", parsed_json.get("choice", 0)))  # Support both field names
                    description = parsed_json.get("description", "No description provided in fallback.")
                    reasoning = parsed_json.get("reasoning", "No reasoning provided in fallback.")

                    if choices_length is not None and not (0 <= index < choices_length):
                        logger.warning(f"Fallback: Model returned choice index {index} out of range. Defaulting to 0.")
                        index = 0
                    
                    return Choice(
                        index=index, 
                        description=description,
                        reasoning=reasoning
                    )
                else:
                    logger.error("Fallback: No JSON object found in the response.")
            except json.JSONDecodeError as json_err:
                logger.error(f"Fallback: Error parsing JSON from response: {json_err}")
            except Exception as ex:
                logger.error(f"Fallback: Error processing response: {ex}")

            # If JSON parsing fails, return a default error response
            return Choice(
                index=0, # Default to first choice
                description="Error occurred during response parsing",
                reasoning=f"Failed to parse LLM response: {response_text[:200]}..."
            )
        except Exception as e2:
            logger.error(f"Error using regular completion with OpenRouter: {e2}")
            return Choice(
                index=0, # Default to first choice
                description="Error occurred during LLM call",
                reasoning=f"Error calling LLM: {str(e2)}"
            )
