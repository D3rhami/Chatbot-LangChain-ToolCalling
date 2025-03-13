# Standard library imports
import logging
from typing import Any, Dict, List

# LangChain imports
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

# Set up logging
logger = logging.getLogger(__name__)


class UserIntent(BaseModel):
    """Model for capturing user intent and extracted entities"""
    primary_intent: str = Field(description="The primary function the user is trying to access")
    summary: str = Field(description="Brief summary of user request in 1-2 sentences")
    extracted_entities: Dict[str, Any] = Field(description="Key-value pairs of information extracted from user input")
    confidence: float = Field(description="Confidence score between 0-1 for the primary intent")

    def has_required_entities(self, required_entities: List[str]) -> Dict[str, bool]:
        """Checks if all required entities are present"""
        return {entity: entity in self.extracted_entities for entity in required_entities}

    def missing_entities(self, required_entities: List[str]) -> List[str]:
        """Returns a list of missing required entities"""
        return [entity for entity in required_entities if entity not in self.extracted_entities]


async def preprocess_query(query: str, model_name: str, api_key: SecretStr) -> UserIntent:
    """
    Preprocess the user query to understand intent and extract entities

    Args:
        query: The user's query
        model_name: The name of the LLM model to use
        api_key: API key for the LLM

    Returns:
        UserIntent object containing the extracted information
    """
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
                model=model_name,
                base_url="https://api.avalai.ir/v1",
                api_key=api_key,
                max_retries=2
        )

        # Create the preprocessing prompt
        preprocessing_prompt = f"""
        Analyze the following user query and extract:
        1. The primary intent (one of: check_order, place_order, check_product_inventory, or unknown)
        2. A brief 1-2 sentence summary of what the user is asking for
        3. Any key information entities like:
           - order_id: Any order ID numbers mentioned
           - customer_name: Customer's name if provided
           - phone_number: Phone number if provided
           - product_id: Any product IDs mentioned
           - product_name: Any product names mentioned
        4. A confidence score (0-1) for how certain you are about the primary intent

        Format the response as valid JSON with these keys: primary_intent, summary, extracted_entities, confidence

        USER QUERY: {query}
        """

        # Get the analysis from the LLM
        human_message = HumanMessage(content=preprocessing_prompt)
        response = await llm.ainvoke([human_message])

        # Parse the response
        import json
        import re

        # Extract the JSON part from the response
        json_match = re.search(r'\{.*}', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            intent_data = json.loads(json_str)

            # Create and return the UserIntent object
            return UserIntent(
                    primary_intent=intent_data.get("primary_intent", "unknown"),
                    summary=intent_data.get("summary", ""),
                    extracted_entities=intent_data.get("extracted_entities", {}),
                    confidence=float(intent_data.get("confidence", 0.0))
            )
        else:
            logger.warning("Failed to extract JSON from LLM response")
            # Return a default UserIntent with unknown intent
            return UserIntent(
                    primary_intent="unknown",
                    summary="Could not understand the user query",
                    extracted_entities={},
                    confidence=0.0
            )

    except Exception as e:
        logger.error(f"Error in preprocess_query: {str(e)}")
        # Return a default UserIntent with error info
        return UserIntent(
                primary_intent="unknown",
                summary=f"Error processing query: {str(e)}",
                extracted_entities={},
                confidence=0.0
        )
