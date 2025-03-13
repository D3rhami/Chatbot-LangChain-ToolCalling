# Standard library imports
import logging
from typing import Optional
from get_prompts import prompts

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Import from process
from process import FunctionResponse

# Set up logging
logger = logging.getLogger(__name__)


async def postprocess_result(
        query: str,
        function_response: Optional[FunctionResponse],
        follow_up_question: Optional[str],
        model_name: str,
        api_key: SecretStr
) -> str:
    """
    Post-process results by generating a human-friendly response with HTML formatting.

    Args:
        query: The original user query
        function_response: The function response (if any)
        follow_up_question: A follow-up question (if any)
        model_name: The name of the LLM model to use
        api_key: API key for the LLM

    Returns:
        String containing the human-friendly response in a html format
    """
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
                model=model_name,
                base_url="https://api.avalai.ir/v1",
                api_key=api_key,
                max_retries=2
        )

        # Create a system message with guidelines for generating HTML responses
        system_prompt = prompts["system_prompt"]["value"]

        # Prepare the prompt based on the input
        if follow_up_question:
            prompt = prompts["follow_up_prompt"]["value"].format(
                    query=query,
                    follow_up_question=follow_up_question
            )
        elif function_response:
            prompt = prompts["function_response_prompt"]["value"].format(
                    query=query,
                    function_response=function_response,
                    result_json=str(function_response.result)
            )
        else:
            prompt = prompts["unknown_intent_prompt"]["value"].format(query=query)

        # Get the humanized response
        messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
        ]

        response = await llm.ainvoke(messages)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error in postprocess_result: {str(e)}")

        # Fallback response in case of errors
        return "<p>Sorry, I encountered an issue processing your request. Please try again.</p>"
