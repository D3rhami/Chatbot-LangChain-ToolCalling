# Standard library imports
import logging
from typing import List

# LangChain imports
from langchain_openai import ChatOpenAI

# Import from process module
from process import FunctionResponse, tools

# Set up logging
logger = logging.getLogger(__name__)


def generate_response_from_function_results(query: str, function_responses: List[FunctionResponse]) -> str:
    """
    Generate a human-readable response based on function results when LLM is unavailable

    Args:
        query: The original user query
        function_responses: List of function responses

    Returns:
        A human-readable response based on the function results
    """
    if not function_responses:
        return f"I processed your query '{query}', but couldn't generate a proper response due to technical limitations."

    # For multiply_numbers specifically
    multiply_results = [resp for resp in function_responses if resp.function_name == "multiply_numbers"]
    if len(multiply_results) == 2:
        # Handle the specific case we saw in the logs
        first_result = multiply_results[0].result
        second_result = multiply_results[1].result
        return f"The product of {multiply_results[0].arguments['a']} × {multiply_results[0].arguments['b']} is {first_result}. When that answer is multiplied by {multiply_results[1].arguments['b']}, the result is {second_result}."

    # Generic response based on function results
    result_descriptions = []
    for resp in function_responses:
        if resp.function_name == "multiply_numbers":
            result_descriptions.append(
                    f"The product of {resp.arguments['a']} and {resp.arguments['b']} is {resp.result}")
        elif resp.function_name == "summarize_text":
            result_descriptions.append(f"Analysis of your text: {resp.result}")
        else:
            result_descriptions.append(f"Result from {resp.function_name}: {resp.result}")

    return f"Based on your query '{query}', here are the results: {'; '.join(result_descriptions)}"


async def postprocess_results(messages: List, query: str, function_responses: List[FunctionResponse], api_key: str,
                              model_name: str) -> str:
    """
    Post-process results by getting a final response from the LLM

    Args:
        messages: The conversation history
        query: The original user query
        function_responses: List of function responses
        api_key: The API key for the LLM
        model_name: The name of the model to use

    Returns:
        The final response to the user
    """
    # Initialize the LLM with retry settings
    llm = ChatOpenAI(
            model=model_name,
            base_url="https://api.avalai.ir/v1",
            api_key=api_key,
            max_retries=3,
            request_timeout=60
    )

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Prepare final response - handle potential rate limits
    try:
        # Get the final response from the LLM
        logger.info("Requesting final response from LLM")
        final_response = llm_with_tools.invoke(messages)
        logger.info("Final response received successfully")
        return final_response.content
    except Exception as e:
        logger.error(f"Error getting final response: {str(e)}")
        # If we hit rate limits again, construct a reasonable response from what we have
        fallback_response = generate_response_from_function_results(query, function_responses)
        logger.info(f"Generated fallback response: {fallback_response}")
        return fallback_response
