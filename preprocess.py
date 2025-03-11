# Standard library imports
import logging
import random
import time
from typing import List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
# LangChain imports
from langchain_openai import ChatOpenAI

# Import from process module
from process import execute_function_call, FunctionResponse, tools

# Set up logging
logger = logging.getLogger(__name__)


async def preprocess_query(query: str, api_key: str, model_name: str) -> Tuple[
    List, Optional[AIMessage], List[FunctionResponse]]:
    """
    Preprocess the query by sending it to the LLM and getting tool calls

    Args:
        query: The user's query
        api_key: The API key for the LLM
        model_name: The name of the model to use

    Returns:
        Tuple containing messages, LLM output, and function responses
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

    # Create the initial message
    messages = [HumanMessage(content=query)]

    # Handle rate limiting with retries
    max_retries = 3
    llm_output = None

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}: Sending initial query to LLM")
            llm_output = llm_with_tools.invoke(messages)
            logger.info(f"LLM response received successfully")
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                # Wait and retry with exponential backoff
                wait_time = (2 ** attempt) + random.random()
                logger.info(f"Rate limit hit, retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All retries failed or other error: {str(e)}")
                return messages, None, []

    # If no response from LLM, return empty results
    if llm_output is None:
        return messages, None, []

    # Add LLM response to messages
    messages.append(llm_output)

    # Process tool calls if any
    function_responses = []

    if hasattr(llm_output, 'tool_calls') and llm_output.tool_calls:
        from langchain_core.messages import ToolMessage

        logger.info(f"LLM requested {len(llm_output.tool_calls)} tool call(s)")

        for tool_call in llm_output.tool_calls:
            function_response = execute_function_call(tool_call)
            if function_response:
                function_responses.append(function_response)
                # Add the tool message to continue the conversation
                messages.append(ToolMessage(
                        content=str(function_response.result),
                        tool_call_id=tool_call["id"]
                ))

    return messages, llm_output, function_responses
