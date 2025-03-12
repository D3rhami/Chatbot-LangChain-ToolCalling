# Standard library imports
import logging
import random
import time
from typing import List, Optional, Tuple

# LangChain imports
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Import from process module
from process import execute_function_call, FunctionResponse, tools

# Set up logging
logger = logging.getLogger(__name__)


async def preprocess_query(messages: List, api_key: SecretStr, model_name: str) -> Tuple[
    List, Optional[AIMessage], List[FunctionResponse]]:
    """
    Preprocess the query by sending it to the LLM and getting tool calls

    Args:
        messages: The conversation history
        api_key: The API key for the LLM
        model_name: The name of the model to use

    Returns:
        Tuple containing messages, LLM output, and function responses
    """
    llm = ChatOpenAI(
            model=model_name,
            base_url="https://api.avalai.ir/v1",  # Removed extra spaces
            api_key=api_key,
            max_retries=3
    )

    # Get the current tools from the registry
    current_tools = tools()
    llm_with_tools = llm.bind_tools(current_tools)

    llm_output = _invoke_llm_with_retries(llm_with_tools, messages, max_retries=3)

    if llm_output is None:
        return messages, None, []

    function_responses = _process_tool_calls(llm_output, messages)

    return messages, llm_output, function_responses


def _invoke_llm_with_retries(llm_with_tools, messages, max_retries):
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}: Sending query with {len(messages)} messages to LLM")
            return llm_with_tools.invoke(messages)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.random()
                logger.info(f"Rate limit hit, retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All retries failed or other error: {str(e)}")
                return None
    return None


def _process_tool_calls(llm_output, messages):
    function_responses = []
    if hasattr(llm_output, 'tool_calls') and llm_output.tool_calls:
        from langchain_core.messages import ToolMessage
        logger.info(f"LLM requested {len(llm_output.tool_calls)} tool call(s)")

        # Store reference to the AIMessage with tool_calls
        tool_call_msg = llm_output
        messages.append(tool_call_msg)  # Append this message first

        for tool_call in llm_output.tool_calls:
            function_response = execute_function_call(tool_call)
            if function_response:
                function_responses.append(function_response)
                messages.append(ToolMessage(
                        content=str(function_response.result),
                        tool_call_id=tool_call["id"]
                ))
    return function_responses