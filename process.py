# Standard library imports
import logging
from typing import Dict, Any, List, Optional

# LangChain imports
from langchain_core.tools import tool
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)


# Define our tools (functions that the AI can call)
@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers and return the result.

    Args:
        a: The first number
        b: The second number

    Returns:
        The product of the two numbers
    """
    logger.info(f"Executing multiply_numbers with a={a}, b={b}")
    result = a * b
    logger.info(f"Result of multiplication: {result}")
    return result


@tool
def summarize_text(text: str) -> Dict[str, Any]:
    """Analyze text and return summary information.

    Args:
        text: The text to analyze

    Returns:
        A dictionary containing the length of the text and other metrics
    """
    logger.info(f"Executing summarize_text with text length: {len(text)}")
    words = text.split()
    result = {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
    }
    logger.info(f"Text summary result: {result}")
    return result


# Create a mapping of tool names to functions
tools = [multiply_numbers, summarize_text]
tool_mapping = {
        "multiply_numbers": multiply_numbers,
        "summarize_text": summarize_text
}


class FunctionResponse(BaseModel):
    function_name: str
    arguments: Dict[str, Any]
    result: Any

    def model_dump(self):
        """Compatible method for both Pydantic v1 and v2"""
        if hasattr(super(), "model_dump"):
            return super().model_dump()
        return self.dict()  # For Pydantic v1 compatibility


def execute_function_call(tool_call):
    """
    Execute a function call and handle potential errors

    Args:
        tool_call: The tool call object from the LLM

    Returns:
        FunctionResponse object or None if the tool is not found
    """
    tool_name = tool_call["name"].lower()
    logger.info(f"Processing tool call: {tool_name}")

    tool = tool_mapping.get(tool_name)

    if tool:
        try:
            # Execute the function with the arguments provided by the LLM
            tool_args = tool_call["args"]
            logger.info(f"Tool arguments: {tool_args}")

            tool_output = tool.invoke(tool_args)
            logger.info(f"Tool output: {tool_output}")

            return FunctionResponse(
                    function_name=tool_name,
                    arguments=tool_args,
                    result=tool_output
            )
        except Exception as e:
            # Handle any errors in function execution
            logger.error(f"Error executing function {tool_name}: {str(e)}")
            return FunctionResponse(
                    function_name=tool_name,
                    arguments=tool_call["args"],
                    result=f"Error executing function: {str(e)}"
            )
    else:
        logger.warning(f"Unknown tool requested: {tool_name}")
    return None
