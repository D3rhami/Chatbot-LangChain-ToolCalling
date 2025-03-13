# Standard library imports
import logging
from typing import Optional

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
        system_prompt = """
        You are a helpful assistant that represents an e-commerce system.

        When responding to the user:
        1. DO NOT mention internal function names directly
        2. DO NOT expose internal parameter names or types
        3. DO NOT mention how the system works internally
        4. Present results in a user-friendly, conversational way
        5. If there was an error, explain it simply without technical details
        6. Be concise but complete
        7. Format the response as valid HTML with appropriate tags for:
           - Paragraphs (<p>)
           - Bold text (<strong>)
           - Tables (<table>, <tr>, <td>)
           - Lists (<ul>, <li>)
           - Headings (<h1>, <h2>, etc.)
        8. Use inline styles or classes for basic formatting (e.g., colors, alignment)
        """

        # Prepare the prompt based on the input
        if follow_up_question:
            prompt = f"""
            The user asked: "{query}"

            I need to ask a follow-up question:
            {follow_up_question}

            Please create a helpful, conversational response in HTML format that presents this follow-up question in a user-friendly way.
            """
        elif function_response:
            result_json = str(function_response.result)
            prompt = f"""
            The user asked: "{query}"

            I executed the "{function_response.function_name}" operation with these parameters:
            {function_response.arguments}

            Here are the results:
            {result_json}

            Please create a helpful, conversational response in HTML format that presents this information in a user-friendly way.
            Do not expose internal function names or technical details.
            """
        else:
            prompt = f"""
            The user asked: "{query}"

            I couldn't determine the intent or execute a function. Please create a helpful, conversational response in HTML format that asks the user for clarification.
            """

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
