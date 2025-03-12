# Standard library imports
import logging
from typing import List

# LangChain imports
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Import from process module
from process import FunctionResponse, tools, generate_system_knowledge

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
        return "I need specific information to help you with one of my available functions. Please provide more details."

    # Generic response based on function results
    result_descriptions = []
    for resp in function_responses:
        if resp.function_name == "check_order":
            if "error" in resp.result:
                result_descriptions.append(f"I couldn't find the order you requested: {resp.result.get('error')}")
            else:
                status = resp.result.get("status", "Unknown")
                customer = resp.result.get("customer", {}).get("name", "Unknown customer")
                items_count = len(resp.result.get("items", []))
                total_price = resp.result.get("total_price", 0)
                result_descriptions.append(
                    f"Order #{resp.result.get('order_id')} for {customer} is currently {status}. " +
                    f"It contains {items_count} items with a total price of ${total_price:.2f}."
                )
        elif resp.function_name == "place_order":
            if "error" in resp.result:
                result_descriptions.append(f"I couldn't place your order: {resp.result.get('error')}")
            else:
                order_id = resp.result.get("order_id")
                customer = resp.result.get("customer", {}).get("name")
                products_count = len(resp.result.get("products", []))
                total_price = resp.result.get("total_price", 0)
                result_descriptions.append(
                    f"Successfully placed order #{order_id} for {customer}. " +
                    f"Added {products_count} products with a total price of ${total_price:.2f}."
                )
        elif resp.function_name == "check_product_inventory":
            if "error" in resp.result:
                result_descriptions.append(f"I couldn't check the inventory: {resp.result.get('error')}")
            elif "product" in resp.result:
                product = resp.result.get("product", {})
                result_descriptions.append(
                    f"Product '{product.get('name')}' (ID: {product.get('id')}) has " +
                    f"{product.get('stock_quantity')} items in stock. Price: ${product.get('price'):.2f}."
                )
            elif "products" in resp.result:
                count = resp.result.get("count", 0)
                result_descriptions.append(
                    f"Found {count} products in inventory. " +
                    f"Total items in stock: {resp.result.get('total_items_in_stock', 0)}."
                )
        else:
            # For new functions that might be added in the future
            # Don't expose internal function names or details
            result_descriptions.append(f"Operation completed successfully: {resp.result}")

    return f"Here are the results: {'; '.join(result_descriptions)}"


async def postprocess_results(messages: List, query: str, function_responses: List[FunctionResponse],
                              api_key: SecretStr,
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
            base_url="https://api.avalai.ir/v1",  # Removed extra spaces
            api_key=api_key,
            max_retries=3
    )

    # Get the current tools from the registry
    current_tools = tools()
    llm_with_tools = llm.bind_tools(current_tools)

    # Prepare a security prompt to ensure we don't leak backend details
    security_system_prompt = """
    When responding to the user:
    1. DO NOT mention internal function names directly
    2. DO NOT expose internal parameter names or types
    3. DO NOT mention how the system works internally
    4. Present results in a user-friendly way without technical jargon
    5. Focus only on the results that are relevant to the user

    For example, instead of saying "I called the check_order function with parameter order_id=5", 
    say "I've checked the status of your order #5 for you".
    """

    # Prepare final response - handle potential rate limits
    try:
        # Get the final response from the LLM
        logger.info("Requesting final response from LLM")

        # Create a temporary set of messages with the security prompt
        from langchain_core.messages import SystemMessage

        # Check if we already have a system message
        has_system = any(isinstance(msg, SystemMessage) for msg in messages)

        temp_messages = messages.copy()
        if has_system:
            # Find and update the existing system message
            for i, msg in enumerate(temp_messages):
                if isinstance(msg, SystemMessage):
                    original_content = msg.content
                    temp_messages[i] = SystemMessage(content=f"{original_content}\n\n{security_system_prompt}")
                    break
        else:
            # Add a new system message with both contents
            system_message = SystemMessage(content=f"{generate_system_knowledge()}\n\n{security_system_prompt}")
            temp_messages.insert(0, system_message)

        final_response = llm_with_tools.invoke(temp_messages)
        logger.info("Final response received successfully")

        # If no function was called, ensure we don't provide a generic response
        if not function_responses:
            return "I need specific information to help you with one of my available functions. Please provide more details."

        return final_response.content
    except Exception as e:
        logger.error(f"Error getting final response: {str(e)}")
        # If we hit rate limits again, construct a reasonable response from what we have
        fallback_response = generate_response_from_function_results(query, function_responses)
        logger.info(f"Generated fallback response: {fallback_response}")
        return fallback_response
