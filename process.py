# Standard library imports
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

# LangChain imports
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Import our database module
from database import check_inventory, check_order_status, register_order
from function_manager import FunctionRegistry, FunctionResponse
# Import from preprocess
from preprocess import UserIntent

# Set up logging
logger = logging.getLogger(__name__)

# Initialize the function registry
registry = FunctionRegistry()

# Register functions
registry.register(
        "check_order",
        check_order_status,
        ["order_id"],
        "Check the status of an order by its ID"
)

registry.register(
        "place_order",
        register_order,
        ["customer_name", "phone_number"],
        "Register a new order for a customer with optional product IDs"
)

registry.register(
        "check_product_inventory",
        check_inventory,
        [],  # Both parameters are optional
        "Check inventory for a product by ID or name, or get all inventory"
)

# Update schema on startup
try:
    registry.update_schema()
except Exception as e:
    logger.error(f"Failed to initialize database schema: {str(e)}")


def validate_and_convert_entities(extracted_entities: Dict[str, Any], required_params: Dict[str, type]) -> Dict[
    str, Any]:
    """
    Validate and convert extracted entities to match the required parameter types.

    Args:
        extracted_entities: Extracted entities from the user query
        required_params: Dictionary of required parameters and their types

    Returns:
        Dictionary of validated and converted entities
    """
    validated_entities = {}
    for param, param_type in required_params.items():
        if param in extracted_entities:
            try:
                # Convert the entity to the required type
                validated_entities[param] = param_type(extracted_entities[param])
            except (ValueError, TypeError):
                logger.warning(f"Failed to convert {param} to {param_type}")
                raise ValueError(f"Invalid type for {param}. Expected {param_type}.")
        else:
            logger.warning(f"Missing required parameter: {param}")
            raise ValueError(f"Missing required parameter: {param}")

    return validated_entities


async def process_intent(
        user_intent: UserIntent,
        model_name: str,
        api_key: SecretStr
) -> Tuple[Optional[FunctionResponse], Optional[str]]:
    """
    Process the user intent and execute functions or generate questions.

    Args:
        user_intent: UserIntent object from preprocessing
        model_name: The name of the LLM model to use
        api_key: API key for the LLM

    Returns:
        Tuple containing:
        - FunctionResponse if a function was executed (or None)
        - Follow-up question string if more information is needed (or None)
    """
    intent = user_intent.primary_intent

    # Get the function details
    func_details = registry.get_function(intent)

    # If we don't have a recognized intent or confidence is too low
    if not func_details or user_intent.confidence < 0.7:
        # Use LLM to determine the most likely intent and follow-up question
        return await _generate_clarification_question(user_intent, model_name, api_key)

    # Check if we have all required parameters
    required_params = func_details["required_params"]
    missing_params = user_intent.missing_entities(required_params)

    # If we're missing required parameters
    if missing_params:
        # Generate a follow-up question for the missing parameters
        return await _generate_missing_params_question(user_intent, missing_params, model_name, api_key)

    try:
        # Validate and convert extracted entities
        validated_entities = validate_and_convert_entities(
                user_intent.extracted_entities,
                {param: int if param.endswith("_id") else str for param in required_params}
        )

        # Execute the function with validated entities
        return await _execute_function(intent, validated_entities), None

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return None, f"I need more information. {str(e)}"


async def _execute_function(function_name: str, params: Dict[str, Any]) -> FunctionResponse:
    """Execute a function with the given parameters."""
    func_details = registry.get_function(function_name)
    if not func_details:
        return FunctionResponse(
                function_name=function_name,
                arguments=params,
                result={"error": f"Function {function_name} not found"},
                success=False,
                error_message=f"Function {function_name} not found"
        )

    try:
        # Filter params to only include those that match the function signature
        func = func_details["func"]
        sig = inspect.signature(func)
        valid_params = {}

        for param_name, param in sig.parameters.items():
            if param_name in params:
                valid_params[param_name] = params[param_name]

        # Execute the function
        result = func(**valid_params)

        return FunctionResponse(
                function_name=function_name,
                arguments=valid_params,
                result=result,
                success="error" not in result
        )
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {str(e)}")
        return FunctionResponse(
                function_name=function_name,
                arguments=params,
                result={"error": str(e)},
                success=False,
                error_message=str(e)
        )


async def _generate_clarification_question(
        user_intent: UserIntent,
        model_name: str,
        api_key: SecretStr
) -> Tuple[None, str]:
    """Generate a clarification question when intent is unclear"""
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
                model=model_name,
                base_url="https://api.avalai.ir/v1",
                api_key=api_key,
                max_retries=2
        )

        # List available functions for the LLM
        available_functions = []
        for name, details in registry.get_all_functions().items():
            required = ", ".join(details["required_params"])
            available_functions.append(f"{name} (Required: {required}): {details['description']}")

        functions_str = "\n".join(available_functions)

        # Create the prompt
        prompt = f"""
        The user's intent was unclear. Based on their query summary: "{user_intent.summary}"

        Available functions:
        {functions_str}

        1. Determine which function would best address their needs
        2. Create a helpful, conversational question asking for the specific information needed

        Your response should be conversational and only include the question to ask the user.
        """

        # Get the response
        human_message = HumanMessage(content=prompt)
        response = await llm.ainvoke([human_message])

        return None, response.content.strip()
    except Exception as e:
        logger.error(f"Error generating clarification question: {str(e)}")
        return None, "I need more information to help you. Could you please clarify what you're looking for?"


async def _generate_missing_params_question(
        user_intent: UserIntent,
        missing_params: List[str],
        model_name: str,
        api_key: SecretStr
) -> Tuple[None, str]:
    """Generate a question to ask for missing parameters"""
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
                model=model_name,
                base_url="https://api.avalai.ir/v1",
                api_key=api_key,
                max_retries=2
        )

        # Get function details
        func_details = registry.get_function(user_intent.primary_intent)
        function_desc = func_details["description"] if func_details else "this operation"

        # Create the prompt
        prompt = f"""
        The user wants to {function_desc}. Based on their query: "{user_intent.summary}"

        I've already extracted these details:
        {user_intent.extracted_entities}

        But I still need the following information:
        {", ".join(missing_params)}

        Create a helpful, conversational question asking for specifically these missing pieces of information.
        Your response should be conversational and only include the question to ask the user.
        """

        # Get the response
        human_message = HumanMessage(content=prompt)
        response = await llm.ainvoke([human_message])

        return None, response.content.strip()
    except Exception as e:
        logger.error(f"Error generating missing params question: {str(e)}")
        return None, f"I need a few more details. Could you please provide your {', '.join(missing_params)}?"


# Export essential functions
def get_system_knowledge():
    return registry.get_system_knowledge()
