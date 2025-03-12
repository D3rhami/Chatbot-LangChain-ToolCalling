# Standard library imports
import logging
import inspect
from typing import Any, Dict, List, Callable, Optional
from functools import wraps

# LangChain imports
from langchain_core.tools import tool
from pydantic import BaseModel

# Import our database module
from database import check_order_status, register_order, check_inventory, get_database_schema

# Set up logging
logger = logging.getLogger(__name__)


# Tool registry to dynamically manage available tools
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.tool_list = []
        self.db_schema = None

    def register(self, func):
        """Register a tool function with the registry"""
        tool_name = func.get_name()
        self.tools[tool_name] = func
        # Recreate the tool list whenever a tool is registered
        self.tool_list = list(self.tools.values())
        logger.info(f"Registered tool: {tool_name}")
        return func

    def unregister(self, tool_name):
        """Remove a tool from the registry"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            # Recreate the tool list whenever a tool is unregistered
            self.tool_list = list(self.tools.values())
            logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False

    def get_tool(self, tool_name):
        """Get a tool by name"""
        return self.tools.get(tool_name)

    def get_all_tools(self):
        """Get all registered tools"""
        return self.tool_list

    def update_schema(self):
        """Update the database schema"""
        try:
            self.db_schema = get_database_schema()
            logger.info("Database schema updated successfully")
        except Exception as e:
            logger.error(f"Error updating database schema: {str(e)}")
            self.db_schema = {"error": str(e)}

    def generate_system_knowledge(self):
        """Dynamically generate system knowledge based on registered tools"""
        tool_descriptions = []

        # Update schema if needed
        if self.db_schema is None:
            try:
                self.update_schema()
            except Exception as e:
                logger.error(f"Could not update schema: {str(e)}")

        for tool_name, tool_func in self.tools.items():
            # Extract parameter information from function signature
            sig = inspect.signature(tool_func)
            params = []
            for name, param in sig.parameters.items():
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else "any"
                params.append(f"{name} ({param_type})")

            # Extract docstring
            docstring = tool_func.__doc__ or "No description available"

            # Add to tool descriptions
            tool_descriptions.append(f"{tool_name} - Parameters: {', '.join(params)}\n   {docstring.strip()}")

        system_knowledge = f"""
You are a specialized assistant that can ONLY perform these specific functions:
{chr(10).join(f"{i + 1}. {desc}" for i, desc in enumerate(tool_descriptions))}

You CANNOT answer general questions or engage in conversations outside of these functions.
If a user asks something that doesn't clearly map to one of these functions, identify the most relevant function and ask them to provide the specific information needed to execute it.

Database Schema Information:
{self._format_schema_info()}

Do not provide general information, opinions, or engage in conversations unrelated to these functions.
Always map user requests to one of the available functions, or ask for clarification if needed.
"""
        return system_knowledge

    def _format_schema_info(self):
        """Format database schema information for the system knowledge"""
        if not self.db_schema or "error" in self.db_schema:
            return "Database schema information not available."

        schema_info = []
        for table_name, table_info in self.db_schema.items():
            columns = table_info.get("columns", [])
            column_info = ", ".join([f"{col['name']} ({col['type']})" for col in columns])
            schema_info.append(f"- Table '{table_name}': {column_info}")

        return "\n".join(schema_info)


# Create a global tool registry
registry = ToolRegistry()


# Define our tools (functions that the AI can call)
@registry.register
@tool
def check_order(order_id: int) -> Dict[str, Any]:
    """Check the status of an order.

    Args:
        order_id: The ID of the order to check

    Returns:
        Dict containing order details including status, customer information, and items
    """
    logger.info(f"Executing check_order with order_id={order_id}")
    result = check_order_status(order_id)
    logger.info(f"Result of check_order: {result}")
    return result


@registry.register
@tool
def place_order(customer_name: str, phone_number: str, product_ids: List[int] = None) -> Dict[str, Any]:
    """Register a new order for a customer.

    Args:
        customer_name: The name of the customer
        phone_number: The phone number of the customer
        product_ids: List of product IDs to include in the order (optional)

    Returns:
        Dict containing the result of the order registration
    """
    logger.info(
        f"Executing place_order with customer_name={customer_name}, phone_number={phone_number}, product_ids={product_ids}")
    result = register_order(customer_name, phone_number, product_ids)
    logger.info(f"Result of place_order: {result}")
    return result


@registry.register
@tool
def check_product_inventory(product_id: Optional[int] = None, product_name: Optional[str] = None) -> Dict[str, Any]:
    """Check inventory for a product by ID or name, or get all inventory if no parameters are provided.

    Args:
        product_id: The ID of the product to check (optional)
        product_name: The name of the product to check (optional)

    Returns:
        Dict containing inventory information
    """
    logger.info(f"Executing check_product_inventory with product_id={product_id}, product_name={product_name}")
    result = check_inventory(product_id, product_name)
    logger.info(f"Result of check_product_inventory: {result}")
    return result


class FunctionResponse(BaseModel):
    function_name: str
    arguments: Dict[str, Any]
    result: Any

    def model_dump(self, *args, **kwargs):
        """Compatible method for both Pydantic v1 and v2"""
        if hasattr(super(), "model_dump"):
            return super().model_dump()
        return self


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

    tool = registry.get_tool(tool_name)

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


# Helper function to identify missing parameters for a function
def identify_missing_params(function_name, provided_args):
    """
    Identify what parameters are missing for a function call

    Args:
        function_name: The name of the function
        provided_args: The arguments that were provided

    Returns:
        List of missing parameter names
    """
    this_tool = registry.get_tool(function_name)
    if not this_tool:
        return []

    sig = inspect.signature(this_tool)
    required_params = [
            param.name for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty
    ]

    missing_params = [param for param in required_params if param not in provided_args]
    return missing_params


# Initialize schema on startup
try:
    registry.update_schema()
except Exception as e:
    logger.error(f"Failed to initialize database schema: {str(e)}")

# Properties for external access
tools = registry.get_all_tools
generate_system_knowledge = registry.generate_system_knowledge