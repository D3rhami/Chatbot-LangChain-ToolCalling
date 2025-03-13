"""
This module manages the registration, execution, and response handling of functions
that interact with the database and other systems. It provides the `FunctionRegistry`
class for registering and managing functions, and the `FunctionResponse` class for
handling the results of function execution.
"""

# Standard library imports
import logging
from typing import Any, Callable, Dict, List, Optional

from openai import BaseModel

from database import get_database_schema

# Set up logging
logger = logging.getLogger(__name__)


class FunctionResponse(BaseModel):
    """
    Represents the response from executing a function.

    Attributes:
        function_name (str): The name of the executed function.
        arguments (Dict[str, Any]): The arguments passed to the function.
        result (Any): The result returned by the function.
        success (bool): Whether the function executed successfully.
        error_message (Optional[str]): Error message if the function failed.
    """

    function_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool = True
    error_message: Optional[str] = None

    def model_dump(self, *args, **kwargs):
        """Compatible method for both Pydantic v1 and v2"""
        if hasattr(super(), "model_dump"):
            return super().model_dump(*args, **kwargs)
        return self.model_dump(*args, **kwargs)


class FunctionRegistry:
    """
    Manages the registration and execution of functions.

    Attributes:
        functions (Dict[str, Dict[str, Any]]): Registered functions and their metadata.
        db_schema (Optional[Dict[str, Any]]): Database schema for system knowledge.
    """

    def __init__(self):
        self.functions = {}
        self.db_schema = None

    def register(self, func_name: str, func: Callable, required_params: List[str], description: str):
        """
        Register a function with the registry.

        Args:
            func_name (str): The name of the function.
            func (Callable): The function to register.
            required_params (List[str]): List of required parameters.
            description (str): Description of the function.
        """
        self.functions[func_name] = {
                "func": func,
                "required_params": required_params,
                "description": description
        }
        logger.info(f"Registered function: {func_name}")

    def get_function(self, func_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a function by name.

        Args:
            func_name (str): The name of the function.

        Returns:
            Optional[Dict[str, Any]]: Function details if found, otherwise None.
        """
        return self.functions.get(func_name)

    def get_all_functions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered functions.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of all registered functions.
        """
        return self.functions

    def update_schema(self):
        """Update the database schema."""
        try:
            self.db_schema = get_database_schema()
            logger.info("Database schema updated successfully")
        except Exception as e:
            logger.error(f"Error updating database schema: {str(e)}")
            self.db_schema = {"error": str(e)}

    def get_system_knowledge(self) -> str:
        """
        Generate system knowledge about available functions.

        Returns:
            str: A formatted string describing available functions and database schema.
        """
        if self.db_schema is None:
            try:
                self.update_schema()
            except Exception as e:
                logger.error(f"Failed to update schema: {str(e)}")

        function_descriptions = []
        for name, details in self.functions.items():
            params = ", ".join(details["required_params"])
            function_descriptions.append(f"{name} - Required parameters: {params}\n   {details['description']}")

        return f"""
You are a specialized assistant that can perform these specific functions:
{chr(10).join(f"{i + 1}. {desc}" for i, desc in enumerate(function_descriptions))}

Database Schema Information:
{self._format_schema_info()}
        """

    def _format_schema_info(self) -> str:
        """
        Format database schema for system knowledge.

        Returns:
            str: A formatted string describing the database schema.
        """
        if not self.db_schema or "error" in self.db_schema:
            return "Database schema information not available."

        schema_info = []
        for table_name, table_info in self.db_schema.items():
            columns = table_info.get("columns", [])
            column_info = ", ".join([f"{col['name']} ({col['type']})" for col in columns])
            schema_info.append(f"- Table '{table_name}': {column_info}")

        return "\n".join(schema_info)
