# Standard library imports
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# LangChain Memory imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from postprocess import postprocess_results
# Import our custom modules
from preprocess import preprocess_query
from process import generate_system_knowledge, identify_missing_params, tools

# Set up logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

# Get API key from environment variables
API_KEY = os.getenv("API_KEY")
model_name = "gpt-4o-mini"

# Define the Query model
from pydantic import BaseModel


class Query(BaseModel):
    prompt: str
    user_id: str = None
    clear_memory: bool = False


# Memory manager using LangChain's ConversationBufferMemory
class MemoryManager:
    def __init__(self):
        self.memories = {}

    def get_memory(self, user_id):
        """Get or create memory for a user"""
        if user_id not in self.memories:
            history = ChatMessageHistory()
            # Add system message with dynamic knowledge
            system_message = SystemMessage(content=generate_system_knowledge())
            history.add_message(system_message)
            self.memories[user_id] = history
        return self.memories[user_id]

    def clear_memory(self, user_id):
        """Clear memory for a user but keep system message"""
        if user_id in self.memories:
            system_content = generate_system_knowledge()
            history = ChatMessageHistory()
            history.add_message(SystemMessage(content=system_content))
            self.memories[user_id] = history
            return True
        return False

    def add_human_message(self, user_id, content):
        """Add a human message to memory"""
        memory = self.get_memory(user_id)
        memory.add_message(HumanMessage(content=content))

    def add_ai_message(self, user_id, content):
        """Add an AI message to memory"""
        memory = self.get_memory(user_id)
        memory.add_message(AIMessage(content=content))

    def get_messages(self, user_id):
        """Get all messages for a user"""
        memory = self.get_memory(user_id)
        return memory.messages

    def prune_memory(self, user_id, max_messages=10):
        """Prune memory to keep a maximum number of messages"""
        memory = self.get_memory(user_id)
        messages = memory.messages

        if len(messages) > max_messages:
            # Keep system message and last (max_messages-1) exchanges
            new_history = ChatMessageHistory()
            # Find the system message
            system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]

            if system_messages:
                # Add system message
                new_history.add_message(system_messages[0])

            # Add most recent messages up to the limit
            remaining_slots = max_messages - len(system_messages)
            for msg in messages[-remaining_slots:]:
                if not isinstance(msg, SystemMessage):
                    new_history.add_message(msg)

            self.memories[user_id] = new_history


# Initialize memory manager
memory_manager = MemoryManager()


@app.get("/user_id")
async def get_user_id():
    """Generate a random user ID"""
    global users_count
    user_id = users_count
    users_count += 1
    # Initialize memory for this user
    memory_manager.get_memory(user_id)
    return {"user_id": user_id}


@app.post("/process")
async def process_query(query: Query):
    """Process the user's query using LLM with tools"""
    logger.info(f"Received query: {query.prompt}")

    # Handle user_id
    user_id = query.user_id if query.user_id else "default"

    # Clear memory if requested
    if query.clear_memory:
        memory_manager.clear_memory(user_id)
        logger.info(f"Cleared memory for user {user_id}")
        return {
                "response": "Memory has been cleared. How can I help you?",
                "function_calls": []
        }

    try:
        # Add the user's message to memory
        memory_manager.add_human_message(user_id, query.prompt)

        # Get the full conversation history
        messages = memory_manager.get_messages(user_id)

        # Step 1: Preprocess - Send the query to the LLM to get tool calls
        _, llm_output, function_responses = await preprocess_query(
                messages=messages,
                api_key=API_KEY,
                model_name=model_name
        )

        # Handle case where LLM fails to respond
        if not llm_output:
            logger.warning("No valid response from LLM after all attempts")

            # Generate a generic fallback response
            response_text = "I'm here to assist with specific functions. Please try again with details for one of my available tools."

            # Add the response to memory
            memory_manager.add_ai_message(user_id, response_text)
            memory_manager.prune_memory(user_id)

            return {
                    "response": response_text,
                    "function_calls": []
            }

        # If the LLM didn't make any tool calls, determine what's missing
        if not hasattr(llm_output, 'tool_calls') or not llm_output.tool_calls:
            logger.info("LLM provided direct response without tool calls - analyzing query")

            # Try to determine the most relevant function based on the query
            from langchain_openai import ChatOpenAI

            # Create a temporary prompt to analyze the query
            analysis_prompt = f"""
            You are a query analyzer. Based on this user query: "{query.prompt}"
            Which of these functions would be most appropriate to call?

            {generate_system_knowledge()}

            Respond with just the function name, or "none" if no function is appropriate.
            """

            try:
                # Create a simpler LLM call to analyze the query
                analyzer_llm = ChatOpenAI(
                        model=model_name,
                        base_url="https://api.avalai.ir/v1",
                        api_key=API_KEY,
                        max_retries=2
                )

                # Get the analyzer's response
                analysis_result = analyzer_llm.invoke(analysis_prompt)
                function_guess = analysis_result.content.strip().lower()

                # Check if a valid function was identified
                if function_guess in [tool.__name__.lower() for tool in tools()]:
                    # Identify what parameters would be needed
                    missing_params = identify_missing_params(function_guess, {})

                    response_text = f"I need more information to help you. For the {function_guess} function, please provide: {', '.join(missing_params)}"
                else:
                    # If no relevant function was identified
                    response_text = "I can only help with specific functions. Please provide details for one of my available tools."
            except Exception as e:
                logger.error(f"Error in query analysis: {str(e)}")
                response_text = "I can only help with specific functions. Please provide details for one of my available tools."

            # Add the response to memory
            memory_manager.add_ai_message(user_id, response_text)
            memory_manager.prune_memory(user_id)

            return {
                    "response": response_text,
                    "function_calls": []
            }

        # Step 2: Process - Generate the final response
        response_content = await postprocess_results(
                messages,
                query.prompt,
                function_responses,
                API_KEY,
                model_name
        )

        # Add the response to memory
        memory_manager.add_ai_message(user_id, response_content)

        # Prune memory if needed
        memory_manager.prune_memory(user_id)

        # Return the enhanced response to the user
        return {
                "response": response_content,
                "function_calls": [resp.model_dump() for resp in function_responses]
        }

    except Exception as e:
        # Catch-all error handler
        logger.error(f"Unhandled error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    users_count = 1
    logger.info("Starting server")
    uvicorn.run(app, host="127.0.0.1", port=8000)