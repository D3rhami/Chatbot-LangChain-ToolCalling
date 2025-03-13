# Standard library imports
import logging
import os
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
# FastAPI imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr

from memory_manager import MemoryManager
from postprocess import postprocess_result
# Import our modules
from preprocess import preprocess_query
from process import process_intent

load_dotenv()

# Set up logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
API_KEY = SecretStr(os.getenv("API_KEY"))
MODEL_NAME = "gpt-4o-mini"  # Default model


# Query model
class Query(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    clear_memory: bool = False


# Initialize memory manager
memory_manager = MemoryManager()
users_count = 1


@app.get("/user_id")
async def get_user_id():
    """Generate a user ID."""
    global users_count
    user_id = str(users_count)
    users_count += 1
    # Initialize memory for this user
    memory_manager.get_memory(user_id)
    return {"user_id": user_id}


@app.post("/process")
async def process_query(query: Query):
    """Process the user's query using the improved pipeline."""
    logger.info(f"Received query: {query.prompt}")

    # Handle user_id
    user_id = query.user_id if query.user_id else "default"

    # Clear memory if requested
    if query.clear_memory:
        memory_manager.clear_memory(user_id)
        logger.info(f"Cleared memory for user {user_id}")
        return {"response": "Memory has been cleared. How can I help you?"}

    try:
        # Add the user's message to memory
        memory_manager.add_human_message(user_id, query.prompt)

        # Step 1: Preprocess - Understand user intent and extract entities
        user_intent = await preprocess_query(
                query=query.prompt,
                model_name=MODEL_NAME,
                api_key=API_KEY
        )
        logger.info(f"Preprocessing results: Intent={user_intent.primary_intent}, Confidence={user_intent.confidence}")

        # Step 2: Process - Execute function or generate follow-up question
        function_response, follow_up_question = await process_intent(
                user_intent=user_intent,
                model_name=MODEL_NAME,
                api_key=API_KEY
        )

        if function_response:
            logger.info(f"Executed function: {function_response.function_name}")
        elif follow_up_question:
            logger.info(f"Generated follow-up question: {follow_up_question}")

        # Step 3: Postprocess - Generate human-friendly response
        final_response = await postprocess_result(
                query=query.prompt,
                function_response=function_response,
                follow_up_question=follow_up_question,
                model_name=MODEL_NAME,
                api_key=API_KEY
        )

        # Add the response to memory
        memory_manager.add_ai_message(user_id, final_response)

        # Prune memory if needed
        memory_manager.prune_memory(user_id)

        return {"response": final_response}

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        error_response = await postprocess_result(
                query=query.prompt,
                function_response=None,
                follow_up_question=None,
                model_name=MODEL_NAME,
                api_key=API_KEY
        )
        return {"response": error_response}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server")
    uvicorn.run(app, host="127.0.0.1", port=8000)
