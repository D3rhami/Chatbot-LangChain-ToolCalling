# Standard library imports
import os
import random
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import our custom modules
from preprocess import preprocess_query
from process import execute_function_call, FunctionResponse
from postprocess import postprocess_results, generate_response_from_function_results

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

# Define the Query model (moved to server.py for simplicity)
from pydantic import BaseModel


class Query(BaseModel):
    prompt: str


@app.get("/user_id")
async def get_user_id():
    """Generate a random user ID"""
    user_id = random.randint(1000, 9999)
    return {"user_id": user_id}


@app.post("/process")
async def process_query(query: Query):
    """Process the user's query using LLM with tools"""
    logger.info(f"Received query: {query.prompt}")
    try:
        # Step 1: Preprocess - Send the query to the LLM to get tool calls
        messages, llm_output, function_responses = await preprocess_query(query.prompt, API_KEY, model_name)

        if not llm_output:
            logger.warning("No valid response from LLM after all attempts")
            return {
                    "response": f"I'm sorry, I couldn't process your query '{query.prompt}' at this time.",
                    "function_calls": []
            }

        # If the LLM didn't make any tool calls, return its response directly
        if not hasattr(llm_output, 'tool_calls') or not llm_output.tool_calls:
            logger.info("LLM provided direct response without tool calls")
            return {
                    "response": llm_output.content,
                    "function_calls": []
            }

        # Step 2: Process - Generate the final response
        response_content = await postprocess_results(messages, query.prompt, function_responses, API_KEY, model_name)

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

    logger.info("Starting server")
    uvicorn.run(app, host="127.0.0.1", port=8000)