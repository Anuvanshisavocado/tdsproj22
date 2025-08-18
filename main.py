import os
import tempfile
import json
import logging
from fastapi import FastAPI, UploadFile, Request, HTTPException, Response
from starlette.responses import JSONResponse

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import python_code_interpreter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Agent Setup ---
aipipe_token = os.environ.get("AIPIPE_TOKEN")
aipipe_base_url = "https://aipipe.org/openai/v1"

if not aipipe_token:
    logger.error("AIPIPE_TOKEN environment variable not set.")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=aipipe_token,
    base_url=aipipe_base_url
)

tools = [python_code_interpreter]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world-class data analyst. Your task is to answer questions based on provided files and instructions. You must use the 'python_code_interpreter' tool to perform all of your work. The user's files are available in the tool's current working directory. Your final answer MUST be a single, valid JSON array or object printed to stdout by the tool, as requested in the user's prompt."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


@app.get("/")
@app.get("/api/")
async def health_check():
    """Provides a simple health check endpoint for GET requests."""
    return {"status": "ok"}


@app.post("/")
@app.post("/api/")
async def analyze(request: Request):
    """
    Handles POST requests. Distinguishes between real file uploads
    and probe requests by checking for 'questions.txt'.
    """
    if not aipipe_token:
        raise HTTPException(status_code=500, detail="Server is not configured with an AIPIPE_TOKEN.")

    form_data = await request.form()
    files = [value for value in form_data.values() if isinstance(value, UploadFile)]
    
    questions_file = None
    data_files = []
    for file in files:
        if hasattr(file, 'filename') and file.filename == 'questions.txt':
            questions_file = file
        elif hasattr(file, 'filename'):
            data_files.append(file)
    
    # FINAL FIX: If questions.txt is missing, return a 204 No Content status.
    if not questions_file:
        logger.info("Received POST without 'questions.txt'. Assuming probe. Returning 204 No Content.")
        return Response(status_code=204)

    # --- From here, the logic proceeds only if it's the real data request ---
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            all_uploaded_files = [questions_file] + data_files
            data_file_names = [f.filename for f in data_files]

            for uploaded_file in all_uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.filename)
                content = await uploaded_file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                await uploaded_file.seek(0)

            questions_content = (await questions_file.read()).decode("utf-8")
            logger.info(f"Received questions:\n{questions_content}")
            logger.info(f"Received data files: {data_file_names}")

            input_prompt = (
                f"Please answer the following questions:\n\n---\n{questions_content}\n---\n\n"
                f"The following data files are available in the current directory for your analysis: {data_file_names}\n\n"
                "Generate the final answer in the precise JSON format requested by the questions."
            )

            response = await agent_executor.ainvoke({"input": input_prompt})
            final_answer_str = response.get("output", "{}")
            logger.info(f"Agent output: {final_answer_str}")
            
            return JSONResponse(content=json.loads(final_answer_str))

        except json.JSONDecodeError:
            error_msg = f"Agent returned a non-JSON output: {final_answer_str}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
        finally:
            os.chdir(original_cwd)
