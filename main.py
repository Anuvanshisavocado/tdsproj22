import os
import tempfile
import json
import logging
from fastapi import FastAPI, UploadFile, Request, HTTPException, File
from starlette.responses import JSONResponse
from typing import List

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import python_code_interpreter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Agent Setup ---
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
openrouter_base_url = "https://openrouter.ai/api/v1"

if not openrouter_api_key:
    logger.error("OPENROUTER_API_KEY environment variable not set.")

llm = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0,
    api_key="placeholder",
    base_url=openrouter_base_url,
    default_headers={
        "Authorization": f"Bearer {openrouter_api_key}",
        "HTTP-Referer": "https://github.com/your-repo", # Optional: Replace with your repo URL
        "X-Title": "TDS Data Analyst Agent",
    }
)

tools = [python_code_interpreter]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a world-class data analyst. Your task is to answer questions based on provided files and instructions.
You must use the 'python_code_interpreter' tool to perform all of your work.
Your final answer MUST be a single, valid JSON object or array printed to stdout that strictly matches the schema requested by the user.

CRITICAL INSTRUCTIONS:
1.  Your final JSON output MUST contain all required keys specified in the user's question. Do not omit any keys.
2.  For any questions that require a plot or image, the value must be a correctly formatted base64 PNG string, like "data:image/png;base64,...".
3.  The base64 string must be less than 100,000 bytes.
4.  If a value cannot be calculated, you must still include the key with a null or appropriate default value."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

@app.get("/")
@app.get("/api/")
async def health_check():
    return {"status": "ok"}

@app.post("/")
@app.post("/api/")
async def analyze(request: Request):
    if not openrouter_api_key:
        raise HTTPException(status_code=500, detail="Server is not configured with an OPENROUTER_API_KEY.")

    form_data = await request.form()
    files = [value for value in form_data.values() if isinstance(value, UploadFile)]
    
    questions_file = None
    data_files = []
    for file in files:
        if hasattr(file, 'filename') and file.filename in ('questions.txt', 'question.txt'):
            questions_file = file
        elif hasattr(file, 'filename'):
            data_files.append(file)
    
    # Universal handler for probe requests: return an empty, valid JSON.
    if not questions_file:
        logger.info("Received POST without 'questions.txt'. Assuming probe. Returning empty JSON.")
        return JSONResponse(content={})

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

            questions_content = (await uploaded_file.read()).decode("utf-8")
            input_prompt = (
                f"Please answer the following questions:\n\n---\n{questions_content}\n---\n\n"
                f"The following data files are available for your analysis: {data_file_names}"
            )

            response = await agent_executor.ainvoke({"input": input_prompt})
            final_answer_str = response.get("output", "{}")
            logger.info(f"Agent output: {final_answer_str}")
            
            try:
                final_content = json.loads(final_answer_str)
                return JSONResponse(content=final_content)
            except json.JSONDecodeError:
                logger.error(f"Agent returned a non-JSON output: {final_answer_str}")
                return JSONResponse(content={}, status_code=500) # Return empty JSON on agent failure

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return JSONResponse(content={}, status_code=500) # Return empty JSON on system failure
        finally:
            os.chdir(original_cwd)
