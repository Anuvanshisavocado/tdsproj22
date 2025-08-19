import os
import tempfile
import json
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import python_code_interpreter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Env Setup ---
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
AIPIPE_BASE_URL = "https://aipipe.org/openai/v1"

if not AIPIPE_TOKEN:
    logger.error("AIPIPE_TOKEN environment variable not set.")

# LangChain LLM config with AIPipe, use a model that supports tool-calling (e.g., gpt-4o, gpt-3.5-turbo)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=AIPIPE_TOKEN,
    base_url=AIPIPE_BASE_URL
)

tools = [python_code_interpreter]

# Strict system prompt to enforce JSON output and base64 charts
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a world-class data analyst. Your task is to answer the user's questions by writing and executing Python code.

Workflow:
1. Analyze questions and data files.
2. Plan step-by-step to produce all required answers.
3. Use the 'python_code_interpreter' tool to execute Python.
4. Your FINAL output must be a single valid JSON object containing all the required keys.
5. No explanations, only the JSON.
6. For chart/image data, base64 PNG strings must start with "data:image/png;base64," and be under 100kB.

Required keys: total_sales (number), top_region (string), day_sales_correlation (number),
bar_chart (base64 PNG string), median_sales (number), total_sales_tax (number), cumulative_sales_chart (base64 PNG string).

If a value cannot be computed, set null or "" (empty string for images).
"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Fallback JSON on error or missing keys
fallback_result = {
    "total_sales": None,
    "top_region": None,
    "day_sales_correlation": None,
    "bar_chart": "",
    "median_sales": None,
    "total_sales_tax": None,
    "cumulative_sales_chart": ""
}

@app.get("/")
@app.get("/api/")
async def health_check():
    return {"status": "ok"}

@app.post("/")
@app.post("/api/")
async def analyze(
    questions_file: UploadFile = File(...),
    other_files: List[UploadFile] = File(default=[])
):
    if not AIPIPE_TOKEN:
        return JSONResponse(status_code=500, content={"detail": "Server not configured with AIPIPE_TOKEN."})

    files = [questions_file] + other_files

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            data_filenames = [f.filename for f in other_files]
            # Save all files to temp dir
            for f in files:
                filepath = os.path.join(temp_dir, f.filename)
                content = await f.read()
                with open(filepath, "wb") as out_file:
                    out_file.write(content)
                await f.seek(0)

            questions_content = (await questions_file.read()).decode("utf-8")
            prompt_text = f"Questions:\n{questions_content}\n\nAvailable files: {data_filenames}"

            logger.info(f"Prompt:\n{prompt_text}")

            response = await agent_executor.ainvoke({"input": prompt_text})

            final_output = response.get("output") if isinstance(response, dict) else response

            if not final_output or not isinstance(final_output, str):
                logger.error(f"Invalid or empty output: {final_output}")
                return JSONResponse(content=fallback_result)

            logger.info(f"Final agent output:\n{final_output}")

            try:
                json_output = json.loads(final_output)
                # Ensure all required keys exist
                for key in fallback_result:
                    if key not in json_output:
                        json_output[key] = fallback_result[key]
                return JSONResponse(content=json_output)
            except json.JSONDecodeError:
                logger.error(f"Output is not valid JSON: {final_output}")
                return JSONResponse(content=fallback_result)

        except Exception as e:
            logger.error(f"Exception during request handling: {e}", exc_info=True)
            return JSONResponse(content={"detail": f"Internal server error: {e}"}, status_code=500)
        finally:
            os.chdir(original_cwd)
