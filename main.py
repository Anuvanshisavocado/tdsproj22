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

import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Agent Setup ---
aipipe_token = os.environ.get("AIPIPE_TOKEN")
aipipe_base_url = "https://aipipe.org/openai/v1"

if not aipipe_token:
    logger.error("AIPIPE_TOKEN environment variable not set.")

llm = ChatOpenAI(
    model="gpt-4o",  # or another supported model, e.g. gpt-3.5-turbo
    temperature=0,
    api_key=aipipe_token,
    base_url=aipipe_base_url,
)

tools = [python_code_interpreter]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a world-class data analyst. Your primary goal is to answer user's questions by writing and executing Python code.

**Workflow**
1. Analyze the user's questions and data files.
2. Plan step-by-step how to answer all required parts.
3. Use python_code_interpreter tool multiple times if needed.
4. FINAL OUTPUT must be a single valid JSON object with ALL required keys.
5. Output no explanations, only the final JSON.
6. All chart/image fields must be base64 PNG strings starting with "data:image/png;base64," and < 100k bytes.

If any key's value cannot be computed, set to null or empty string (for images).

"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Fallback schema with null defaults and empty strings for charts
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
    if not aipipe_token:
        return JSONResponse(status_code=500, content={"detail": "Server missing AIPIPE_TOKEN."})

    all_files = [questions_file] + other_files

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            data_file_names = [f.filename for f in other_files]
            for uf in all_files:
                file_path = os.path.join(temp_dir, uf.filename)
                content = await uf.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                await uf.seek(0)

            questions_content = (await questions_file.read()).decode("utf-8")
            input_prompt = f"Questions:\n{questions_content}\n\nAvailable data files: {data_file_names}"

            logger.info(f"Running agent with prompt:\n{input_prompt}")
            response = await agent_executor.ainvoke({"input": input_prompt})

            final_answer_str = response.get("output") if isinstance(response, dict) else response
            if not final_answer_str or not isinstance(final_answer_str, str):
                logger.error(f"Invalid or empty output from agent: {final_answer_str}")
                return JSONResponse(content=fallback_result)

            logger.info(f"Agent output: {final_answer_str}")

            try:
                final_content = json.loads(final_answer_str)
                # Optional: Validate keys here and add missing keys with null or "" if needed
                for key in fallback_result.keys():
                    if key not in final_content:
                        final_content[key] = fallback_result[key]
                return JSONResponse(content=final_content)
            except json.JSONDecodeError:
                logger.error(f"Agent output not valid JSON: {final_answer_str}")
                return JSONResponse(content=fallback_result)

        except Exception as e:
            logger.error(f"Internal error: {e}", exc_info=True)
            return JSONResponse(content={"detail": f"Internal Server Error: {str(e)}"}, status_code=500)
        finally:
            os.chdir(original_cwd)
