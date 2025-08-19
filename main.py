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

# --- Environment Setup ---
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
AIPIPE_BASE_URL = "https://aipipe.org/openai/v1"

if not AIPIPE_TOKEN:
    logger.error("AIPIPE_TOKEN environment variable not set.")

# Configure the LLM with AIPipe API
llm = ChatOpenAI(
    model="gpt-4o",  # Use gpt-4o or gpt-3.5-turbo as appropriate
    temperature=0,
    api_key=AIPIPE_TOKEN,
    base_url=AIPIPE_BASE_URL,
    max_tokens=1500 
)

tools = [python_code_interpreter]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a world-class data analyst. Your primary goal is to answer the user's questions by writing and executing Python code.

Workflow:
1. Analyze questions and data files.
2. Plan step-by-step how to answer all required parts.
3. Use the 'python_code_interpreter' tool as needed.
4. FINAL OUTPUT must be a single valid JSON object containing all required keys:
   - total_sales (number)
   - top_region (string)
   - day_sales_correlation (number)
   - bar_chart (base64 PNG string beginning with "data:image/png;base64,"; under 100kB)
   - median_sales (number)
   - total_sales_tax (number)
   - cumulative_sales_chart (base64 PNG string beginning with "data:image/png;base64,"; under 100kB)
5. Do NOT output explanations or other text. Only JSON.
6. If a value cannot be computed, use null (numbers) or empty string (images).

"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Fallback JSON in case of failure
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

    all_files = [questions_file] + other_files

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            data_file_names = [f.filename for f in other_files]
            # Save uploaded files to temp directory
            for f in all_files:
                file_path = os.path.join(temp_dir, f.filename)
                content = await f.read()
                with open(file_path, "wb") as out_file:
                    out_file.write(content)
                await f.seek(0)

            questions_content = (await questions_file.read()).decode("utf-8")

            input_prompt = f"Questions:\n{questions_content}\n\nAvailable data files: {data_file_names}"

            logger.info(f"Agent prompt:\n{input_prompt}")

            # Call the LangChain agent with prompt
            response = await agent_executor.ainvoke({"input": input_prompt})

            final_answer_str = response.get("output") if isinstance(response, dict) else response

            if not final_answer_str or not isinstance(final_answer_str, str):
                logger.error(f"Invalid or empty agent output: {final_answer_str}")
                return JSONResponse(content=fallback_result)

            logger.info(f"Agent output:\n{final_answer_str}")

            try:
                final_content = json.loads(final_answer_str)
                # Ensure all required keys present; fill missing with fallback values
                for key, default_val in fallback_result.items():
                    if key not in final_content:
                        final_content[key] = default_val
                return JSONResponse(content=final_content)
            except json.JSONDecodeError:
                logger.error(f"Agent output is not valid JSON: {final_answer_str}")
                return JSONResponse(content=fallback_result)

        except Exception as e:
            logger.error(f"Internal server error: {e}", exc_info=True)
            return JSONResponse(content={"detail": f"Internal server error: {str(e)}"}, status_code=500)
        finally:
            os.chdir(original_cwd)

