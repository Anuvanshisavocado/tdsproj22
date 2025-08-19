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

# --- Agent Setup ---

aipipe_token = os.environ.get("AIPIPE_TOKEN")
aipipe_base_url = "https://aipipe.org/openai/v1"

if not aipipe_token:
    logger.error("AIPIPE_TOKEN environment variable not set.")

llm = ChatOpenAI(
    model="gpt-4o",  # or your preferred AIPipe-supported model
    temperature=0,
    api_key=aipipe_token,
    base_url=aipipe_base_url,
)

tools = [python_code_interpreter]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a world-class data analyst. Your primary goal is to answer the user's questions by writing and executing Python code.


**Your workflow is as follows:**
1. Analyze the Request: Understand the user's questions and the data files provided.
2. Plan Your Steps: Think step-by-step to produce all parts of the answer.
3. Execute Code: Use the 'python_code_interpreter' tool repeatedly as needed.
4. Final Output: Your absolute final action must be to output a single, valid JSON object or array containing the complete response. Do NOT output any other text or notes.


**IMPORTANT:**
- Your final JSON must contain all required keys specified in the user's questions.
- If any value cannot be computed, include the key with null or appropriate default.
- For any images/charts, base64 strings must begin with "data:image/png;base64," and be under 100k bytes."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

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
        return JSONResponse(status_code=500, content={"detail": "Server is not configured with an AIPIPE_TOKEN."})

    data_files = other_files
    all_files = [questions_file] + data_files

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            data_file_names = [f.filename for f in data_files]
            for uploaded_file in all_files:
                file_path = os.path.join(temp_dir, uploaded_file.filename)
                content = await uploaded_file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                await uploaded_file.seek(0)

            questions_content = (await questions_file.read()).decode("utf-8")
            input_prompt = (
                f"Questions:\n{questions_content}\n\n"
                f"Available data files: {data_file_names}"
            )

            logger.info(f"Running agent with prompt:\n{input_prompt}")

            response = await agent_executor.ainvoke({"input": input_prompt})

            final_answer_str = response.get("output") if isinstance(response, dict) else response

            if not final_answer_str or not isinstance(final_answer_str, str):
                logger.error(f"Agent returned invalid output: {final_answer_str}")
                return JSONResponse(content=fallback_result)

            logger.info(f"Agent output: {final_answer_str}")

            try:
                final_content = json.loads(final_answer_str)
                return JSONResponse(content=final_content)
            except json.JSONDecodeError:
                logger.error(f"Agent returned invalid JSON: {final_answer_str}")
                return JSONResponse(content=fallback_result)

        except Exception as e:
            logger.error(f"Unexpected error in analyze endpoint: {e}", exc_info=True)
            return JSONResponse(content={"detail": f"Internal server error: {e}"}, status_code=500)
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
