import os
import tempfile
import json
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import List, Annotated

# Make sure to install these: pip install langchain langchain-openai
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import python_code_interpreter

# --- IMPORTANT: Set your OpenAI API Key ---
# Best practice is to set this as an environment variable
# os.environ["OPENAI_API_KEY"] = "sk-..."

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Agent Setup ---
# Use a powerful model capable of complex reasoning and tool use (e.g., gpt-4o, gpt-4-turbo)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [python_code_interpreter]

# This is the master prompt that tells the agent how to behave
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world-class data analyst. Your task is to answer questions based on provided files and instructions. You must use the 'python_code_interpreter' tool to perform all of your work. The user's files are available in the tool's current working directory. Your final answer MUST be a single, valid JSON array or object printed to stdout by the tool, as requested in the user's prompt."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create and configure the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

@app.post("/api/")
async def analyze(
    questions_file: Annotated[UploadFile, File()],
    files: List[UploadFile] = File([]) # Use an empty list as default for optional files
):
    # Use a temporary directory to securely handle file uploads
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir) # Change CWD for the agent's tool

        try:
            # Save all uploaded files to the temporary directory
            file_names = []
            all_files = [questions_file] + files
            for uploaded_file in all_files:
                file_path = os.path.join(temp_dir, uploaded_file.filename)
                with open(file_path, "wb") as f:
                    f.write(await uploaded_file.read())
                if uploaded_file.filename != "questions.txt":
                    file_names.append(uploaded_file.filename)
            
            questions_content = (await questions_file.read()).decode("utf-8")
            logger.info(f"Received questions:\n{questions_content}")

            # Create a detailed input prompt for the agent
            input_prompt = (
                f"Please answer the following questions:\n\n---\n{questions_content}\n---\n\n"
                f"The following data files are available in the current directory for your analysis: {file_names}\n\n"
                "Generate the final answer in the precise JSON format requested by the questions."
            )

            # Invoke the agent and get the response
            response = await agent_executor.ainvoke({"input": input_prompt})
            
            # The agent's final output should be the JSON string from the tool
            final_answer_str = response.get("output", "{}")
            logger.info(f"Agent output: {final_answer_str}")

            # Parse the string into a JSON object/array before returning
            return json.loads(final_answer_str)

        except json.JSONDecodeError:
            # Handle cases where the LLM output is not valid JSON
            error_msg = f"Agent returned a non-JSON output: {final_answer_str}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
        finally:
            # IMPORTANT: Restore the original working directory
            os.chdir(original_cwd)