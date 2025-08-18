import io
import base64
import sys
import os
from contextlib import redirect_stdout
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
from langchain.tools import tool

@tool
def python_code_interpreter(code: str) -> str:
    """
    Executes Python code in the current directory and returns the output.
    This tool is a full Python environment. It can handle web scraping (requests, beautifulsoup4),
    data analysis (pandas, numpy), network analysis (networkx), and plotting (matplotlib).
    Any uploaded files (like 'data.csv') are available in the current working directory.
    When you need to generate a plot, the plot will be automatically captured,
    encoded to a base64 data URI, and returned.
    To provide the final answer, print a single valid JSON string to stdout.
    """
    try:
        # Use a dictionary for local variables to capture exec results
        local_vars = {}
        
        # Redirect stdout to capture print statements
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(code, globals(), local_vars)

        # Check if a plot was generated
        if plt.get_fignums():
            fig = plt.gcf()
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            plt.close(fig) # Close the figure to free up memory
            img_buffer.seek(0)
            
            # Check image size before encoding
            b64_string = base64.b64encode(img_buffer.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{b64_string}"
            
            if len(data_uri) > 100000:
                return "Error: The generated plot image is too large (>100kb)."
            return data_uri

        # If no plot, return the captured stdout (for final JSON answer)
        output = buffer.getvalue()
        if output:
            return output.strip()
            
        # If no stdout, try to return the last computed value
        if local_vars:
            # Return the value of the last variable defined in the exec scope
            return str(list(local_vars.values())[-1])

        return "Code executed successfully with no output."

    except Exception as e:
        # Return a formatted error string to the LLM to allow for self-correction
        return f"Execution failed with error: {type(e).__name__}('{e}')"