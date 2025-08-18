import io
import base64
import sys
from contextlib import redirect_stdout
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for servers
import matplotlib.pyplot as plt
from langchain.tools import tool

@tool
def python_code_interpreter(code: str) -> str:
    """
    Executes Python code in the current directory and returns the output.
    This tool is a full Python environment. It can handle web scraping (requests, beautifulsoup4),
    data analysis (pandas, numpy), network analysis (networkx), and plotting (matplotlib).
    Any uploaded files (like 'sample-sales.csv' or 'edges.csv') are available in the current
    working directory. When plotting, the plot is automatically captured and returned as a
    base64 data URI. To provide the final answer, print a single valid JSON string to stdout.
    """
    try:
        local_vars = {}
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(code, globals(), local_vars)

        if plt.get_fignums():
            fig = plt.gcf()
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            plt.close(fig)
            img_buffer.seek(0)
            
            b64_string = base64.b64encode(img_buffer.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{b64_string}"
            
            if len(data_uri) > 100000:
                return "Error: The generated plot image is too large (>100kb)."
            return data_uri

        output = buffer.getvalue().strip()
        if output:
            return output
            
        if local_vars:
            return str(list(local_vars.values())[-1])

        return "Code executed successfully with no output."

    except Exception as e:
        return f"Execution failed with error: {type(e).__name__}('{e}')"
