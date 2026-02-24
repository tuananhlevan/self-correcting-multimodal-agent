import subprocess
import tempfile
import os
from core.state import AgentState

def execute_code_node(state: AgentState) -> dict:
    """
    Takes the generated Python script, writes it to a temporary file, 
    executes it in a separate process, and captures the output or errors.
    """
    
    generated_code = state.get("generated_code", "")
    
    # 1. Edge Case Handling: Catch upstream failures
    if "FAILED_BEFORE_EXECUTION" in generated_code:
        return {
            "execution_result": "FAILED",
            "error_history": [generated_code] # Append the upstream error to history
        }

    # 2. Write the code to a secure, temporary file
    # This file automatically deletes itself when closed or when the script ends
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
        temp_script.write(generated_code)
        temp_file_path = temp_script.name

    try:
        # 3. Execute the script in an isolated subprocess
        # - capture_output=True grabs both print() statements (stdout) and crash logs (stderr)
        # - timeout=10 prevents infinite loops from burning compute
        result = subprocess.run(
            ['python', temp_file_path], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        # 4. Evaluate the result against our deterministic contract
        if result.returncode == 0 and "SUCCESS:" in stdout:
            return {
                "execution_result": stdout,
                "final_answer": stdout.replace("SUCCESS:", "").strip()
            }
        else:
            # Code ran, but logic failed (e.g., missing the SUCCESS tag)
            # Or the code crashed (returncode != 0)
            error_msg = stderr if stderr else f"Logic Error: Script ran but did not output 'SUCCESS:'. Output was: {stdout}"
            
            # We wrap the error in a list [error_msg]. 
            # Because we used operator.add in our State definition, LangGraph will append this!
            return {
                "execution_result": "FAILED",
                "error_history": [error_msg] 
            }

    except subprocess.TimeoutExpired:
        return {
            "execution_result": "FAILED",
            "error_history": ["TimeoutExpired: The generated code took longer than 10 seconds and was killed. Check for infinite loops."]
        }
    finally:
        # 5. Clean up the temporary file so we don't leak storage
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)