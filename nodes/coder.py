import re
from core.state import AgentState
from core.state import AgentState
from core.llm_engine import ai_engine

def extract_python_code(raw_text: str) -> str:
    """
    Strips out markdown and conversational filler to isolate the executable Python script.
    """
    code_match = re.search(r'```python\n(.*?)\n```', raw_text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return raw_text.strip()

def write_code_node(state: AgentState) -> dict:
    # print(f"--- Running Local Coder Node ---")
    
    extracted_data = state.get("extracted_data", {})
    if "extraction_error" in extracted_data:
        return {"generated_code": f"FAILED_BEFORE_EXECUTION: {extracted_data['extraction_error']}"}

    base_prompt = f"""
    You are a Python data analyst. 
    Data: {extracted_data}
    User Query: "{state['user_query']}"
    
    Write a Python script to calculate the exact answer. Hardcode the data dictionary.
    You MUST print the final answer starting with "SUCCESS: " followed by the result.
    Output ONLY valid Python code inside ```python blocks.
    """
    
    if state.get("error_history"):
        base_prompt += "\n\nCRITICAL WARNING: Your previous code crashed. Fix it based on these traces:"
        for err in state.get("error_history"):
            base_prompt += f"\n- Traceback: {err}"

    # For text-only tasks, we omit the "image" dict
    messages = [
        {"role": "system", "content": "You write executable Python code without markdown filler."},
        {"role": "user", "content": [{"type": "text", "text": base_prompt}]}
    ]
    
    # Send to the SAME local 7B model 
    raw_llm_response = ai_engine.generate_response(messages)
    
    clean_code = extract_python_code(raw_llm_response)
    return {"generated_code": clean_code}