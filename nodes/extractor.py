import json
import re
from typing import Dict, Any
from core.state import AgentState
from core.llm_engine import ai_engine

def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Forces the raw text output from the VLM into a valid Python dictionary.
    This prevents crashes if the model adds markdown formatting or conversational filler.
    """
    # Regex to extract content between ```json and ``` if present
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    
    if json_match:
        json_string = json_match.group(1)
    else:
        # If no backticks, assume the whole text is meant to be JSON
        json_string = raw_text.strip()
        
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # If parsing fails entirely, we return a dict with the error so the agent can self-correct
        return {"extraction_error": f"Failed to parse JSON. Raw output was: {raw_text}"}


def extract_data_node(state: AgentState) -> dict:
    
    base_prompt = f"""
        You are an analytical data extraction agent. 
        A user wants to know the answer to this query: "{state['user_query']}"
        
        Your job is NOT to calculate the final answer. Your job is ONLY to extract the RAW data points required for a Python script to calculate the answer later.
        
        You MUST return the data STRICTLY as a valid JSON object matching this exact schema:
        {{
            "reasoning": "Step-by-step logic explaining exactly which raw columns/values are needed to answer the query.",
            "extracted_data": {{
                "Raw_Label_1": Value,
                "Raw_Label_2": Value
            }}
        }}
        
        Failure to put 'reasoning' first will cause system failure. Do not include markdown explanations outside the JSON.
    """
    
    if state.get("error_history"):
        base_prompt += "\n\nCRITICAL WARNING: Your previous extractions failed. Adjust your extraction based on these errors:"
        for err in state.get("error_history"):
            base_prompt += f"\n- Error: {err}"

    # Qwen2.5-VL format for multimodal messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": state["image_path"]},
                {"type": "text", "text": base_prompt},
            ],
        }
    ]
    
    # Send to our local 7B model
    raw_output = ai_engine.generate_response(messages)
    
    extracted_json = clean_and_parse_json(raw_output)
    return {
        "extracted_data": extracted_json,
        "loop_count": state.get("loop_count", 0) + 1
    }