from typing import TypedDict, Annotated, List, Dict, Any
import operator

class AgentState(TypedDict):
    """
    Represents the shared memory of our Self-Correcting Vision Agent.
    Every node in our graph will receive this state and return updates to it.
    """
    # --- Input ---
    image_path: str
    user_query: str
    
    # --- Intermediate Processing ---
    # The structured data extracted by the VLM (e.g., Qwen2-VL)
    extracted_data: Dict[str, Any] 
    
    # The Python code generated to process that data and answer the query
    generated_code: str            
    
    # The stdout or traceback from running the code
    execution_result: str          
    
    # --- Self-Correction Management ---
    # Using Annotated with operator.add tells LangGraph to APPEND to this list 
    # rather than overwrite it. This allows the agent to remember all previous mistakes.
    error_history: Annotated[List[str], operator.add] 
    
    # To prevent infinite billing or infinite loops, we hard-cap the retries
    loop_count: int                
    
    # --- Output ---
    final_answer: str