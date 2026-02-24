from langgraph.graph import StateGraph, END
from core.state import AgentState
from nodes.coder import write_code_node
from nodes.executor import execute_code_node
from nodes.extractor import extract_data_node

# --- 1. Routing Logic (The "Self-Correction" Brain) ---
def verification_router(state: AgentState) -> str:
    """
    This function looks at the execution result and decides the next step.
    This is the core of the 'System 2' thinking.
    """
    result = state.get("execution_result", "")
    
    # If the code executed perfectly and returned an answer
    if "SUCCESS:" in result:
        return "end"
    
    # If we've hit our hard cap to prevent infinite billing/loops
    if state.get("loop_count", 0) >= 3:
        print("Max retries reached. Forcing exit.")
        return "max_retries"
    
    # Otherwise, it failed. We route back to the Vision Extractor to look at the image again.
    return "retry"


# --- 2. Graph ---
def build_graph():
    # Initialize the graph with our custom TypedDict schema
    workflow = StateGraph(AgentState)
    
    # Add the three primary nodes
    workflow.add_node("vision_extractor", extract_data_node)
    workflow.add_node("code_generator", write_code_node)
    workflow.add_node("sandbox_executor", execute_code_node)
    
    # Entry point
    workflow.set_entry_point("vision_extractor")
    
    # Extractor always passes data to the Coder
    workflow.add_edge("vision_extractor", "code_generator")
    
    # Coder always passes its script to the Executor Sandbox
    workflow.add_edge("code_generator", "sandbox_executor")
    
    # The Executor triggers the conditional routing logic
    workflow.add_conditional_edges(
        "sandbox_executor",      # The node making the decision
        verification_router,     # The function evaluating the state
        {
            "end": END,                      # Stop and return final answer
            "max_retries": END,              # Stop to save compute
            "retry": "vision_extractor"      # Self-Correction Loop
        }
    )
    
    # Compile the graph into an executable application
    app = workflow.compile()
    return app