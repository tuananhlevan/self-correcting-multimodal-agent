from langgraph.graph import StateGraph, END
from core.state import AgentState
from nodes.coder import write_code_node
from nodes.executor import execute_code_node
from nodes.extractor import extract_data_node

# We will import the actual execution logic from our nodes/ folder later
# from nodes.extractor import extract_data_node
# from nodes.coder import write_code_node
# from nodes.executor import execute_code_node

# --- 1. Define the Routing Logic (The "Self-Correction" Brain) ---
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
    # print(f"Verification failed. Initiating self-correction loop {state.get('loop_count', 0) + 1}...")
    return "retry"


# --- 2. Build the Graph ---
def build_graph():
    # Initialize the graph with our custom TypedDict schema
    workflow = StateGraph(AgentState)
    
    # Add our three primary nodes (Using dummy functions for now until Step 3)
    workflow.add_node("vision_extractor", extract_data_node)
    workflow.add_node("code_generator", write_code_node)
    workflow.add_node("sandbox_executor", execute_code_node)
    
    # For now, let's pretend we added the nodes. 
    # Here is how the strict flow is defined:
    
    # Set the entry point
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
            "retry": "vision_extractor"      # The Self-Correction Loop!
        }
    )
    
    # Compile the graph into an executable application
    app = workflow.compile()
    return app

if __name__ == "__main__":
    # If you run this file directly, it just proves the graph compiles cleanly
    agent_app = build_graph()
    print("Graph compiled successfully. Ready for node injection.")