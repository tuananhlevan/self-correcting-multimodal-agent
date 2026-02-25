import os
from core.graph import build_graph

def run_agent(image_path: str, query: str):
    """
    Initializes the LangGraph agent, sets the starting state, 
    and streams the execution to the console for full observability.
    """
    # 1. Initialize the compiled graph
    app = build_graph()
    
    # 2. Define the initial state (The starting memory)
    # Notice we start with an empty error_history and 0 loop_count
    initial_state = {
        "image_path": image_path,
        "user_query": query,
        "extracted_data": {},
        "generated_code": "",
        "execution_result": "",
        "error_history": [],
        "loop_count": 0,
        "final_answer": ""
    }
    
    print("=" * 60)
    print(f"🤖 Starting Agentic Analysis")
    print(f"📁 Image: {image_path}")
    print(f"❓ Query: '{query}'")
    print("=" * 60)
    
    # 3. Stream the execution
    # .stream() yields the state updates from each node as they happen.
    # This is crucial for observability—it proves the agent is actually looping.
    try:
        for output in app.stream(initial_state):
            # output is a dict where the key is the node name, and the value is the state update
            for node_name, state_update in output.items():
                print(f"\n[Node Executed]: {node_name.upper()}")
                
                # --- Observability Printouts ---
                if node_name == "vision_extractor":
                    loop_num = state_update.get("loop_count", 1)
                    if loop_num > 1:
                        print(f"   SELF-CORRECTION LOOP ACTIVE (Attempt {loop_num})")
                        print("   Re-evaluating image based on previous errors...")
                    else:
                        print("   Initial image data extracted.")
                        
                elif node_name == "code_generator":
                    print("   Python verification script generated.")
                    
                elif node_name == "sandbox_executor":
                    if "final_answer" in state_update:
                        print(f"\nSUCCESS! Final Answer: {state_update['final_answer']}")
                    elif "error_history" in state_update:
                        # Grab the most recent error we just appended
                        latest_error = state_update['error_history'][-1] 
                        print(f"   EXECUTION FAILED. Caught Error:\n   {latest_error[:150]}...") # Truncate for console
                        print("   Routing back to vision extractor...")
                        
    except Exception as e:
        print(f"\nCritical Pipeline Failure: {e}")

    print("\n" + "=" * 60)
    print("Workflow Complete.")


if __name__ == "__main__":
    # To run this, you need a test image in your root directory.
    sample_image = "revenue_chart.png"
    sample_query = "What is the percentage drop in revenue between Q2 and Q3?"
    
    if not os.path.exists(sample_image):
        print(f"Error: Please place a test image named '{sample_image}' in the root directory.")
        # Create a dummy file just to prevent the script from crashing during dry-runs
        with open(sample_image, 'w') as f:
            f.write("dummy image content")
            
    run_agent(sample_image, sample_query)