import streamlit as st
import tempfile
import os
from PIL import Image
import sys

from dotenv import load_dotenv
load_dotenv()

# Add the root directory to the system path so we can import our core graph
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.graph import build_graph

# --- Page Configuration ---
st.set_page_config(
    page_title="Self-Correcting Vision Agent", 
    page_icon="🤖", 
    layout="wide"
)

st.title("Multimodal Reasoning & Self-Correcting Agent")
st.markdown("""
This agent extracts data from complex charts and writes Python scripts to verify its own math deterministically. 
If the code crashes, it loops back, reads the stack trace, and looks at the image again to correct its mistakes.
""")

# --- Layout ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("1. Input")
    uploaded_file = st.file_uploader("Upload a Chart or Graph", type=["png", "jpg", "jpeg"])
    query = st.text_input("Analytical Query", value="What is the percentage drop in revenue between Q2 and Q3?")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Target Image", use_container_width=True)

with col2:
    st.subheader("2. Agent Reasoning Trace")
    
    if st.button("Run Agentic Analysis", type="primary"):
        if not uploaded_file or not query:
            st.warning("Please upload an image and enter a query.")
        else:
            # 1. Save the uploaded file to a temporary location so our PyTorch model can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_image_path = tmp_file.name

            # 2. Initialize Graph and State
            app = build_graph()
            initial_state = {
                "image_path": temp_image_path,
                "user_query": query,
                "extracted_data": {},
                "generated_code": "",
                "execution_result": "",
                "error_history": [],
                "loop_count": 0,
                "final_answer": ""
            }

            # 3. Stream the Output into the UI
            # st.status creates a great animated loading box that we can update dynamically
            with st.status("Initializing AI Engine...", expanded=True) as status:
                try:
                    for output in app.stream(initial_state):
                        for node_name, state_update in output.items():
                            
                            if node_name == "vision_extractor":
                                loop_num = state_update.get('loop_count', 1)
                                if loop_num > 1:
                                    status.update(label=f"Self-Correction Loop Active (Attempt {loop_num})...")
                                    st.warning(f"**Re-evaluating image based on previous failure...**")
                                else:
                                    status.update(label="Extracting Data from Image...")
                                    
                                st.markdown("##### Extracted JSON:")
                                st.json(state_update.get("extracted_data", {}))
                                st.divider()
                                
                            elif node_name == "code_generator":
                                status.update(label="Writing Python Verification Script...")
                                st.markdown("##### Generated Sandbox Code:")
                                st.code(state_update.get("generated_code", ""), language="python")
                                st.divider()
                                
                            elif node_name == "sandbox_executor":
                                if "final_answer" in state_update:
                                    # SUCCESS!
                                    status.update(label="Execution Successful!", state="complete")
                                    st.success(f"### Final Verified Answer:\n**{state_update['final_answer']}**")
                                    st.balloons() # A little flair for the demo video
                                elif "error_history" in state_update:
                                    # FAILURE! Show the error and loop back
                                    status.update(label="Execution Failed. Triggering Recalibration...")
                                    latest_error = state_update['error_history'][-1]
                                    st.error(f"**Sandbox Exception Caught:**\n\n```text\n{latest_error}\n```")
                                    st.divider()
                                    
                except Exception as e:
                    status.update(label="Critical Pipeline Failure", state="error")
                    st.error(f"System Error: {str(e)}")
                finally:
                    # Clean up the temp image
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)