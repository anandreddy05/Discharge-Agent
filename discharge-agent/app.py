import streamlit as st
import json
import requests

st.set_page_config(page_title="Clinical AI Agent", page_icon="🏥", layout="wide")

st.title("🏥 Agentic Discharge Summary Generator")
st.markdown(
    "Upload patient notes to generate a clinically safe, zero-hallucination discharge summary."
)

# --- SIDEBAR: Persistent Controls ---
with st.sidebar:
    st.header("Upload Medical Record")
    uploaded_file = st.file_uploader("Upload PDF Notes", type=["pdf"])
    generate_btn = st.button(
        "🚀 Generate Report", use_container_width=True, type="primary"
    )

# --- MAIN APP STATE ---
if "summary_data" not in st.session_state:
    st.session_state.summary_data = None
if "logs" not in st.session_state:
    st.session_state.logs = None

# --- EXECUTION LOGIC ---
if generate_btn:
    if not uploaded_file:
        st.sidebar.error("Please upload a PDF first.")
    else:
        with st.spinner("🤖 Agent is reading and reasoning... This may take a minute."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }
                api_url = "http://localhost:8000/generate_summary"
                api_response = requests.post(api_url, files=files, timeout=300)

                if api_response.status_code == 200:
                    data = api_response.json()
                    st.session_state.summary_data = data.get("final_summary")
                    st.session_state.logs = data.get("agent_chain_of_thought")
                else:
                    st.error(
                        f"Backend API Error: {api_response.status_code} - {api_response.text}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "🚨 Could not connect to the backend! Is your FastAPI server (main.py) running?"
                )
            except Exception as e:
                st.error(f"System Error: {str(e)}")

# --- RENDER RESULTS CLEANLY ---
if st.session_state.summary_data:
    st.markdown("### 🕵️‍♂️ Agent Chain of Thought")
    with st.expander("View Agent Execution Logs", expanded=False):
        if not st.session_state.logs:
            st.warning("No logs recorded.")
        else:
            for step in st.session_state.logs:
                st.markdown(f"#### Iteration {step.get('iteration', '?')}")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("**🧠 Reasoning:**")
                    st.info(step.get("reasoning", "No reasoning provided."))
                    st.markdown(
                        f"**🛠️ Action Chosen:** `{step.get('action', 'Unknown')}`"
                    )

                with col2:
                    st.markdown("**📥 Inputs (Args):**")
                    st.json(step.get("args", {}))

                if step.get("result"):
                    st.markdown("**📤 Tool Result:**")
                    st.code(step.get("result"), language="markdown")

                st.divider()

    st.markdown("### 📋 Final Discharge Summary Draft")
    missing_info = st.session_state.summary_data.get("missing_critical_info", [])
    conflicts = st.session_state.summary_data.get("identified_conflicts", [])

    if missing_info or conflicts:
        st.warning("⚠️ **CLINICAL REVIEW REQUIRED**")
        if missing_info:
            st.error(f"**Missing Data:** {', '.join(missing_info)}")
        if conflicts:
            st.error(f"**Conflicts Detected:** {', '.join(conflicts)}")

    st.json(st.session_state.summary_data)
