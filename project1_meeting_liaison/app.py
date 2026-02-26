"""
Streamlit App - Project 1: Autonomous Meeting Liaison
======================================================
PURPOSE: The user-facing interface for the Meeting Liaison agent.

UI STRUCTURE:
  Sidebar → Upload sample emails (feeds the RAG voice database)
  Main Area:
    - Paste meeting transcript OR use a sample
    - Click "Run Agent"
    - See extracted action items
    - See drafted emails (expandable, copyable)

WHY STREAMLIT?
  Streamlit turns Python scripts into web apps with zero HTML/CSS/JS.
  Perfect for data science / AI demos. Each widget (button, text_area, etc.)
  is a Python function call.
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Make sure imports from same folder work
sys.path.insert(0, os.path.dirname(__file__))

from agent import run_meeting_liaison
from rag import build_voice_index

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Liaison Agent",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Autonomous Meeting Liaison")
st.caption("Paste a meeting transcript → Get action items + follow-up emails in your voice")

# ─────────────────────────────────────────────
# SIDEBAR: Upload Voice Samples (RAG)
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🎙️ Your Writing Voice")
    st.info(
        "Upload 3–5 sample emails you've written before. "
        "The agent uses these to match your tone and style when drafting emails."
    )

    uploaded_files = st.file_uploader(
        "Upload sample emails (.txt files)",
        type=["txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        emails_dir = os.path.join(os.path.dirname(__file__), "sample_emails")
        os.makedirs(emails_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            save_path = os.path.join(emails_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        if st.button("📥 Build Voice Index (RAG)", use_container_width=True):
            with st.spinner("Indexing your emails into the vector database..."):
                try:
                    build_voice_index()
                    st.success(f"✅ Voice index built from {len(uploaded_files)} email(s)!")
                except Exception as e:
                    st.error(f"Error building index: {e}")

    st.divider()
    st.caption("**How RAG works here:** Your emails are split into chunks, "
               "converted to vectors (embeddings), and stored in ChromaDB. "
               "When drafting, the agent finds the most similar style examples "
               "and uses them as context for GPT-4o.")

    st.divider()
    st.subheader("⚖️ Responsible AI")
    st.warning(
        "**AI-generated content — human review required.**\n\n"
        "- Drafted emails may contain errors or inaccuracies. Always review and edit before sending.\n"
        "- Meeting transcripts are sent to Azure OpenAI for processing. Do not include sensitive or confidential information.\n"
        "- This tool assists decision-making but does not replace human judgment.\n"
        "- Powered by Azure OpenAI — subject to [Microsoft's Responsible AI principles](https://www.microsoft.com/en-us/ai/responsible-ai)."
    )

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
# Load sample transcript from file (kept in sample_transcripts/ folder for easy editing)
_sample_path = os.path.join(os.path.dirname(__file__), "sample_transcripts", "sample1_q3_budget_review.txt")
with open(_sample_path, "r", encoding="utf-8") as _f:
    SAMPLE_TRANSCRIPT = _f.read().strip()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Meeting Transcript")

    use_sample = st.checkbox("Use sample transcript")
    transcript = st.text_area(
        "Paste your meeting transcript here:",
        value=SAMPLE_TRANSCRIPT if use_sample else "",
        height=350,
        placeholder="Paste your meeting notes or transcript here..."
    )

    run_button = st.button("🚀 Run Agent", type="primary", use_container_width=True)

with col2:
    st.subheader("✅ Extracted Action Items")

    if run_button:
        if not transcript.strip():
            st.warning("Please paste a transcript first.")
        else:
            with st.spinner("Agent is analyzing the transcript..."):
                try:
                    result = run_meeting_liaison(transcript)

                    # Display action items
                    action_items = result.get("action_items", [])
                    if action_items:
                        for i, item in enumerate(action_items, 1):
                            st.markdown(f"**{i}.** {item}")
                    else:
                        st.info("No action items found.")

                    # Store results in session state so emails show below
                    st.session_state["result"] = result

                except Exception as e:
                    st.error(f"Agent error: {e}")
                    st.stop()

# ─────────────────────────────────────────────
# DRAFTED EMAILS SECTION
# ─────────────────────────────────────────────
if "result" in st.session_state:
    st.divider()
    st.subheader("📧 Drafted Follow-Up Emails")
    st.caption("These emails are written in your voice, based on your uploaded samples.")

    drafted_emails = st.session_state["result"].get("drafted_emails", [])

    st.info("💡 These are AI-generated drafts. Review carefully and personalise before sending.")

    for i, item in enumerate(drafted_emails, 1):
        with st.expander(f"✉️ Email {i}: {item['action_item'][:80]}...", expanded=(i == 1)):
            st.text_area(
                label="Draft (copy and edit as needed):",
                value=item["email"],
                height=250,
                key=f"email_{i}"
            )
