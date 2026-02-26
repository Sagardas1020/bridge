"""
Streamlit App - Project 2: Personalized Gift Scout
===================================================
PURPOSE: A chat-style interface for the Gift Scout agent.

UI DESIGN:
  - Chat window showing the full conversation
  - Input box at the bottom for user responses
  - Sidebar showing the collected recipient profile as it fills in
  - Final results displayed at the end of the chat

WHY CHAT UI HERE?
  Project 2 is inherently conversational (interview-style).
  Streamlit's st.chat_message() perfectly renders this as a chat.
  The state is stored in st.session_state between messages —
  this is how Streamlit maintains state across reruns.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from agent import run_turn, get_initial_state, GiftState

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gift Scout Agent",
    page_icon="🎁",
    layout="wide"
)

st.title("🎁 Personalized Gift Scout")
st.caption("Tell me about your recipient and I'll find the perfect gift with live prices")

# ─────────────────────────────────────────────
# SIDEBAR: Recipient Profile (fills in live)
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("👤 Recipient Profile")
    st.caption("This fills in as we chat!")

    if "agent_state" in st.session_state:
        profile = st.session_state["agent_state"].get("recipient_profile", {})
        if profile:
            for key, value in profile.items():
                label = key.replace("_", " ").title()
                if isinstance(value, list):
                    value = ", ".join(value)
                st.markdown(f"**{label}:** {value}")
        else:
            st.info("Profile will appear here as you answer questions.")
    else:
        st.info("Start a conversation to see the profile here.")

    st.divider()

    if st.button("🔄 Start Over", use_container_width=True):
        for key in ["agent_state", "conversation_started"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.divider()
    st.caption("**How this works:**\n\n"
               "1. The agent interviews you with targeted questions\n"
               "2. A router checks if enough info is collected\n"
               "3. Once ready, it calls Tavily (live web search)\n"
               "4. GPT formats the results as personalized recommendations")

    st.divider()
    st.subheader("⚖️ Responsible AI")
    st.warning(
        "**AI-generated recommendations — use your own judgement.**\n\n"
        "- Gift suggestions are AI-generated based on web search results. Verify prices and availability before purchasing.\n"
        "- Product links are sourced from Tavily web search and are not endorsed or verified by this app.\n"
        "- Do not share sensitive personal details about the gift recipient.\n"
        "- Powered by Azure OpenAI — subject to [Microsoft's Responsible AI principles](https://www.microsoft.com/en-us/ai/responsible-ai)."
    )

# ─────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────

# Initialize conversation
if "conversation_started" not in st.session_state:
    with st.spinner("Starting interview..."):
        initial_state = run_turn(get_initial_state())  # Gets first question
        st.session_state["agent_state"] = initial_state
        st.session_state["conversation_started"] = True

# Display the full conversation history
agent_state = st.session_state.get("agent_state", {})
messages = agent_state.get("messages", [])

for msg in messages:
    role = msg["role"]
    content = msg["content"]

    if role == "assistant":
        with st.chat_message("assistant", avatar="🎁"):
            st.markdown(content)
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)

# ─────────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────────

# Only show input if the search isn't done yet
search_done = bool(agent_state.get("search_results", ""))

if not search_done:
    user_input = st.chat_input("Type your answer here...")

    if user_input:
        # Show user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Run the agent
        with st.spinner("Gift Scout is thinking..."):
            new_state = run_turn(
                st.session_state["agent_state"],
                user_message=user_input
            )
            st.session_state["agent_state"] = new_state

        st.rerun()
else:
    st.success("✅ Gift search complete! See recommendations above.")
    st.caption("Click **Start Over** in the sidebar to search for another recipient.")
