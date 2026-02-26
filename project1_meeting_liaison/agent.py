"""
LangGraph Agent - Project 1: Autonomous Meeting Liaison
========================================================
PURPOSE: Orchestrates the full pipeline from raw transcript → action items → drafted emails.

HOW LANGGRAPH WORKS (key concept for your presentation):
  LangGraph treats your agent as a directed graph (like a flowchart).
  - NODES are processing steps (functions that transform state)
  - EDGES connect nodes (define the flow)
  - STATE is shared data that flows through the graph

  This is better than a simple chain because:
  1. You can add conditional branches (e.g., "if no action items found, stop")
  2. Tool calls are first-class citizens (ToolNode handles them automatically)
  3. The state captures the full conversation history

GRAPH STRUCTURE for Meeting Liaison:
  START
    ↓
  [extract_action_items_node]  ← Parses transcript, returns structured action items
    ↓
  [call_tool_node]             ← LangGraph's built-in ToolNode, calls draft_followup_email
    ↓
  [format_output_node]         ← Packages results nicely
    ↓
  END
"""

import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from tools import draft_followup_email

load_dotenv()

# ─────────────────────────────────────────────
# 1. DEFINE STATE
# State = the shared data structure that every node reads from and writes to.
# TypedDict makes it easy to understand what data flows through the graph.
# ─────────────────────────────────────────────
class MeetingState(TypedDict):
    """
    The complete state of the Meeting Liaison agent.
    Every node receives this state dict and returns updates to it.
    """
    transcript: str              # Raw meeting transcript input by user
    action_items: List[str]      # Extracted action items (populated by node 1)
    drafted_emails: List[dict]   # List of {action_item, email} dicts (populated by node 2)
    error: str                   # Any error messages


# ─────────────────────────────────────────────
# 2. DEFINE NODES (the processing steps)
# ─────────────────────────────────────────────

def extract_action_items_node(state: MeetingState) -> dict:
    """
    NODE 1: Extract Action Items
    ----------------------------
    Reads the meeting transcript from state.
    Uses GPT-4o to identify all action items.
    Returns a list of clear, specific action items.

    WHY STRUCTURED OUTPUT?
      We ask GPT to return JSON so we can reliably parse the action items
      as a list (not a paragraph), making the next step predictable.
    """
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        temperature=0,
    )

    prompt = f"""You are an expert meeting analyst.
Analyze the following meeting transcript and extract ALL action items.

An action item must:
- Have a clear owner (who is responsible)
- Have a specific task (what needs to be done)
- Optionally have a deadline (when it's due)

Return ONLY a JSON array of strings. Each string is one action item.
Example: ["John to submit budget report by Friday", "Sarah to schedule follow-up meeting"]

TRANSCRIPT:
{state['transcript']}

Return ONLY the JSON array, nothing else."""

    response = llm.invoke(prompt)

    try:
        action_items = json.loads(response.content)
        if not isinstance(action_items, list):
            action_items = [response.content]
    except json.JSONDecodeError:
        # Fallback: treat entire response as one action item
        action_items = [response.content]

    print(f"[Agent] Extracted {len(action_items)} action item(s).")
    return {"action_items": action_items}


def draft_emails_node(state: MeetingState) -> dict:
    """
    NODE 2: Draft Follow-Up Emails (Tool Node behavior)
    ---------------------------------------------------
    Iterates over each action item and calls the draft_followup_email tool.
    This is where RAG is used — each email is written in the user's voice.

    NOTE: We're calling the tool directly here in a loop.
    In a more advanced version, the LLM itself would decide WHICH tool to call
    and WHEN using function calling — that's the true "Tool Node" pattern.
    """
    drafted_emails = []

    for item in state["action_items"]:
        print(f"[Agent] Drafting email for: {item[:60]}...")
        email_draft = draft_followup_email.invoke({"action_item": item})
        drafted_emails.append({
            "action_item": item,
            "email": email_draft
        })

    return {"drafted_emails": drafted_emails}


# ─────────────────────────────────────────────
# 3. BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_agent():
    """
    Assembles the LangGraph StateGraph.
    Returns a compiled, runnable agent.
    """
    # Initialize the graph with our state schema
    graph = StateGraph(MeetingState)

    # Add nodes (each node is a function that transforms state)
    graph.add_node("extract_action_items", extract_action_items_node)
    graph.add_node("draft_emails", draft_emails_node)

    # Define edges (the flow between nodes)
    graph.add_edge(START, "extract_action_items")
    graph.add_edge("extract_action_items", "draft_emails")
    graph.add_edge("draft_emails", END)

    # Compile into a runnable agent
    agent = graph.compile()
    print("[Agent] Graph compiled successfully.")
    return agent


# ─────────────────────────────────────────────
# 4. MAIN RUN FUNCTION (called from app.py)
# ─────────────────────────────────────────────

def run_meeting_liaison(transcript: str) -> dict:
    """
    Main entry point. Takes a raw transcript string and returns:
    {
        "action_items": [...],
        "drafted_emails": [{"action_item": ..., "email": ...}, ...]
    }
    """
    agent = build_agent()

    initial_state: MeetingState = {
        "transcript": transcript,
        "action_items": [],
        "drafted_emails": [],
        "error": ""
    }

    result = agent.invoke(initial_state) #this starts the LangGraph pipeline
    return result
