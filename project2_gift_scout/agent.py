"""
LangGraph Agent - Project 2: Personalized Gift Scout
=====================================================
PURPOSE: An agent that "interviews" the user about a gift recipient,
         then searches for real products with live prices.

KEY DESIGN PRINCIPLE (important for presentation):
  Multi-turn conversation requires the graph to run ONE step per user message,
  then WAIT for the user to respond. The loop is NOT inside the graph —
  it spans multiple invocations (one per user turn).

  Graph structure per invocation:

  START
    ↓
  [router_node]  ← Checks state: "Do we have enough info?"
    ├─ "interview"  → [interview_node] → END  (returns one question to user)
    └─ "search"     → [search_node] → [format_results_node] → END

  App.py calls agent.invoke() again each time the user replies.
  This is the correct HUMAN-IN-THE-LOOP pattern for LangGraph agents.

HOW THIS DIFFERS FROM PROJECT 1:
  Project 1 is a LINEAR graph: transcript → extract → draft → END
  Project 2 uses a CONDITIONAL graph with a router node, and is
  designed for HUMAN-IN-THE-LOOP — one agent step per user message.

STATE DESIGN:
  We track conversation messages AND a structured recipient profile.
  The profile gets filled in progressively as the interview continues.
"""

import os
import json
from typing import TypedDict, List, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tools import search_gifts

load_dotenv()


# ─────────────────────────────────────────────
# 1. DEFINE STATE
# ─────────────────────────────────────────────
class GiftState(TypedDict):
    """
    Shared state that persists across all turns of the conversation.

    messages:          Full conversation history (user ↔ agent)
    recipient_profile: Structured info collected about the recipient
    questions_asked:   Count of questions asked (limits interview length)
    search_results:    Final gift recommendations from Tavily
    ready_to_search:   True when profile is complete and search can start
    """
    messages: List[dict]
    recipient_profile: dict
    questions_asked: int
    search_results: str
    ready_to_search: bool


# ─────────────────────────────────────────────
# 2. SYSTEM PROMPT
# ─────────────────────────────────────────────

INTERVIEW_SYSTEM_PROMPT = """You are a friendly gift shopping assistant called "Gift Scout".
Your job is to learn about a gift recipient through a natural, warm conversation.

You need to collect all of the following:
- Their name or relationship (e.g. "my mom", "my colleague Mark")
- Age range: child (under 12), teen (13-17), adult (18-60), or senior (60+)
- Main hobbies or interests (at least 2 specific ones)
- Budget range: under $25 / $25-$50 / $50-$100 / $100+
- Occasion: birthday, holiday, thank-you, graduation, etc.

Rules:
- Ask ONE focused question at a time. Be warm and conversational.
- If the user already volunteered information, acknowledge it and ask about what's missing.
- Once you have ALL five pieces of info, respond with EXACTLY this format on two lines:
  PROFILE_COMPLETE
  {"name": "...", "age": "...", "hobbies": ["...", "..."], "budget": "...", "occasion": "..."}
"""


# ─────────────────────────────────────────────
# 3. NODES
# ─────────────────────────────────────────────

def router_node(state: GiftState) -> dict:
    """
    ROUTER NODE
    -----------
    Checks if we have enough info to start searching.
    Force-searches after 6 questions to prevent endless loops.
    Pure logic — does NOT call the LLM.
    """
    force = state.get("questions_asked", 0) >= 6
    return {"ready_to_search": state.get("ready_to_search", False) or force}


def interview_node(state: GiftState) -> dict:
    """
    INTERVIEW NODE
    --------------
    Generates the next interview question based on conversation history.
    Detects when the profile is complete and sets ready_to_search=True.
    Returns to END after one question (human-in-the-loop pattern).
    """
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        temperature=0.7,
    )

    messages = [SystemMessage(content=INTERVIEW_SYSTEM_PROMPT)]
    for msg in state["messages"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    try:
        response = llm.invoke(messages)
        raw_reply = response.content
    except ValueError as e:
        if "content filter" in str(e).lower():
            raw_reply = "I'm sorry, I had trouble processing that. Could you rephrase and tell me a bit about the person you're buying a gift for?"
        else:
            raise

    profile = state.get("recipient_profile", {})
    ready = False

    if "PROFILE_COMPLETE" in raw_reply:
        try:
            json_start = raw_reply.index("{")
            json_end = raw_reply.rindex("}") + 1
            profile = json.loads(raw_reply[json_start:json_end])
        except (ValueError, json.JSONDecodeError):
            pass
        display_reply = (
            "Perfect, I have everything I need! 🎯 "
            "Let me search for the best gift options right now..."
        )
        ready = True
    else:
        display_reply = raw_reply

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": display_reply}],
        "recipient_profile": profile,
        "questions_asked": state.get("questions_asked", 0) + 1,
        "ready_to_search": ready,
    }


def search_node(state: GiftState) -> dict:
    """
    SEARCH NODE (Tool Node)
    -----------------------
    Builds a targeted query from the recipient profile and calls
    the Tavily search tool to retrieve real products with live prices.
    This is the agent's "hand" reaching out to the real world.
    """
    profile = state["recipient_profile"]
    hobbies = ", ".join(profile.get("hobbies", ["general"]))
    budget = profile.get("budget", "under $100")
    occasion = profile.get("occasion", "gift")
    age = profile.get("age", "adult")

    query = (
        f"best {occasion} gift ideas for {age} who loves {hobbies} "
        f"budget {budget} buy online 2025 with prices"
    )
    print(f"[Gift Scout] Tavily query: '{query}'")

    results = search_gifts.invoke({"query": query})
    return {"search_results": results}


def format_results_node(state: GiftState) -> dict:
    """
    FORMAT RESULTS NODE
    -------------------
    Asks GPT-4o to turn raw Tavily results into a warm,
    personalized gift recommendation list.
    """
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        temperature=0.5,
    )
    profile = state["recipient_profile"]

    prompt = f"""The user is searching for a gift for: {json.dumps(profile, indent=2)}

Here are real product search results from the web:
{state['search_results']}

Present the 4-5 most relevant options as a friendly, organized list.
For each gift include:
- 🎁 Product name and brief description
- 💰 Price or price range (if mentioned)
- ⭐ Why it suits this specific person (mention their hobbies/occasion)
- 🔗 Where to buy (link if available)

Be warm, personal, and end with a helpful purchasing tip."""

    response = llm.invoke(prompt)
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response.content}]
    }


# ─────────────────────────────────────────────
# 4. ROUTING FUNCTION (Conditional Edge)
# ─────────────────────────────────────────────

def decide_after_router(state: GiftState) -> Literal["interview", "search"]:
    """
    CONDITIONAL EDGE FUNCTION
    -------------------------
    Reads the ready_to_search flag set by router_node.
    Returns "search" if profile is complete, "interview" otherwise.
    This is how LangGraph implements branching logic.
    """
    return "search" if state.get("ready_to_search", False) else "interview"


# ─────────────────────────────────────────────
# 5. BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_agent():
    """
    Compiles the Gift Scout LangGraph.

    Per invocation:
      START → router → interview → END   (still collecting info)
      START → router → search → format → END  (ready to search)
    """
    graph = StateGraph(GiftState)

    graph.add_node("router", router_node)
    graph.add_node("interview", interview_node)
    graph.add_node("search", search_node)
    graph.add_node("format_results", format_results_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        decide_after_router,
        {"interview": "interview", "search": "search"}
    )
    graph.add_edge("interview", END)
    graph.add_edge("search", "format_results")
    graph.add_edge("format_results", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 6. PUBLIC API (called from app.py)
# ─────────────────────────────────────────────

def get_initial_state() -> GiftState:
    """Creates a fresh empty state to start a new conversation."""
    return {
        "messages": [],
        "recipient_profile": {},
        "questions_asked": 0,
        "search_results": "",
        "ready_to_search": False,
    }


def run_turn(state: GiftState, user_message: str = None) -> GiftState:
    """
    Runs ONE turn of the Gift Scout agent.
    - Adds user_message to state (if provided)
    - Runs the graph: either asks one question OR runs the full search
    - Returns updated state for app.py to store in session_state

    App.py calls this once per user message — the loop spans user turns.
    """
    agent = build_agent()

    if user_message:
        state = {**state, "messages": state["messages"] + [
            {"role": "user", "content": user_message}
        ]}

    return agent.invoke(state)
