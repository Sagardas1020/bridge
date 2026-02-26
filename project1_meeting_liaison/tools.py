"""
Tools Module - Project 1: Meeting Liaison
==========================================
PURPOSE: Defines the LangChain "Tool" that the LangGraph agent can call.

WHY TOOLS?
  In LangGraph, a "Tool Node" is a special node where the agent can choose
  to call external functions (tools). Think of tools as the agent's hands —
  it can reach out and do things like search the web, send emails, query a DB.

  Here our tool is: draft_followup_email
  - Input: An action item (e.g. "John to submit budget by Friday")
  - Process: Retrieves the user's writing style via RAG, then asks GPT to write
  - Output: A complete draft email string
"""

import os
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from rag import retrieve_voice_examples

load_dotenv()


@tool
def draft_followup_email(action_item: str) -> str:
    """
    Drafts a follow-up email for a given action item using the user's writing voice.

    Args:
        action_item: A specific task identified from the meeting transcript
                     e.g. "Sarah to share the Q3 report with the team by Wednesday"

    Returns:
        A complete draft email in the user's writing style.
    """
    # Step 1: Retrieve relevant voice examples from RAG
    # We use the action_item as the query to find stylistically similar emails
    voice_context = retrieve_voice_examples(action_item)

    # Step 2: Build the prompt for the LLM
    prompt = f"""You are a professional email writing assistant. 
Your job is to draft a follow-up email based on a meeting action item.
You MUST write in the exact style, tone, and voice shown in the examples below.

{voice_context}

---

Now draft a follow-up email for this action item:
ACTION ITEM: {action_item}

Write ONLY the email (Subject + Body). Do not add any commentary.
"""

    # Step 3: Call Azure GPT-4o to write the email
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        temperature=0.3,
    )
    response = llm.invoke(prompt)
    return response.content


# List of all tools available to the agent
# (We can add more tools later, e.g., send_email_via_gmail)
ALL_TOOLS = [draft_followup_email]
