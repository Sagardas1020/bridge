"""
Tools Module - Project 2: Personalized Gift Scout
===================================================
PURPOSE: Defines the search tool the agent uses to find real products with live prices.

WHY TAVILY?
  Tavily is a search API designed specifically for LLM agents.
  Unlike Google Search scraping, Tavily returns clean, structured results
  that are easy for LLMs to read. It's LangChain-native and just works.

  The agent uses this tool when it has collected enough info about the
  recipient and is ready to search for gifts.
"""

import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


def get_search_tool():
    """
    Returns the Tavily search tool configured for gift searching.
    max_results=5 means we get 5 product results per search.
    """
    return TavilySearchResults(
        max_results=5,
        description=(
            "Use this tool to search for gift products online. "
            "Input should be a specific search query like "
            "'best gifts for someone who loves hiking under $50'. "
            "Returns real product names, descriptions, and links."
        )
    )


@tool
def search_gifts(query: str) -> str:
    """
    Searches the web for gift product listings based on the recipient's hobbies/interests.

    Args:
        query: A specific gift search query built from the recipient's hobbies/interests.
               Example: "unique gifts for photography enthusiasts"

    Returns:
        A formatted string of product results with names, descriptions, and links.
    """
    search = TavilySearchResults(
        max_results=8,
        search_depth="advanced",
        include_answer=True,
        include_domains=[
            "amazon.com", "etsy.com", "uncommongoods.com",
            "bestbuy.com", "walmart.com", "target.com",
            "giftguide.com", "wirecutter.com", "buzzfeed.com"
        ]
    )
    results = search.invoke(query)

    if not results:
        return "No results found. Try a different search query."

    # Format results nicely
    formatted = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "Product")
        url = result.get("url", "")
        content = result.get("content", "")[:400]  # Keep enough detail
        formatted.append(f"{i}. **{title}**\n   {content}\n   🔗 {url}")

    return "\n\n".join(formatted)


ALL_TOOLS = [search_gifts]
