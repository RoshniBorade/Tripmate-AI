import os
import re
from pathlib import Path
from dotenv import load_dotenv
from tavily import TavilyClient

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

_API_KEY = os.getenv("TAVILY_API_KEY")

_client = None
if _API_KEY:
    try:
        _client = TavilyClient(api_key=_API_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize TavilyClient: {e}")


def tavily_search(query: str, destination: str | None = None, max_results: int = 5) -> str:
    """
    Searches Tavily for hotel and travel information.
    Ensures results are relevant to the destination when provided.
    Gracefully handles API/network errors.
    """
    if not _client:
        return "⚠️ Hotel search unavailable: TAVILY_API_KEY is not configured in `.env`."

    try:
        response = _client.search(
            query=query,
            max_results=max(max_results * 2, 8)  # Fetch extra so we can filter for destination relevance
        )
    except Exception as e:
        return f"⚠️ Hotel search notice: Live search temporarily unavailable ({e})."

    raw_results = response.get("results", [])
    if not raw_results:
        return f"No search results returned for query: {query}"

    # Destination relevance verification
    dest_keywords = []
    if destination:
        # Extract meaningful words (length >= 3)
        words = re.findall(r"[A-Za-z]{3,}", destination.lower())
        dest_keywords = [w for w in words if w not in ["the", "and", "for", "trip", "tour", "best", "near"]]

    verified_results = []
    unverified_results = []

    for r in raw_results:
        title = r.get("title", "Unknown")
        url = r.get("url", "")
        snippet = r.get("content", "").strip()
        combined_text = f"{title} {snippet}".lower()

        # Check if snippet or title contains any destination keyword
        is_relevant = True
        if dest_keywords:
            is_relevant = any(k in combined_text for k in dest_keywords)

        # Truncate long snippets
        if len(snippet) > 350:
            snippet = snippet[:350].rsplit(" ", 1)[0] + "..."

        entry = {
            "title": title,
            "url": url,
            "snippet": snippet,
            "is_relevant": is_relevant
        }

        if is_relevant:
            verified_results.append(entry)
        else:
            unverified_results.append(entry)

    # Prioritize destination-relevant results
    final_list = verified_results[:max_results]
    if len(final_list) < max_results:
        # Fill with remaining only if no destination was specified
        if not destination:
            final_list.extend(unverified_results[:max_results - len(final_list)])

    if not final_list:
        final_list = raw_results[:max_results]

    results_formatted = []
    for i, r in enumerate(final_list, 1):
        results_formatted.append(f"{i}. **{r['title']}**\n   🔗 {r['url']}\n   📝 {r['snippet']}")

    return "\n\n".join(results_formatted)
