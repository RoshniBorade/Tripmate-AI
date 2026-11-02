import os
import re
import json
import time
from pathlib import Path
from typing import TypedDict, Annotated
import operator
import uuid
import certifi
from dotenv import load_dotenv

# Ensure .env is loaded reliably regardless of working directory
_ENV_PATH = Path(__file__).resolve().parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights, resolve_location_to_iata, parse_route


def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# ============================================================
# LLM with Token-Efficient Configuration & Rate-Limit Fallback
# ============================================================
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

def invoke_llm(messages, max_tokens: int = 600, temperature: float = 0.2) -> str:
    """
    Invokes Groq LLM with strict token bounding to prevent TPM (tokens-per-minute) rate limits.
    Includes automatic model fallback (120b -> 20b) if a rate limit or 413/429 occurs.
    """
    # Try primary model first
    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            client = ChatGroq(
                model=model_name,
                api_key=GROQ_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = client.invoke(messages)
            return response.content
        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "tokens" in err_str or "413" in err_str or "429" in err_str:
                time.sleep(1.5)
                continue
            raise e

    # Last resort fallback if both had momentary rate limits
    time.sleep(2.0)
    fallback_client = ChatGroq(
        model=FALLBACK_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return fallback_client.invoke(messages).content


# =========================
# State Definition
# =========================
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    trip_details: dict
    flight_results: str
    hotel_results: str
    itinerary: str
    budget_data: dict
    llm_calls: int


# ============================================================
# Helper: Extract Trip Details (Token-Free Regex + Lightweight Fallback)
# ============================================================
def extract_trip_details(user_query: str, history: list[AnyMessage] | None = None) -> dict:
    """
    Extracts structured trip parameters deterministically using regex first (0 tokens),
    falling back to a tiny bounded LLM call only if essential fields are ambiguous.
    """
    p_dep, p_arr, p_orig, p_dest = parse_route(user_query)

    details = {
        "origin": p_orig,
        "destination": p_dest,
        "duration_days": 5,
        "duration_specified": False,
        "nights": 4,
        "start_date": None,
        "end_date": None,
        "dates_specified": False,
        "travelers": "1 traveler (baseline)",
        "travelers_specified": False,
        "num_travelers": 1,
        "budget_target": None,
        "budget_amount": None,
        "currency": "USD"
    }

    # Extract duration
    m_days = re.search(r"(\d+)\s*(?:day|days)", user_query, re.IGNORECASE)
    if m_days:
        details["duration_days"] = int(m_days.group(1))
        details["duration_specified"] = True
        details["nights"] = max(details["duration_days"] - 1, 1)

    # Extract travelers
    m_travelers = re.search(r"(\d+)\s*(?:person|people|adult|adults|traveler|travelers)", user_query, re.IGNORECASE)
    if m_travelers:
        details["num_travelers"] = int(m_travelers.group(1))
        details["travelers"] = f"{details['num_travelers']} travelers"
        details["travelers_specified"] = True

    # Extract budget
    m_lakhs = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs)", user_query, re.IGNORECASE)
    if m_lakhs:
        lakh_val = float(m_lakhs.group(1))
        details["budget_amount"] = lakh_val * 100000.0
        details["budget_target"] = f"under {lakh_val} lakhs BDT"
        details["currency"] = "BDT"
    else:
        m_curr = re.search(r"(\$|৳|€|£|bdt|inr|usd|eur)\s*([\d,]+)", user_query, re.IGNORECASE)
        if m_curr:
            sym = m_curr.group(1).upper()
            amt = float(m_curr.group(2).replace(",", ""))
            details["budget_amount"] = amt
            if sym in ["$", "USD"]:
                details["currency"] = "USD"
            elif sym in ["৳", "BDT"]:
                details["currency"] = "BDT"
            elif sym in ["€", "EUR"]:
                details["currency"] = "EUR"
            elif sym in ["INR"]:
                details["currency"] = "INR"
            details["budget_target"] = f"{details['currency']} {amt:,.0f}"

    # Currency inference based on origin/query keywords
    q_lower = user_query.lower()
    if "bdt" in q_lower or "taka" in q_lower or "lakh" in q_lower or "bangladesh" in q_lower or (details["origin"] and "bangladesh" in str(details["origin"]).lower()):
        details["currency"] = "BDT"
    elif "inr" in q_lower or "rupee" in q_lower or (details["origin"] and "india" in str(details["origin"]).lower()):
        details["currency"] = "INR"

    # Destination fallback if still not found
    if not details["destination"]:
        m_trip = re.search(r"([A-Za-z]+)\s+trip", user_query, re.IGNORECASE)
        if m_trip:
            details["destination"] = m_trip.group(1).strip()
        else:
            details["destination"] = "Destination"

    # Flag missing departure city
    details["missing_departure"] = (details.get("origin") is None or str(details.get("origin")).strip().lower() in ["", "none", "null"])

    # Resolve IATA codes
    details["origin_iata"] = resolve_location_to_iata(details.get("origin")) if not details["missing_departure"] else None
    details["destination_iata"] = resolve_location_to_iata(details.get("destination"))

    return details


# ============================================================
# Helper: Compute Budget Breakdown (0 Tokens - Deterministic Math)
# ============================================================
def compute_budget_breakdown(trip_details: dict) -> tuple[dict, str]:
    """
    Computes mathematically verified category totals so individual expenses
    always sum exactly to the final budget. Distinguishes verified vs estimated expenses.
    """
    currency = trip_details.get("currency", "USD")
    duration = trip_details.get("duration_days", 5)
    nights = trip_details.get("nights", 4)
    travelers = trip_details.get("num_travelers", 1)
    budget_amt = trip_details.get("budget_amount")

    # If user provided a numeric budget (e.g. 200,000 BDT or $2,500 USD)
    if budget_amt and budget_amt > 0:
        total = float(budget_amt)
        # Consistent proportional allocations
        flights = round(total * 0.35, -2 if currency == "BDT" else 0)
        hotel = round(total * 0.30, -2 if currency == "BDT" else 0)
        food = round(total * 0.15, -2 if currency == "BDT" else 0)
        transit = round(total * 0.10, -2 if currency == "BDT" else 0)
        activities = round(total * 0.05, -2 if currency == "BDT" else 0)
        # Contingency absorbs remainder so sum is EXACTLY equal to total
        contingency = total - (flights + hotel + food + transit + activities)
    else:
        # Default realistic baseline estimates
        if currency == "BDT":
            flights = 65000.0 * travelers
            hotel = (10000.0 * nights)
            food = (3500.0 * duration * travelers)
            transit = (2000.0 * duration * travelers)
            activities = 12000.0 * travelers
            subtotal = flights + hotel + food + transit + activities
            contingency = round(subtotal * 0.10, -2)
            total = subtotal + contingency
        else:
            flights = 650.0 * travelers
            hotel = 100.0 * nights
            food = 45.0 * duration * travelers
            transit = 20.0 * duration * travelers
            activities = 120.0 * travelers
            subtotal = flights + hotel + food + transit + activities
            contingency = round(subtotal * 0.10, 0)
            total = subtotal + contingency

    nightly_rate = round(hotel / max(nights, 1), 2)
    daily_food = round(food / max(duration, 1), 2)
    daily_transit = round(transit / max(duration, 1), 2)

    data = {
        "currency": currency,
        "flights": flights,
        "hotel": hotel,
        "nightly_rate": nightly_rate,
        "food": food,
        "daily_food": daily_food,
        "transit": transit,
        "daily_transit": daily_transit,
        "activities": activities,
        "contingency": contingency,
        "total_budget": total,
        "total_verified": 0.0,
        "total_estimated": total
    }

    curr_sym = "৳" if currency == "BDT" else ("$" if currency == "USD" else f"{currency} ")

    budget_table = f"""| Category | Details / Calculation | Amount ({currency}) | Type |
|---|---|---|---|
| **Flights (Round-trip Airfare)** | {travelers} passenger(s) estimated airfare | {curr_sym}{flights:,.2f} | 💡 AI Estimate |
| **Accommodation** | {nights} nights × {curr_sym}{nightly_rate:,.2f}/night | {curr_sym}{hotel:,.2f} | 💡 AI Estimate |
| **Food & Dining** | {duration} days × {curr_sym}{daily_food:,.2f}/day | {curr_sym}{food:,.2f} | 💡 AI Estimate |
| **Local Transportation** | {duration} days metro passes & local transit | {curr_sym}{transit:,.2f} | 💡 AI Estimate |
| **Sightseeing & Activities** | Attractions, guided tours & entry tickets | {curr_sym}{activities:,.2f} | 💡 AI Estimate |
| **Contingency / Buffer** | Incidentals, emergency fund & shopping buffer | {curr_sym}{contingency:,.2f} | 💡 AI Estimate |
| **TOTAL ESTIMATED BUDGET** | **Exact sum of all individual categories above** | **{curr_sym}{total:,.2f}** | **Verified Math** |

- **Total Verified Expenses:** {curr_sym}0.00 *(Live booking ticket prices are not available via flight/hotel tracking APIs)*
- **Total Estimated Expenses:** {curr_sym}{total:,.2f}
- **Arithmetic Check:** {curr_sym}{flights:,.2f} + {curr_sym}{hotel:,.2f} + {curr_sym}{food:,.2f} + {curr_sym}{transit:,.2f} + {curr_sym}{activities:,.2f} + {curr_sym}{contingency:,.2f} = **{curr_sym}{total:,.2f}** (100% matched)"""

    return data, budget_table


# =========================
# Flight Agent
# =========================
def flight_agent(state: TravelState):
    user_query = state["user_query"]
    history = state.get("messages", [])

    # Extract trip details once to share across all agents
    trip_details = state.get("trip_details") or {}
    if not trip_details or not trip_details.get("destination"):
        trip_details = extract_trip_details(user_query, history)

    origin = trip_details.get("origin")
    destination = trip_details.get("destination")
    missing_departure = trip_details.get("missing_departure", False)

    # Search live flights matching ONLY this route
    if missing_departure:
        flight_data = (
            f"⚠️ **Departure City Missing**\n\n"
            f"You did not specify a departure city or airport. Live flight search requires both an origin and destination to retrieve accurate schedules.\n\n"
            f"👉 **Please specify your departure city** (e.g., Dhaka, New York, London, Delhi) so matching live flights to **{destination or 'your destination'}** can be fetched.\n\n"
            f"*Note: TripMate AI strictly displays flights matching your exact route. No unrelated routes are shown.*"
        )
    else:
        flight_data = search_flights(
            query=user_query,
            origin=origin,
            destination=destination,
            limit=3
        )

    # Provide AI-estimated flight pricing guidance clearly distinguished from live API
    curr = trip_details.get("currency", "USD")
    curr_sym = "৳" if curr == "BDT" else ("$" if curr == "USD" else f"{curr} ")
    travelers_count = trip_details.get("num_travelers", 1)

    est_fare = 65000 if curr == "BDT" else 650
    total_est_fare = est_fare * travelers_count

    flight_guidance = f"""
### 💡 AI-Estimated Airfare Guidance (Ballpark Estimates — Not Confirmed Bookings)
- **Route Corridor:** {origin or '[Departure City Required]'} ➡️ {destination or '[Destination]'}
- **Estimated Economy Ticket:** ~{curr_sym}{est_fare:,} per passenger (Total: ~{curr_sym}{total_est_fare:,} for {travelers_count} passenger(s))
- **Important Distinction:** AviationStack provides live flight tracking and schedules only. Real-time ticket booking prices are not provided by tracking APIs.
- **Booking Recommendation:** Verify real-time booking prices on Google Flights, Skyscanner, or official airline websites.
"""

    combined_flight_results = f"{flight_data}\n\n{flight_guidance}"

    return {
        "trip_details": trip_details,
        "flight_results": combined_flight_results,
        "messages": [
            AIMessage(content="Flight analysis completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Hotel Agent (Token-Optimized)
# =========================
def hotel_agent(state: TravelState):
    trip_details = state.get("trip_details", {})
    destination = trip_details.get("destination", "the destination")
    nights = trip_details.get("nights", 4)
    currency = trip_details.get("currency", "USD")
    curr_sym = "৳" if currency == "BDT" else ("$" if currency == "USD" else f"{currency} ")

    # Target search query strictly to destination
    hotel_search_query = f"Top hotels in {destination} accommodations"
    raw_hotel_search = tavily_search(
        query=hotel_search_query,
        destination=destination,
        max_results=3
    )

    prompt = f"""Recommend 3 hotel tiers (Budget, Mid-range, Luxury) for {destination} ({nights} nights).

Search Results:
{raw_hotel_search}

Instructions:
1. ONLY recommend hotels located in {destination}.
2. For each hotel, calculate: Nightly Rate × {nights} nights = Total Accommodation Cost.
3. Mark with [Verified from Search] or [AI Estimate].
Keep description concise."""

    content = invoke_llm([
        SystemMessage(content="You are a hotel consultant. Return concise hotel suggestions with exact cost arithmetic."),
        HumanMessage(content=prompt)
    ], max_tokens=550)

    return {
        "hotel_results": content,
        "messages": [
            AIMessage(content="Hotel options evaluated.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Itinerary Agent (Token-Optimized)
# =========================
def itinerary_agent(state: TravelState):
    trip_details = state.get("trip_details", {})
    destination = trip_details.get("destination", "Destination")
    duration_days = trip_details.get("duration_days", 5)
    nights = trip_details.get("nights", 4)
    start_date = trip_details.get("start_date")
    dates_specified = trip_details.get("dates_specified", False)
    travelers = trip_details.get("travelers", "1 traveler (baseline)")

    prompt = f"""Create a concise day-by-day itinerary for {destination} for exactly {duration_days} days ({nights} nights).
Travelers: {travelers}
Requested Dates: {start_date if dates_specified else 'Not specified (use tentative schedule)'}

Instructions:
1. If dates are not specified, add:
"📅 **Travel Dates Notice:** Specific dates were not provided. Scheduled as tentative {duration_days} days."
2. Include Day 1 to Day {duration_days} with bullet points for Morning, Afternoon, Evening.
3. Keep activities geographically realistic and concise."""

    content = invoke_llm([
        SystemMessage(content="You are an expert itinerary planner. Output a structured, concise daily schedule."),
        HumanMessage(content=prompt)
    ], max_tokens=950)

    return {
        "itinerary": content,
        "messages": [AIMessage(content=content)],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Final Response Agent (Zero Token-Duplication Assembly)
# =========================
def final_agent(state: TravelState):
    trip_details = state.get("trip_details", {})
    budget_data, budget_table = compute_budget_breakdown(trip_details)

    destination = trip_details.get("destination", "Destination")
    origin = trip_details.get("origin")
    missing_departure = trip_details.get("missing_departure", False)
    duration_days = trip_details.get("duration_days", 5)
    nights = trip_details.get("nights", 4)
    travelers = trip_details.get("travelers", "1 traveler")
    currency = trip_details.get("currency", "USD")

    # Generate only Section 1 and Section 6 with LLM (fast, small token footprint)
    summary_prompt = f"""Generate:
1. Trip Summary (Destination: {destination}, Origin: {'Missing' if missing_departure else origin}, Duration: {duration_days}D/{nights}N, Travelers: {travelers}, Budget: {trip_details.get('budget_target') or 'Standard'}).
2. Section 6 Practical Recommendations & Next Steps (visa, local transport cards, cash vs card, emergency tips, next steps).
Keep concise."""

    summary_content = invoke_llm([
        SystemMessage(content="You are an AI travel consultant. Provide concise trip summary and practical travel recommendations."),
        HumanMessage(content=summary_prompt)
    ], max_tokens=500)

    # Modular assembly of the 6 sections in Python — guarantees 0 token waste and 100% data integrity
    flight_section = state.get("flight_results", "")
    hotel_section = state.get("hotel_results", "")
    itinerary_section = state.get("itinerary", "")
    tot_budget = budget_data.get("total_budget", 0)
    target_budget_str = trip_details.get("budget_target") or f"{currency} {tot_budget:,.2f}"
    dep_city_str = "⚠️ **Departure City: Not specified — Please provide your departure city**" if missing_departure else f"{origin} ({trip_details.get('origin_iata', '')})"
    dates_str = trip_details.get("start_date") if trip_details.get("dates_specified") else "Tentative schedule (exact dates not specified)"

    final_answer = f"""# 1. Trip Summary
- **Destination:** {destination}
- **Departure City:** {dep_city_str}
- **Duration:** {duration_days} Days / {nights} Nights
- **Travel Dates:** {dates_str}
- **Travelers:** {travelers}
- **Target Budget:** {target_budget_str}

---

# 2. Flight Information
{flight_section}

---

# 3. Hotel Suggestions
{hotel_section}

---

# 4. Day-by-Day Itinerary
{itinerary_section}

---

# 5. Estimated Budget & Financial Breakdown
{budget_table}

---

# 6. Final Recommendations & Next Steps
{summary_content}"""

    return {
        "budget_data": budget_data,
        "messages": [AIMessage(content=final_answer)],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

def get_live_connection():
    return psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        row_factory=dict_row
    )

_conn = get_live_connection()
checkpointer = PostgresSaver(_conn)
checkpointer.setup()
travel_graph = graph.compile(checkpointer=checkpointer)


def ensure_db_connection():
    """
    Ensures the PostgreSQL connection to Render is alive.
    Render terminates idle TCP connections after a few minutes;
    this auto-reconnects transparently to prevent 500 errors.
    """
    global _conn, checkpointer, travel_graph
    try:
        if _conn is None or _conn.closed:
            _conn = get_live_connection()
            checkpointer = PostgresSaver(_conn)
            travel_graph = graph.compile(checkpointer=checkpointer)
        else:
            with _conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception:
        try:
            if _conn and not _conn.closed:
                _conn.close()
        except Exception:
            pass
        _conn = get_live_connection()
        checkpointer = PostgresSaver(_conn)
        travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# Function for FastAPI
# =========================
def run_travel_agent(user_input: str, thread_id: str | None = None):
    ensure_db_connection()

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "trip_details": {},
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "budget_data": {},
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "budget_data": result.get("budget_data", {}),
        "trip_details": result.get("trip_details", {}),
        "llm_calls": result.get("llm_calls", 0),
    }
