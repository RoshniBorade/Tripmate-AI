import os
import re
from pathlib import Path
import certifi
import airportsdata
import pycountry
import requests
from dotenv import load_dotenv

# Ensure .env is loaded regardless of execution directory
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

BASE_URL = "https://api.aviationstack.com/v1/flights"

AIRPORTS = airportsdata.load("IATA")

COUNTRY_ALIASES = {
    "usa": "US",
    "u.s.a": "US",
    "u.s.": "US",
    "america": "US",
    "united states": "US",
    "uk": "GB",
    "u.k.": "GB",
    "britain": "GB",
    "england": "GB",
    "uae": "AE",
    "dubai": "AE",
    "south korea": "KR",
    "korea": "KR",
    "russia": "RU",
    "vietnam": "VN",
    "bangladesh": "BD",
    "india": "IN",
    "japan": "JP",
    "china": "CN",
    "singapore": "SG",
    "malaysia": "MY",
    "thailand": "TH",
    "indonesia": "ID",
    "nepal": "NP",
    "qatar": "QA",
    "saudi arabia": "SA",
    "turkey": "TR",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES",
}

COUNTRY_MAIN_AIRPORT = {
    "BD": "DAC",
    "IN": "DEL",
    "JP": "NRT",
    "US": "JFK",
    "GB": "LHR",
    "AE": "DXB",
    "SG": "SIN",
    "MY": "KUL",
    "TH": "BKK",
    "ID": "CGK",
    "CN": "PEK",
    "KR": "ICN",
    "NP": "KTM",
    "QA": "DOH",
    "SA": "JED",
    "TR": "IST",
    "CA": "YYZ",
    "AU": "SYD",
    "DE": "FRA",
    "FR": "CDG",
    "IT": "FCO",
    "ES": "MAD",
}

CITY_MAIN_AIRPORT = {
    "dhaka": "DAC",
    "delhi": "DEL",
    "new delhi": "DEL",
    "mumbai": "BOM",
    "kolkata": "CCU",
    "chennai": "MAA",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "tokyo": "NRT",
    "osaka": "KIX",
    "kyoto": "KIX",
    "new york": "JFK",
    "london": "LHR",
    "dubai": "DXB",
    "singapore": "SIN",
    "kuala lumpur": "KUL",
    "bangkok": "BKK",
    "doha": "DOH",
    "istanbul": "IST",
    "toronto": "YYZ",
    "sydney": "SYD",
    "paris": "CDG",
    "rome": "FCO",
    "madrid": "MAD",
    "frankfurt": "FRA",
}


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    stop_words = [
        "flight", "flights", "ticket", "tickets", "trip", "travel",
        "plan", "complete", "days", "day", "including", "hotel",
        "hotels", "sightseeing", "under", "budget", "info", "information"
    ]
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words).strip()


def country_name_to_code(text: str):
    text = clean_text(text)

    if text in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[text]

    try:
        country = pycountry.countries.lookup(text)
        return country.alpha_2
    except LookupError:
        pass

    for country in pycountry.countries:
        country_name = country.name.lower()
        if country_name in text:
            return country.alpha_2

    for alias, code in COUNTRY_ALIASES.items():
        if alias in text:
            return code

    return None


def airport_country_matches(airport: dict, country_code: str) -> bool:
    airport_country = str(airport.get("country", "")).upper().strip()

    if airport_country == country_code:
        return True

    try:
        country = pycountry.countries.get(alpha_2=country_code)
        if country and airport_country.lower() == country.name.lower():
            return True
    except Exception:
        pass

    return False


def get_best_airport_for_country(country_code: str):
    preferred = COUNTRY_MAIN_AIRPORT.get(country_code)

    if preferred and preferred in AIRPORTS:
        return preferred

    candidates = []

    for iata, airport in AIRPORTS.items():
        if not iata:
            continue

        if airport_country_matches(airport, country_code):
            name = str(airport.get("name", "")).lower()
            city = str(airport.get("city", "")).lower()

            score = 0
            if "international" in name:
                score += 50
            if "intl" in name:
                score += 40
            if "capital" in name:
                score += 20
            if city:
                score += 5

            candidates.append((score, iata))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def resolve_location_to_iata(location: str):
    """
    Converts country/city/airport/IATA into IATA code.
    """
    if not location:
        return None

    raw_location = location.strip()

    # Direct 3-letter IATA code
    if re.fullmatch(r"[A-Za-z]{3}", raw_location):
        code = raw_location.upper()
        if code in AIRPORTS:
            return code

    location_clean = clean_text(raw_location)

    if not location_clean:
        return None

    # City preferred airport
    if location_clean in CITY_MAIN_AIRPORT:
        return CITY_MAIN_AIRPORT[location_clean]

    # Country preferred airport
    country_code = country_name_to_code(location_clean)
    if country_code:
        airport = get_best_airport_for_country(country_code)
        if airport:
            return airport

    # Exact city match from airport database
    city_matches = []

    for iata, airport in AIRPORTS.items():
        city = str(airport.get("city", "")).lower().strip()
        name = str(airport.get("name", "")).lower().strip()

        score = 0
        if city == location_clean:
            score += 100
        elif location_clean in city:
            score += 70

        if location_clean in name:
            score += 50

        if "international" in name:
            score += 10

        if score > 0:
            city_matches.append((score, iata))

    if city_matches:
        city_matches.sort(reverse=True)
        return city_matches[0][1]

    return None


def find_location_mentions(query: str):
    """
    Finds country or city names inside a natural language query.
    """
    q = query.lower()
    mentions = []

    for alias in COUNTRY_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", q):
            mentions.append(alias)

    for country in pycountry.countries:
        name = country.name.lower()
        if len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", q):
            mentions.append(name)

    for city in CITY_MAIN_AIRPORT:
        if re.search(rf"\b{re.escape(city)}\b", q):
            mentions.append(city)

    unique_mentions = []
    for item in mentions:
        if item not in unique_mentions:
            unique_mentions.append(item)

    return unique_mentions


def clean_location_name(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"\b(?:\d+|plan|a|complete|days|day|trip|tour|vacation|travel|holiday|under|with|and|for)\b", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned else text.strip().title()


def parse_route(query: str):
    """
    Parses origin and destination from natural language query.
    Returns:
    (dep_iata, arr_iata, origin_name, dest_name)

    If departure city is missing, dep_iata and origin_name will be None.
    NEVER defaults to arbitrary airports like DAC.
    """
    q = query.strip()
    q_lower = q.lower()

    # 1. Direct IATA codes like DAC to NRT or JFK - LHR
    iata_codes = re.findall(r"\b[A-Z]{3}\b", q)
    if len(iata_codes) >= 2:
        dep = iata_codes[0].upper()
        arr = iata_codes[1].upper()
        if dep in AIRPORTS and arr in AIRPORTS:
            return dep, arr, dep, arr

    # 2. Pattern: from X to Y
    match = re.search(
        r"\bfrom\s+([a-zA-Z\s]+?)\s+\bto\s+([a-zA-Z\s]+?)(?:\s+(?:on|for|under|including|with|in|at)\b|[.!?]|$)",
        q_lower,
    )
    if match:
        orig = clean_location_name(match.group(1).strip())
        dest = clean_location_name(match.group(2).strip())
        dep_iata = resolve_location_to_iata(orig)
        arr_iata = resolve_location_to_iata(dest)
        return dep_iata, arr_iata, orig, dest

    # 3. Pattern: to Y from X
    match = re.search(
        r"\bto\s+([a-zA-Z\s]+?)\s+\bfrom\s+([a-zA-Z\s]+?)(?:\s+(?:on|for|under|including|with|in|at)\b|[.!?]|$)",
        q_lower,
    )
    if match:
        dest = clean_location_name(match.group(1).strip())
        orig = clean_location_name(match.group(2).strip())
        dep_iata = resolve_location_to_iata(orig)
        arr_iata = resolve_location_to_iata(dest)
        return dep_iata, arr_iata, orig, dest

    # 4. Pattern: [Destination] trip/vacation/tour from [Origin]
    match = re.search(
        r"([a-zA-Z\s]+?)\s+(?:trip|vacation|tour|travel|holiday)\s+from\s+([a-zA-Z\s]+?)(?:\s+(?:on|for|under|including|with|in|at)\b|[.!?]|$)",
        q_lower,
    )
    if match:
        dest = clean_location_name(match.group(1).strip())
        orig = clean_location_name(match.group(2).strip())
        dep_iata = resolve_location_to_iata(orig)
        arr_iata = resolve_location_to_iata(dest)
        return dep_iata, arr_iata, orig, dest

    # 5. Fallback via location mentions
    mentions = find_location_mentions(q)

    if len(mentions) >= 2:
        # Check which mention is after "from"
        m0_from = bool(re.search(rf"\bfrom\s+{re.escape(mentions[0])}\b", q_lower))
        m1_from = bool(re.search(rf"\bfrom\s+{re.escape(mentions[1])}\b", q_lower))

        if m1_from and not m0_from:
            orig = mentions[1]
            dest = mentions[0]
        else:
            orig = mentions[0]
            dest = mentions[1]

        orig = clean_location_name(orig)
        dest = clean_location_name(dest)
        dep_iata = resolve_location_to_iata(orig)
        arr_iata = resolve_location_to_iata(dest)
        return dep_iata, arr_iata, orig, dest

    if len(mentions) == 1:
        loc = mentions[0]
        # Check if user said "from [loc]"
        if re.search(rf"\bfrom\s+{re.escape(loc)}\b", q_lower):
            orig = clean_location_name(loc)
            dep_iata = resolve_location_to_iata(orig)
            return dep_iata, None, orig, None
        else:
            # It is the destination! Departure city is MISSING!
            dest = clean_location_name(loc)
            arr_iata = resolve_location_to_iata(dest)
            return None, arr_iata, None, dest

    return None, None, None, None


def format_flight(flight: dict) -> str:
    airline = flight.get("airline", {}).get("name") or "Unknown airline"
    flight_number = flight.get("flight", {}).get("iata") or "Unknown flight number"
    status = flight.get("flight_status") or "Unknown"

    dep = flight.get("departure", {}) or {}
    arr = flight.get("arrival", {}) or {}

    dep_airport = dep.get("airport") or "Unknown departure airport"
    dep_iata = dep.get("iata") or "Unknown"
    dep_terminal = dep.get("terminal") or "N/A"
    dep_gate = dep.get("gate") or "N/A"
    dep_scheduled = dep.get("scheduled") or "Unknown"
    dep_delay = dep.get("delay")
    dep_delay_text = f"{dep_delay} minutes" if dep_delay is not None else "On time / N/A"

    arr_airport = arr.get("airport") or "Unknown arrival airport"
    arr_iata = arr.get("iata") or "Unknown"
    arr_terminal = arr.get("terminal") or "N/A"
    arr_gate = arr.get("gate") or "N/A"
    arr_scheduled = arr.get("scheduled") or "Unknown"
    arr_delay = arr.get("delay")
    arr_delay_text = f"{arr_delay} minutes" if arr_delay is not None else "On time / N/A"

    return f"""- **Airline:** {airline} | **Flight Number:** {flight_number}
  - **Status:** `{status}`
  - **Departure:** {dep_airport} ({dep_iata}) | Terminal: {dep_terminal} | Gate: {dep_gate}
  - **Departure Time:** {dep_scheduled} (Delay: {dep_delay_text})
  - **Arrival:** {arr_airport} ({arr_iata}) | Terminal: {arr_terminal} | Gate: {arr_gate}
  - **Arrival Time:** {arr_scheduled} (Delay: {arr_delay_text})"""


def search_flights(query: str = "", origin: str | None = None, destination: str | None = None, limit: int = 5) -> str:
    """
    Searches AviationStack for live flights matching ONLY the user's specific route.
    If departure city is missing, explicitly asks the user and returns no flights.
    Never displays unrelated routes or global flights.
    """
    if not API_KEY:
        return (
            "⚠️ **Flight API Key Missing**: AVIATIONSTACK_API_KEY is not set in `.env`.\n"
            "Live flight schedules cannot be fetched."
        )

    # Resolve origin and destination
    dep_iata = None
    arr_iata = None
    orig_name = origin
    dest_name = destination

    if orig_name:
        dep_iata = resolve_location_to_iata(orig_name)
    if dest_name:
        arr_iata = resolve_location_to_iata(dest_name)

    # If not provided explicitly, try parsing from query
    if not dep_iata or not arr_iata:
        p_dep, p_arr, p_orig, p_dest = parse_route(query)
        if not dep_iata and p_dep:
            dep_iata = p_dep
            orig_name = orig_name or p_orig
        if not arr_iata and p_arr:
            arr_iata = p_arr
            dest_name = dest_name or p_dest

    # Case 1: Departure city is missing!
    if not orig_name and not dep_iata:
        dest_display = dest_name or "your destination"
        return (
            f"⚠️ **Departure City Missing**\n\n"
            f"You did not specify a departure city or airport. Live flight search requires both an origin and destination to retrieve accurate schedules.\n\n"
            f"👉 **Please specify your departure city** (e.g., Dhaka, New York, London, Delhi) so matching live flights to **{dest_display}** can be retrieved.\n\n"
            f"*Note: TripMate AI only displays flights matching your exact route. No unrelated routes are shown.*"
        )

    # Case 2: Destination is missing!
    if not dest_name and not arr_iata:
        return (
            "⚠️ **Destination City Missing**\n\n"
            "Please specify your travel destination to search for matching flights."
        )

    # Case 3: Could not resolve IATA codes
    if not dep_iata:
        return (
            f"⚠️ **Unrecognized Departure Location**: Could not map \"{orig_name}\" to a commercial airport IATA code.\n"
            f"Please provide a major nearby city or 3-letter airport code (e.g., DAC, JFK, LHR, DEL)."
        )

    if not arr_iata:
        return (
            f"⚠️ **Unrecognized Destination Location**: Could not map \"{dest_name}\" to a commercial airport IATA code.\n"
            f"Please provide a major nearby city or 3-letter airport code (e.g., NRT, HND, DXB, CDG)."
        )

    # Case 4: Query AviationStack for EXACT route
    params = {
        "access_key": API_KEY,
        "dep_iata": dep_iata,
        "arr_iata": arr_iata,
        "limit": min(limit, 100),
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=20)
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"⚠️ **Flight API Connection Notice**: Could not connect to AviationStack ({e}). Live flight schedules temporarily unavailable."
    except ValueError:
        return "⚠️ **Flight API Error**: Returned invalid response."

    if "error" in data:
        err = data["error"]
        return (
            f"⚠️ **Live Flight API Notice**: {err.get('message', 'API request failed')} (Code: {err.get('code')}).\n"
            f"Real-time schedule lookup is unavailable."
        )

    flight_data = data.get("data", [])

    # Strict filtering: MUST match dep_iata AND arr_iata exactly
    matching_flights = [
        f for f in flight_data
        if f.get("departure", {}).get("iata") == dep_iata and f.get("arrival", {}).get("iata") == arr_iata
    ]

    route_header = f"**Route:** {orig_name or dep_iata} ({dep_iata}) ➡️ {dest_name or arr_iata} ({arr_iata})"

    if not matching_flights:
        return (
            f"### 🔴 Verified Live Flight Status (via AviationStack API)\n\n"
            f"{route_header}\n\n"
            f"**Live Status:** No direct scheduled flights are currently active in the real-time AviationStack feed for this direct route.\n\n"
            f"📌 *Note: Flight schedules vary by season and day of the week. Connecting flights with layovers may be required for this corridor.\n"
            f"Ticket pricing is not provided by live tracking APIs; refer to the AI-estimated fare guidance below.*"
        )

    formatted_flights = [format_flight(f) for f in matching_flights[:limit]]
    flights_list_str = "\n\n".join(formatted_flights)

    return (
        f"### 🔴 Verified Live Flight Results (via AviationStack API)\n\n"
        f"{route_header}\n\n"
        f"{flights_list_str}\n\n"
        f"📌 *Note: AviationStack provides real-time operational status and schedules only. Live ticket fares are not available from flight tracking APIs.*"
    )
