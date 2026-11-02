# TripMate AI

TripMate AI is a web-based travel planner powered by a LangGraph multi-agent workflow. Describe a trip in natural language to get flight information, hotel suggestions, a day-by-day itinerary, and a budget breakdown in one response.

## Features

- Extracts destinations, trip duration, dates, travelers, and budget from a travel request.
- Looks up flight information with AviationStack and travel or hotel suggestions with Tavily.
- Generates itineraries and recommendations with Groq-hosted language models, with a fallback model for rate limits.
- Runs flight, hotel, itinerary, and response agents in a LangGraph workflow.
- Saves conversation state using a PostgreSQL LangGraph checkpointer.
- Provides a browser interface with copy and PDF download options.

Flight and hotel results depend on the availability and coverage of their external APIs. Add the corresponding API keys to enable those searches.

## Requirements

- Python 3.11 or newer
- PostgreSQL database
- API credentials for Groq and the PostgreSQL database
- AviationStack and Tavily API credentials for live flight and web-search results

## Setup

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/RoshniBorade/Tripmate-AI.git
   cd Tripmate-AI
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and set your credentials:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key
   DATABASE_URL=your_postgresql_connection_string
   AVIATIONSTACK_API_KEY=your_aviationstack_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

   `GROQ_API_KEY` and `DATABASE_URL` are required. The flight and web-search keys enable their respective integrations. Keep `.env` private; do not commit real credentials.

## Run locally

Start the FastAPI application from the project root:

```bash
uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. The application initializes the PostgreSQL checkpointer when the backend starts, so the database must be reachable.

The health endpoint is available at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

## API

### `POST /api/travel`

Request a travel plan. `thread_id` is optional; reuse a returned ID to continue a conversation.

```json
{
  "message": "Plan a 7-day trip to Japan from Bangladesh for two people",
  "thread_id": null
}
```

A successful response includes `answer`, `thread_id`, `flight_results`, `hotel_results`, `itinerary`, and `llm_calls`.

## Docker

Build and run the container from the project root:

```bash
docker build -t tripmate-ai .
docker run --env-file .env -p 8000:8000 tripmate-ai
```

The container listens on port `8000`. Ensure the configured PostgreSQL database is reachable from the container.

## Project layout

```text
app.py                 FastAPI application and HTTP endpoints
backend.py             LangGraph workflow, agents, and PostgreSQL persistence
tools/flight_tool.py   Flight search and location resolution
tools/tavily_tool.py   Travel and hotel web search
templates/             Web interface template
static/                Frontend JavaScript and styles
requirements.txt       Python dependencies
Dockerfile             Container build configuration
```
