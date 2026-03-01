# ============================================================
#  LANGGRAPH MULTI-TOOL AGENT
#  Tools: Wikipedia Scraper | Weather API | Calculator
#  LLM  : Groq (LLaMA 3.3 70B) - Free & Fast
#  Flow : LangGraph ReAct Agent with Stateful Chat Memory
# ============================================================
#
#  INSTALL DEPENDENCIES (run once):
#
#  pip install langgraph langchain-core langchain-groq
#  pip install requests beautifulsoup4 python-dotenv
#
#  FREE APIs USED:
#  - Wikipedia     : https://en.wikipedia.org/wiki/{topic}  (no key needed)
#  - Open-Meteo    : https://open-meteo.com  (no key needed, 100% free)
#  - Calculator    : pure Python math (no API needed)
#  - Groq LLM      : https://console.groq.com (free API key)
#
#  CREATE a .env file:
#  GROQ_API_KEY=your_groq_api_key_here
# ============================================================

import os
import re
import math
import json
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from typing import Annotated

# --- LangChain / LangGraph ---
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# ============================================================
# CONFIG
# ============================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"   # free, best quality

# ============================================================
# TOOL 1: Wikipedia Scraper (Free, No API Key)
#
#  Dynamically scrapes: https://en.wikipedia.org/wiki/{topic}
#  Extracts the first meaningful paragraphs from the page.
#  No API key needed — uses requests + BeautifulSoup.
# ============================================================
@tool
def wikipedia_search(topic: str) -> str:
    """
    Search Wikipedia for information about any person, place, event, or concept.
    Use this tool when the user asks about facts, history, biographies, or general knowledge.
    Input: topic name (e.g. 'Python programming language', 'Elon Musk', 'India')
    """
    try:
        # Format topic for URL: replace spaces with underscores
        url_topic = topic.strip().replace(" ", "_")
        url       = f"https://en.wikipedia.org/wiki/{url_topic}"

        headers  = {"User-Agent": "Mozilla/5.0 (educational-agent/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 404:
            return f"Wikipedia page not found for '{topic}'. Try a more specific topic name."

        if response.status_code != 200:
            return f"Failed to fetch Wikipedia page (status {response.status_code})."

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract main content paragraphs (skip empty ones)
        paragraphs = soup.find_all("p")
        content    = []
        char_count = 0

        for p in paragraphs:
            text = p.get_text().strip()
            # Skip very short paragraphs (usually navigation/empty)
            if len(text) > 80:
                content.append(text)
                char_count += len(text)
                # Stop after ~2000 characters to keep context focused
                if char_count >= 2000:
                    break

        if not content:
            return f"Could not extract content from Wikipedia page for '{topic}'."

        result = f"Wikipedia summary for '{topic}':\n\n" + "\n\n".join(content[:4])
        return result

    except requests.exceptions.Timeout:
        return "Wikipedia request timed out. Please try again."
    except Exception as e:
        return f"Wikipedia scraping error: {str(e)}"


# ============================================================
# TOOL 2: Weather Report (Free, No API Key - Open-Meteo)
#
#  Uses Open-Meteo API: completely free, no key required.
#  First geocodes the city using Open-Meteo geocoding API,
#  then fetches current weather for that location.
# ============================================================
@tool
def get_weather(city: str) -> str:
    """
    Get the current real-time weather report for any city in the world.
    Use this tool when the user asks about weather, temperature, or climate conditions.
    Input: city name (e.g. 'Chennai', 'London', 'New York', 'Tokyo')
    """
    try:
        # Step 1: Geocode city name -> latitude/longitude
        geo_url    = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
        geo_resp   = requests.get(geo_url, params=geo_params, timeout=10)
        geo_data   = geo_resp.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"City '{city}' not found. Please check the city name and try again."

        location  = geo_data["results"][0]
        lat       = location["latitude"]
        lon       = location["longitude"]
        city_name = location.get("name", city)
        country   = location.get("country", "")
        timezone  = location.get("timezone", "UTC")

        # Step 2: Fetch current weather using lat/lon
        weather_url    = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude"          : lat,
            "longitude"         : lon,
            "current"           : [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
                "cloud_cover"
            ],
            "timezone"          : timezone,
            "forecast_days"     : 1
        }
        w_resp = requests.get(weather_url, params=weather_params, timeout=10)
        w_data = w_resp.json()

        if "current" not in w_data:
            return f"Could not fetch weather data for {city_name}."

        current = w_data["current"]

        # Weather code descriptions (WMO standard)
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Heavy thunderstorm"
        }
        code        = current.get("weather_code", 0)
        description = weather_codes.get(code, "Unknown conditions")

        report = f"""
Weather Report for {city_name}, {country}
{'='*45}
Condition        : {description}
Temperature      : {current.get('temperature_2m')}°C
Feels Like       : {current.get('apparent_temperature')}°C
Humidity         : {current.get('relative_humidity_2m')}%
Wind Speed       : {current.get('wind_speed_10m')} km/h
Wind Direction   : {current.get('wind_direction_10m')}°
Precipitation    : {current.get('precipitation')} mm
Cloud Cover      : {current.get('cloud_cover')}%
Time             : {current.get('time')}
{'='*45}
        """.strip()

        return report

    except requests.exceptions.Timeout:
        return "Weather request timed out. Please try again."
    except Exception as e:
        return f"Weather fetch error: {str(e)}"


# ============================================================
# TOOL 3: Calculator (Pure Python, No API)
#
#  Handles: basic math, percentages, square roots, powers,
#  trigonometry, logarithms, and complex expressions.
#  Safe eval using math module only — no code injection risk.
# ============================================================
@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations. Handles basic arithmetic, percentages,
    square roots, powers, trigonometry, and logarithms.
    Use this tool for any math question or numerical computation.
    Input examples: '25 * 4', 'sqrt(144)', '15% of 200', '2**10', 'sin(90)'
    """
    try:
        # Clean the expression
        expr = expression.strip()

        # Handle "X% of Y" pattern
        percent_match = re.match(r"(\d+\.?\d*)%\s*of\s*(\d+\.?\d*)", expr, re.IGNORECASE)
        if percent_match:
            percent = float(percent_match.group(1))
            total   = float(percent_match.group(2))
            result  = (percent / 100) * total
            return f"{percent}% of {total} = {result}"

        # Handle "X% + Y" or discount patterns
        expr = re.sub(r"(\d+)%", r"(\1/100)", expr)

        # Safe math namespace — only allow math functions
        safe_namespace = {
            "sqrt" : math.sqrt,
            "abs"  : abs,
            "pow"  : pow,
            "round": round,
            "sin"  : lambda x: math.sin(math.radians(x)),
            "cos"  : lambda x: math.cos(math.radians(x)),
            "tan"  : lambda x: math.tan(math.radians(x)),
            "log"  : math.log10,
            "ln"   : math.log,
            "pi"   : math.pi,
            "e"    : math.e,
            "ceil" : math.ceil,
            "floor": math.floor,
        }

        result = eval(expr, {"__builtins__": {}}, safe_namespace)

        # Format: show int if whole number, else round to 6 decimals
        if isinstance(result, float):
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 6)

        return f"{expression} = {result}"

    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except Exception as e:
        return f"Calculation error for '{expression}': {str(e)}. Try simpler expressions like '25*4' or 'sqrt(16)'."


# ============================================================
# LANGGRAPH STATE
#
#  TypedDict defines the shape of the graph's state.
#  'messages' uses add_messages reducer which APPENDS
#  new messages instead of replacing — this is how
#  LangGraph maintains full conversation history (memory).
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# BUILD LANGGRAPH AGENT
#
#  LangGraph Flow (ReAct pattern):
#
#  START
#    |
#    v
#  [agent_node]          <- LLM decides: answer or call tool?
#    |
#    |-- "call_tool" --> [tool_node] -> back to agent_node
#    |
#    |-- "end"       --> END
#
#  The LLM sees the full message history every time,
#  so it always has context of the whole conversation.
# ============================================================
def build_agent():

    # --- Register all tools ---
    tools    = [wikipedia_search, get_weather, calculator]
    tool_node = ToolNode(tools)

    # --- Load Groq LLM and bind tools ---
    # bind_tools tells the LLM what tools are available
    # and teaches it how to call them with proper JSON args
    llm = ChatGroq(
        api_key    =GROQ_API_KEY,
        model_name =GROQ_MODEL,
        temperature=0.3
    )
    llm_with_tools = llm.bind_tools(tools)

    # System prompt: defines agent personality and tool usage rules
    SYSTEM_PROMPT = """You are a smart and helpful AI assistant with access to three tools:

1. wikipedia_search(topic) - Search Wikipedia for facts, history, biographies, concepts
2. get_weather(city)       - Get real-time weather for any city in the world
3. calculator(expression)  - Perform math calculations

RULES:
- Use wikipedia_search when asked about people, places, events, science, history, etc.
- Use get_weather when asked about weather, temperature, or climate for a city
- Use calculator for any math, numbers, percentages, or calculations
- You can call multiple tools in sequence if needed
- Always answer in a clear, friendly, and concise way
- Remember the full conversation history — refer to previous messages when relevant
"""

    # --- Agent Node: LLM decides what to do ---
    def agent_node(state: AgentState):
        messages = state["messages"]

        # Add system prompt at the start if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # --- Router: should we call a tool or are we done? ---
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]

        # If the LLM called any tools -> go to tool_node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tool"

        # Otherwise -> end and return answer to user
        return "end"

    # --- Build the Graph ---
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("agent")

    # Add conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "call_tool": "tools",   # LLM wants to call a tool
            "end"       : END       # LLM has final answer
        }
    )

    # After tool runs -> always go back to agent
    graph.add_edge("tools", "agent")

    # Compile the graph
    return graph.compile()


# ============================================================
# MAIN - Interactive Chat Loop with Memory
#
#  The 'conversation_history' list grows with every turn.
#  Each time we call the agent, we pass the FULL history
#  so the LLM always knows what was said before.
#  This is LangGraph's built-in stateful memory via
#  the add_messages reducer in AgentState.
# ============================================================
def main():
    print("\n" + "="*55)
    print("   LANGGRAPH MULTI-TOOL AGENT")
    print(f"   LLM  : Groq ({GROQ_MODEL})")
    print("   Tools: Wikipedia | Weather | Calculator")
    print("="*55)
    print("\n  Ask me anything!")
    print("  Examples:")
    print("    'Who is Elon Musk?'")
    print("    'What is the weather in Chennai?'")
    print("    'What is 25% of 840?'")
    print("    'Tell me about Python programming'")
    print("    'What is sqrt(256) + 10?'")
    print("\n  Type 'exit' to quit | 'clear' to reset memory")
    print("="*55 + "\n")

    # Build the agent graph
    agent = build_agent()

    # Conversation history — this IS the memory
    # Every message (human + AI + tool) is stored here
    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Clear memory and start fresh
        if user_input.lower() == "clear":
            conversation_history = []
            print("\n[Memory cleared. Starting fresh conversation.]\n")
            continue

        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))

        print("\nAgent: ", end="", flush=True)

        try:
            # Invoke the graph with FULL conversation history
            # LangGraph's add_messages reducer handles state properly
            result = agent.invoke({"messages": conversation_history})

            # Get all messages from result
            result_messages = result["messages"]

            # Find the final AI response (last AIMessage without tool_calls)
            final_response = None
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage):
                    if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        final_response = msg.content
                        break

            if final_response:
                print(final_response)
            else:
                print("I processed your request.")

            # Update conversation history with all new messages
            # (includes tool calls, tool results, and final answer)
            conversation_history = result_messages

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                print("Rate limit hit. Please wait a moment and try again.")
            else:
                print(f"Error: {error_msg}")

        print("-" * 55 + "\n")


if __name__ == "__main__":
    main()