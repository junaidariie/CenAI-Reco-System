# =========================
# IMPORTS
# =========================
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import json
from dotenv import load_dotenv

load_dotenv()


# =========================
# TAVILY TOOL
# =========================
@tool
def tavily_search(query: str) -> dict:
    """
    Perform a real-time web search using Tavily.
    """
    try:
        search = TavilySearchResults(max_results=2)
        results = search.run(query)
        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}
    

# =========================
# LLM
# =========================
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.4,
    streaming=True
)


# =========================
# STATE
# =========================
class MovieState(TypedDict, total=False):
    title: str
    overview: str
    web_context: str
    key_plot_points: str
    iconic_moments: str
    themes: str
    interesting_facts: str
    songs: str
    trailer: str
    summary: str


# =========================
# NODE: FETCH WEB CONTEXT
# =========================
def fetch_web_context(state: MovieState):
    title = state["title"]

    query = f"""
    Find reliable and up-to-date information about the movie "{title}".

    Focus on:
    - Official trailers (studio or verified YouTube channels)
    - Soundtrack / songs (Spotify, Apple Music, IMDb soundtrack)
    - Verified trivia or interesting facts
    - Release details and reception (optional)

    Prefer sources like:
    - IMDb
    - Wikipedia
    - Official studio websites
    - Verified YouTube channels
    - Major entertainment publications

    Avoid:
    - Fan theories
    - Reviews without factual info
    - Opinion-heavy blogs
    """

    web = tavily_search.run(query)

    return {
        "web_context": str(web)
    }

# =========================
# HELPER PROMPT RUNNER
# =========================
def run_llm(prompt: str) -> str:
    return llm.invoke(prompt).content


# =========================
# ANALYSIS NODES
# =========================
def find_key_points(state: MovieState):
    prompt = f"""
You are a professional movie analyst.

Movie title: {state['title']}

Overview:
{state['overview']}

Verified web context (may include reviews, trivia, or plot confirmations):
{state['web_context']}

Task:
Extract the MOST IMPORTANT plot points that define the story.

Guidelines:
- Focus on STORY EVENTS, not themes or opinions
- Keep it chronological
- Avoid unnecessary details or long explanations
- Do NOT invent scenes not supported by the overview or web context

Output format (strict):
- Bullet list
- 5–7 plot points max
- Each point: 1 concise sentence
"""
    return {"key_plot_points": run_llm(prompt)}


def find_iconic_moments(state: MovieState):
    prompt = f"""
You are a film analyst identifying ICONIC moments.

Movie title: {state['title']}

Overview:
{state['overview']}

Verified web context (reviews, trivia, cultural references):
{state['web_context']}

Task:
Identify the most ICONIC moments from the movie.

Definition of iconic:
- Scenes that audiences remember most
- Moments often referenced in reviews, memes, or pop culture
- Visually, emotionally, or narratively standout scenes

Guidelines:
- Do NOT summarize the full plot
- Avoid repeating basic plot points
- Focus on memorable SCENES or MOMENTS
- Base choices on common recognition (not personal opinion)

Output format (strict):
- Numbered list
- 4–6 iconic moments
- Each item:
  • Scene title (short)
  • One-sentence explanation of why it’s iconic
"""
    return {"iconic_moments": run_llm(prompt)}

def find_themes(state: MovieState):
    prompt = f"""
You are a movie analyst focusing on THEMES.

Movie title: {state['title']}

Overview:
{state['overview']}

Verified web context (critical analysis, reviews, commentary):
{state['web_context']}

Task:
Identify the CORE THEMES explored in the movie.

Guidelines:
- Themes should be CONCEPTS (not plot points or morals)
- Avoid vague words like "life" or "journey" unless specific
- Base themes on story events and critical interpretation
- Do NOT over-explain

Output format (strict):
- Bullet list
- 3–5 themes only
- Each theme format:
  **Theme name** – one concise explanatory sentence
"""
    return {"themes": run_llm(prompt)}

def find_interesting_facts(state: MovieState):
    prompt = f"""
You are a movie researcher collecting VERIFIED trivia.

Movie title: {state['title']}

Overview:
{state['overview']}

Verified web context (interviews, trivia, production notes, reviews):
{state['web_context']}

Task:
Extract interesting and lesser-known facts about the movie.

Guidelines:
- Facts must be BASED on the web context or widely known sources
- Avoid speculation or unverified claims
- Focus on production, casting, behind-the-scenes, or reception
- Do NOT repeat plot points

Output format (strict):
- Bullet list
- 4–6 facts
- Each fact:
  • One concise sentence
  • Clearly factual (no opinions)
"""
    return {"interesting_facts": run_llm(prompt)}

def find_songs(state: MovieState):
    prompt = f"""
You are extracting OFFICIAL soundtrack information.

Movie title: {state['title']}

Verified web context (soundtrack listings, music platforms, official sources):
{state['web_context']}

Task:
Identify the official soundtrack songs associated with this movie.

Rules:
- Include ONLY officially released songs (not background score unless famous)
- Prefer reliable sources (Spotify, YouTube, Apple Music, IMDb soundtrack)
- Do NOT guess or invent songs
- Do NOT add explanations or extra text

Output format (STRICT — follow exactly):
- One song per line
- Each line format:
  [song name, official link]

If no reliable song information is found:
- Return an empty list: []
"""
    return {"songs": run_llm(prompt)}


def find_trailer(state: MovieState):
    prompt = f"""
You are retrieving OFFICIAL movie trailer information.

Movie title: {state['title']}

Verified web context (official YouTube channels, studio pages, IMDb, Wikipedia):
{state['web_context']}

Task:
Find official trailer links for this movie.

Rules:
- ONLY official trailers (no fan edits, reactions, reviews)
- Prefer studio or verified YouTube channels
- Do NOT invent or approximate links
- Do NOT include commentary or descriptions

Output format (STRICT — follow exactly):
- One trailer per line
- Each line format:
  [trailer name, official link]

If no official trailer is found:
- Return an empty list: []
"""
    return {"trailer": run_llm(prompt)}


# =========================
# FINAL SUMMARY
# =========================
def generate_summary(state: MovieState):
    prompt = f"""
You are generating a FINAL movie summary for a frontend application.

Movie title: {state['title']}

Use ONLY the information provided below.
Do NOT add new facts.
Do NOT use markdown.
Do NOT include extra text.

INPUT DATA
---------

KEY PLOT POINTS:
{state['key_plot_points']}

ICONIC MOMENTS:
{state['iconic_moments']}

THEMES:
{state['themes']}

INTERESTING FACTS:
{state['interesting_facts']}

SONGS:
{state['songs']}

TRAILERS:
{state['trailer']}

---------

TASK:
Return a VALID JSON object that follows this schema EXACTLY.

JSON SCHEMA (STRICT):
{{
  "overview": "2–3 sentence high-level movie overview",
  "key_moments": ["moment 1", "moment 2", "moment 3"],
  "themes": ["theme 1", "theme 2"],
  "notable_facts": ["fact 1", "fact 2"],
  "soundtrack_highlights": ["song name 1", "song name 2"],
  "official_trailer": "trailer name"
}}
"""
    return {"summary": run_llm(prompt)}



# =========================
# GRAPH
# =========================
graph = StateGraph(MovieState)

graph.add_node("fetch_web_context", fetch_web_context)
graph.add_node("find_key_points", find_key_points)
graph.add_node("find_iconic_moments", find_iconic_moments)
graph.add_node("find_themes", find_themes)
graph.add_node("find_interesting_facts", find_interesting_facts)
graph.add_node("find_songs", find_songs)
graph.add_node("find_trailer", find_trailer)
graph.add_node("generate_summary", generate_summary)

graph.add_edge(START, "fetch_web_context")

graph.add_edge("fetch_web_context", "find_key_points")
graph.add_edge("fetch_web_context", "find_iconic_moments")
graph.add_edge("fetch_web_context", "find_themes")
graph.add_edge("fetch_web_context", "find_interesting_facts")
graph.add_edge("fetch_web_context", "find_songs")
graph.add_edge("fetch_web_context", "find_trailer")

graph.add_edge("find_key_points", "generate_summary")
graph.add_edge("find_iconic_moments", "generate_summary")
graph.add_edge("find_themes", "generate_summary")
graph.add_edge("find_interesting_facts", "generate_summary")
graph.add_edge("find_songs", "generate_summary")
graph.add_edge("find_trailer", "generate_summary")

graph.add_edge("generate_summary", END)

workflow = graph.compile()

def summarise_movie(title: str, overview: str):
    result = workflow.invoke({
        "title": title,
        "overview": overview
    })

    raw_summary = result["summary"]

    try:
        return json.loads(raw_summary)
    except json.JSONDecodeError:
        raise ValueError("LLM returned invalid JSON")


#print(summarise_movie("Jumanji", "Four teenagers are sucked into a magical video game..."))