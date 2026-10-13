"""
agents/planner_agent.py

Week 12's first agent node: task decomposition. Reads state["user_query"],
asks the LLM to break it into subtasks and decide which downstream agents
are actually needed, matching the architecture diagram's "7.1 Planner
Agent (Task Decomposition)" box — Understand User Query, Decompose into
Subtasks, Select Required Agents, Plan Execution Flow.

Why `required_agents` is more than decorative, despite state.py's own
"largely fixed order" scope note
-------------------------------------------------------------------------
state.py's docstring is explicit that this pipeline doesn't support
arbitrary branching or cycles — but "always run all 7 downstream agents,
every time, regardless of the query" would make the diagram's own
"Select Required Agents" bullet meaningless. The middle ground this file
commits to: agents/graph.py (built later) uses `required_agents` for
simple binary include/skip decisions per agent — e.g. a query with no
figures/tables in its retrieved evidence has no reason to run the
Multimodal Understanding agent, and a query with no temporal angle has no
reason to run Trend & Evolution. That's a bounded, single-hop routing
decision, not dynamic replanning or cycles — consistent with the scope
state.py already committed to.

Why the LLM is asked for strict JSON, with a fallback if parsing fails
---------------------------------------------------------------------------
An LLM asked to "decompose this query" in free text gives prose, not
something graph.py's conditional edges can act on programmatically.
Asking for a fixed JSON shape and parsing it is the standard way to get
structured output from a plain chat completion API (no function-calling
schema is set up here — see below for why). Real LLM output is not
perfectly reliable at following a format request 100% of the time, so
_parse_plan() falls back to "run every agent, treat the whole query as
one subtask" on any parsing failure — the SAFE default (over-inclusion)
rather than under-including and silently skipping an agent the query
actually needed.

Why plain JSON-in-the-prompt, not a function-calling/tool-schema API
---------------------------------------------------------------------------
llm/base.py's LLMProvider interface is intentionally minimal (messages
in, text out — see that file's own docstring) specifically so Groq and
Gemini could be swapped without either provider needing to implement a
shared function-calling schema, which differs meaningfully between the
two SDKs. Asking for JSON in the prompt and parsing the text response
costs a little reliability compared to native structured output, but
keeps every agent in this project talking to the same simple
LLMProvider.generate_text() interface, provider-agnostic, exactly as
llm/base.py's own docstring commits to.
"""

from __future__ import annotations

import json
import re

from temporal_graphrag.agents.state import AgentState
from temporal_graphrag.llm.factory import llm_factory
from temporal_graphrag.llm.base import LLMMessage

__all__ = ["ALL_AGENT_NAMES", "planner_node"]

# Every downstream agent name the Planner can select — kept as one list
# here so graph.py (built later) and this file agree on the exact same
# set of valid names, rather than each hardcoding its own copy.
ALL_AGENT_NAMES = [
    "retrieval",
    "multimodal",
    "reasoning",
    "verification",
    "summarization",
    "trend",
    "evaluation",
]

_SYSTEM_PROMPT = f"""You are the planning component of a research-paper \
question-answering system. Given a user's question, decompose it into a \
short list of concrete subtasks, and decide which of the following \
downstream processing stages are actually needed to answer it well:

{", ".join(ALL_AGENT_NAMES)}

- "retrieval" is ALWAYS needed (every query needs evidence).
- "multimodal" is only needed if the question is about figures, tables, \
charts, diagrams, or numeric results likely shown in a table.
- "trend" is only needed if the question asks how something has changed, \
evolved, or developed over time, or asks about the newest/latest work.
- "verification" and "summarization" are needed for essentially every \
query that expects a real answer (not just a citation lookup).
- "evaluation" is needed whenever a real summary was produced (which is \
almost always, if "summarization" is included).
- "reasoning" is needed whenever the question requires connecting or \
comparing multiple pieces of evidence, not just restating one fact.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"subtasks": ["...", "..."], "required_agents": ["...", "..."]}}
"""


async def planner_node(state: AgentState) -> dict:
    """LangGraph node: reads state["user_query"], returns a partial state
    update with "subtasks" and "required_agents" populated."""
    user_query = state["user_query"]

    provider = llm_factory.get_provider(
        [LLMMessage(role="system", content=_SYSTEM_PROMPT), LLMMessage(role="user", content=user_query)]
    )
    response_text = await provider.generate_text(user_query, system=_SYSTEM_PROMPT, temperature=0.2)

    subtasks, required_agents = _parse_plan(response_text, user_query)

    return {
        "subtasks": subtasks,
        "required_agents": required_agents,
        "messages": [
            {"role": "assistant", "content": f"[Planner] {len(subtasks)} subtask(s), agents: {required_agents}"}
        ],
    }


def _parse_plan(response_text: str, user_query: str) -> tuple[list[str], list[str]]:
    """Parses the LLM's JSON response into (subtasks, required_agents).
    Falls back to (whole query as one subtask, every agent) on any
    parsing problem — see module docstring on why over-inclusion is the
    safe default here, not under-inclusion."""
    try:
        json_text = _extract_json(response_text)
        parsed = json.loads(json_text)
        subtasks = [str(s) for s in parsed.get("subtasks", [])]
        required_agents = [a for a in parsed.get("required_agents", []) if a in ALL_AGENT_NAMES]

        if not subtasks or not required_agents:
            raise ValueError("empty subtasks or required_agents")

        # "retrieval" is non-negotiable — see the system prompt's own
        # rule. Enforced here too, in case the model didn't include it.
        if "retrieval" not in required_agents:
            required_agents = ["retrieval"] + required_agents

        return subtasks, required_agents
    except Exception:
        return [user_query], list(ALL_AGENT_NAMES)


def _extract_json(text: str) -> str:
    """Strips a markdown code fence if the model wrapped its JSON in one
    (```json ... ``` is a common pattern even when explicitly asked not
    to), and falls back to finding the first {...} block otherwise."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text