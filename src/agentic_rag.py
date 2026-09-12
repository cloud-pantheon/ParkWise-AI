import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from google.genai import errors

from rag import generate_answer
from retriever import search


PLANNER_MODEL = "gemini-3.5-flash-lite"
MAX_RETRIEVAL_ATTEMPTS = 3

VALID_PARKS = {
    "redwood.pdf",
    "mount_rainier.pdf",
    "rocky_mountain.pdf",
}


@dataclass
class AgentTraceStep:
    stage: str
    detail: str


@dataclass
class AgentResult:
    answer: str
    sources: List[Dict[str, Any]]
    trace: List[AgentTraceStep] = field(default_factory=list)
    rewritten_queries: List[str] = field(default_factory=list)
    attempts: int = 0
    confidence: str = "unknown"


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _call_json(client, prompt: str) -> Dict[str, Any]:
    models = [PLANNER_MODEL, "gemini-3.6-flash", "gemini-3.7-flash"]

    for model in models:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            payload = _extract_json(response.text or "")
            if payload:
                return payload
        except (errors.ServerError, errors.ClientError):
            continue
        except Exception:
            continue

    return {}


def plan_question(question: str, client) -> Dict[str, Any]:
    prompt = f"""
You are the planning agent for ParkWise, a National Park document assistant.
Decide how local document retrieval should be performed.

Available document collections:
- redwood.pdf = Redwood National & State Parks
- mount_rainier.pdf = Mount Rainier National Park
- rocky_mountain.pdf = Rocky Mountain National Park

Return ONLY valid JSON with this exact structure:
{{
  "intent": "short intent label",
  "rewritten_query": "retrieval-optimized query",
  "park_filters": ["zero or more exact filenames from the list above"],
  "needs_comparison": false,
  "reason": "one short explanation"
}}

Rules:
- Preserve the user's meaning.
- If one park is clearly requested, select it.
- If multiple parks are requested, include all of them.
- If no park is specified, use an empty park_filters list.
- Rewrite vague wording into search-friendly park terminology.
- Set needs_comparison true only when the user asks to compare parks/options.

USER QUESTION:
{question}
"""
    plan = _call_json(client, prompt)

    rewritten = str(plan.get("rewritten_query") or question).strip()
    filters = plan.get("park_filters") or []
    if not isinstance(filters, list):
        filters = []
    filters = [f for f in filters if f in VALID_PARKS]

    return {
        "intent": str(plan.get("intent") or "park information"),
        "rewritten_query": rewritten,
        "park_filters": filters,
        "needs_comparison": bool(plan.get("needs_comparison", False)),
        "reason": str(plan.get("reason") or "Prepared a retrieval query."),
    }


def assess_evidence(question: str, results: List[Dict[str, Any]], client) -> Dict[str, Any]:
    if not results:
        return {
            "sufficient": False,
            "confidence": "low",
            "reason": "No document evidence was retrieved.",
            "better_query": question,
        }

    evidence = "\n\n".join(
        f"SOURCE {i}: {r['source']} page {r['page']} score {r['score']:.3f}\n{r['text'][:1200]}"
        for i, r in enumerate(results, start=1)
    )

    prompt = f"""
You are the retrieval evaluator for an Agentic RAG system.
Judge whether the retrieved passages contain enough evidence to answer the user's question accurately.

Return ONLY valid JSON:
{{
  "sufficient": true,
  "confidence": "high|medium|low",
  "reason": "one short explanation",
  "better_query": "a better retrieval query if insufficient, otherwise an empty string"
}}

Be strict: semantic similarity alone is not enough. The passages must actually contain information that can answer the question.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence}
"""
    assessment = _call_json(client, prompt)

    if assessment:
        return {
            "sufficient": bool(assessment.get("sufficient", False)),
            "confidence": str(assessment.get("confidence") or "low"),
            "reason": str(assessment.get("reason") or "Evidence evaluated."),
            "better_query": str(assessment.get("better_query") or "").strip(),
        }

    # Deterministic fallback if the planning model is unavailable.
    best_score = float(results[0].get("score", 0.0))
    sufficient = best_score >= 0.42
    return {
        "sufficient": sufficient,
        "confidence": "medium" if sufficient else "low",
        "reason": "Used retrieval-score fallback because the evaluator was unavailable.",
        "better_query": question if not sufficient else "",
    }


def _deduplicate_results(results: List[Dict[str, Any]], limit: int = 6):
    unique = []
    seen = set()

    for result in sorted(results, key=lambda r: r["score"], reverse=True):
        key = (result["source"], result["page"], result["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
        if len(unique) >= limit:
            break

    return unique


def run_agentic_rag(
    question: str,
    chunks,
    embeddings,
    embedding_model,
    client,
    max_attempts: int = MAX_RETRIEVAL_ATTEMPTS,
) -> AgentResult:
    """Run a small Plan -> Retrieve -> Evaluate -> Retry -> Answer agent loop."""
    trace: List[AgentTraceStep] = []
    rewritten_queries: List[str] = []

    plan = plan_question(question, client)
    current_query = plan["rewritten_query"]
    park_filters: Optional[List[str]] = plan["park_filters"] or None

    trace.append(AgentTraceStep(
        "Plan",
        f"Intent: {plan['intent']}. {plan['reason']}"
    ))
    trace.append(AgentTraceStep(
        "Scope",
        "Parks: " + (", ".join(park_filters) if park_filters else "all available parks")
    ))

    best_results: List[Dict[str, Any]] = []
    final_assessment = {
        "sufficient": False,
        "confidence": "low",
        "reason": "No retrieval attempted.",
        "better_query": "",
    }

    for attempt in range(1, max_attempts + 1):
        rewritten_queries.append(current_query)
        trace.append(AgentTraceStep(
            f"Retrieve {attempt}",
            f'Searching for: "{current_query}"'
        ))

        results = search(
            current_query,
            chunks,
            embeddings,
            embedding_model,
            top_k=6,
            park_filters=park_filters,
            use_auto_park_filter=False,
        )

        if not results and park_filters:
            trace.append(AgentTraceStep(
                "Fallback",
                "No evidence in the selected park scope; retrying across all park documents."
            ))
            park_filters = None
            results = search(
                current_query,
                chunks,
                embeddings,
                embedding_model,
                top_k=6,
                park_filters=None,
                use_auto_park_filter=False,
            )

        # Keep evidence from successful attempts so later retries can improve rather than discard it.
        best_results = _deduplicate_results(best_results + results, limit=8)
        assessment = assess_evidence(question, best_results[:6], client)
        final_assessment = assessment

        trace.append(AgentTraceStep(
            f"Evaluate {attempt}",
            f"{assessment['confidence'].title()} confidence. {assessment['reason']}"
        ))

        if assessment["sufficient"]:
            break

        better_query = assessment.get("better_query", "").strip()
        if not better_query or better_query.lower() == current_query.lower():
            better_query = f"{current_query} rules requirements restrictions details"

        current_query = better_query
        trace.append(AgentTraceStep(
            "Replan",
            "Evidence was insufficient, so the agent rewrote the retrieval query."
        ))

    if not best_results:
        return AgentResult(
            answer="I couldn't find enough information in the available park documents.",
            sources=[],
            trace=trace,
            rewritten_queries=rewritten_queries,
            attempts=len(rewritten_queries),
            confidence="low",
        )

    answer = generate_answer(question, best_results[:6], client)
    trace.append(AgentTraceStep(
        "Answer",
        "Generated the final response from the accumulated document evidence."
    ))

    return AgentResult(
        answer=answer,
        sources=best_results[:6],
        trace=trace,
        rewritten_queries=rewritten_queries,
        attempts=len(rewritten_queries),
        confidence=final_assessment.get("confidence", "unknown"),
    )
