"""Separate useful-answer work from safety and routing overhead without hiding unknown costs."""
from security_cost_helpers import estimate_cost_usd

OVERHEAD_ROLES = {"safety_filter", "router", "cache_validation", "quality_check"}


def summarize_chat(usage, prices):
    buckets = [counts for models in usage.values() for counts in models.values()]
    result = {key: sum(counts.get(key, 0) for counts in buckets)
              for key in ("calls", "failed_calls", "missing_usage_calls", "input", "output", "cached")}
    result["tokens"] = result["input"] + result["output"]
    result["elapsed_seconds"] = round(sum(counts.get("elapsed_seconds", 0) for counts in buckets), 4)
    result["estimated_chat_cost_usd"] = estimate_cost_usd(usage, prices)
    return result


def account_for_costs(result, prices):
    """Recorded tokens are subtotals when a request failed or omitted usage metadata."""
    usage = result["usage"]
    overhead = {role: models for role, models in usage.items() if role in OVERHEAD_ROLES}
    answer_work = {role: models for role, models in usage.items() if role not in OVERHEAD_ROLES}
    report = {
        "by_role": {role: summarize_chat({role: models}, prices) for role, models in usage.items()},
        "total": summarize_chat(usage, prices),
        "overhead": summarize_chat(overhead, prices),
        "answer_work": summarize_chat(answer_work, prices),
        "incomplete_estimates": [],
    }
    for role, models in usage.items():
        for model, counts in models.items():
            if counts["missing_usage_calls"]:
                report["incomplete_estimates"].append(f"{role}/{model}: missing usage for {counts['missing_usage_calls']} call(s)")
            if model not in prices:
                report["incomplete_estimates"].append(f"{role}/{model}: no configured price")
    retrieval = result.get("search_calls", 0)
    cache = result.get("cache_embedding_calls", 0)
    report["embeddings"] = {
        "retrieval_queries": retrieval, "cache_requests": cache,
        "tokens": None if retrieval + cache else 0,
        "estimated_cost_usd": None if retrieval + cache else 0.0,
        "cache_elapsed_seconds": result.get("cache_embedding_seconds", 0.0),
    }
    if retrieval + cache:
        report["incomplete_estimates"].append("Embedding tokens and cost are not measured; chat estimate excludes them")
    total = report["total"]
    result["model_calls"] = total["calls"]
    result["estimated_chat_cost_usd"] = total["estimated_chat_cost_usd"]
    result["estimated_total_cost_usd"] = None if retrieval + cache else total["estimated_chat_cost_usd"]
    result["cost_accounting"] = report
    return result
