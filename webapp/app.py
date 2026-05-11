import os
import json
import hashlib
import math
import re
import time
import threading
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify, render_template, send_from_directory, stream_with_context, Response

from chat_store import ChatStore
from providers import ProviderAPIError, get_provider, GoogleProvider, OpenRouterProvider
from reranker import Reranker
from settings_store import SettingsStore
from vector_store import VectorStore
from zotero_reader import ZoteroReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'public'))

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.errorhandler(Exception)
def handle_unhandled_exception(exc):
    app.logger.exception("Unhandled exception")
    message = public_error_message(exc, 'An unexpected error occurred.')
    return jsonify({"error": message}), 500

settings_store = SettingsStore(app.instance_path)
chat_store = ChatStore(app.instance_path)

zotero = ZoteroReader()
vector_store = VectorStore()
reranker = None

_credential_rate_timestamps = []
_CREDENTIAL_RATE_LIMIT_WINDOW = 60
_CREDENTIAL_RATE_LIMIT_MAX = 20


def _check_credential_rate_limit():
    now = time.time()
    global _credential_rate_timestamps
    _credential_rate_timestamps = [
        t for t in _credential_rate_timestamps
        if now - t < _CREDENTIAL_RATE_LIMIT_WINDOW
    ]
    if len(_credential_rate_timestamps) >= _CREDENTIAL_RATE_LIMIT_MAX:
        return False
    _credential_rate_timestamps.append(now)
    return True


def get_reranker(client=None, chat_model="", reranker_mode="", reranker_model=""):
    global reranker
    config = get_current_config()
    if not reranker_mode:
        reranker_mode = config.get("reranker_mode") or os.environ.get("RESEARCHARR_RERANKER_MODE") or "local"
    if not reranker_model:
        reranker_model = config.get("reranker_model") or ""
    if client is None:
        active_credential = settings_store.get_active_credential()
        if active_credential:
            client = get_provider(active_credential["provider"], active_credential["api_key"])
    if reranker is not None and getattr(reranker, 'mode', None) == reranker_mode and (reranker_mode != 'api' or getattr(reranker._impl, '_reranker_model', '') == reranker_model):
        return reranker
    reranker = Reranker(mode=reranker_mode, client=client, reranker_model=reranker_model)
    return reranker

_model_catalog_cache = {}
_model_catalog_cache_ttl = 300


def _get_cached_model_catalog(client, active_credential):
    provider_key = f"{active_credential.get('provider')}:{active_credential.get('id', '')}"
    now = time.time()
    cached = _model_catalog_cache.get(provider_key)
    if cached and (now - cached.get('_fetched_at', 0)) < _model_catalog_cache_ttl:
        return cached
    catalog = client.fetch_models()
    catalog['_fetched_at'] = now
    _model_catalog_cache[provider_key] = catalog
    return catalog


def default_index_progress_state():
    return {
        "active": False,
        "stage": "idle",
        "run_id": "",
        "started_at": None,
        "provider": "",
        "embeddings_model": "",
        "matched_items": 0,
        "excluded_items": 0,
        "total_items": 0,
        "processed_items": 0,
        "indexed_items": 0,
        "failed_items": 0,
        "current_item_title": "",
        "message": "",
        "percent": 0.0,
    }


index_progress_lock = threading.RLock()
index_progress_state = default_index_progress_state()

PREFERRED_EMBEDDINGS_MODELS = {
    "openrouter": ["openai/text-embedding-3-small", "openai/text-embedding-3-large"],
    "openai": ["text-embedding-3-small", "text-embedding-3-large"],
    "google": ["text-embedding-004"],
}

QUERY_CITATION_INSTRUCTION = (
    "Answer based on the provided context. Cite sources using [N] format. "
    "Use only attachment full-text passages as evidence. Do not cite titles, abstracts, notes, "
    "or bibliography metadata as if they were supporting passages. Paraphrase normally when helpful, "
    "without quotation marks. Use double quotes only when reproducing an exact supporting fragment, "
    "and place that exact fragment immediately before its citation marker, for example: "
    "\"quoted fragment\"[1]. Place each citation marker immediately after the claim or clause it supports. "
    "If multiple sources support the same claim, place their markers consecutively, for example: claim[1][2][3]. "
    "Do not assume any fixed maximum number of citations. "
    "Do not stack citations at the end of a long sentence unless each cited source directly supports that "
"entire sentence. If a specific detail is not in the sources, mention it briefly but do not build "
"the entire response around what is missing. When the user requests a table or structured format "
"and you have partial information, build the table anyway using «—» for missing cells rather than "
"refusing. Do NOT pivot to a different topic just because the "
"sources cover it — if the sources do not address the user's question, say so clearly instead of "
"answering a different question. Focus on delivering a useful, well-structured answer. "
    "When it helps the user, format the answer with Markdown, including ordered or unordered "
    "lists, tables, block quotes, and fenced code blocks."
)

QUERY_SYNTHESIS_INSTRUCTION = (
    "Answer based on the provided context. Synthesize the key ideas, arguments, and findings "
    "from the sources into a coherent, well-structured response. You do not need to anchor each "
    "sentence to a specific text fragment — instead, identify the main ideas the sources support "
    "and present them as an integrated answer. Cite sources using [N] format after each idea or "
    "claim to indicate which sources back it up. If multiple sources converge on the same idea, "
    "group their citations together, for example: idea[1][3][5]. "
    "Use only attachment full-text passages as evidence. Do not cite titles, abstracts, notes, "
    "or bibliography metadata as if they were supporting passages. "
    "Do not assume any fixed maximum number of citations. "
    "If a specific detail is not in the sources, mention it briefly but do not build "
    "the entire response around what is missing. When the user requests a table or structured format "
    "and you have partial information, build the table anyway using \xc2\xab\xe2\x80\x94\xc2\xbb for missing cells rather than "
    "refusing. Do NOT pivot to a different topic just because the "
    "sources cover it \xe2\x80\x94 if the sources do not address the user's question, say so clearly instead of "
    "answering a different question. Focus on delivering a useful, well-structured answer. "
    "When it helps the user, format the answer with Markdown, including ordered or unordered "
    "lists, tables, block quotes, and fenced code blocks."
)

QUERY_PARAPHRASE_INSTRUCTION = (
    "Answer based on the provided context. Reformulate the content in your own words, "
    "maintaining the original meaning and accuracy without reproducing verbatim fragments. "
    "Cite sources using [N] format at the end of the relevant clause or paragraph. "
    "Use only attachment full-text passages as evidence. Do not cite titles, abstracts, notes, "
    "or bibliography metadata as if they were supporting passages. Do not use quotation marks "
    "for cited content; instead, paraphrase and place the citation marker at the end of the paraphrased "
    "claim, for example: paraphrased claim[1]. Place each citation marker immediately after the claim "
    "it supports. If multiple sources support the same claim, place their markers consecutively, "
    "for example: claim[1][2][3]. Do not assume any fixed maximum number of citations. "
    "If a specific detail is not in the sources, mention it briefly but do not build "
    "the entire response around what is missing. When the user requests a table or structured format "
    "and you have partial information, build the table anyway using «—» for missing cells rather than "
    "refusing. Do NOT pivot to a different topic just because the "
    "sources cover it \xe2\x80\x94 if the sources do not address the user's question, say so clearly instead of "
    "answering a different question. Focus on delivering a useful, well-structured answer. "
    "When it helps the user, format the answer with Markdown, "
    "including ordered or unordered "
    "lists, tables, block quotes, and fenced code blocks."
)
ANSWER_VALIDATION_SYSTEM_PROMPT = (
    "You are a quality reviewer for retrieval-augmented answers. Your job is to ensure the draft "
    "answer is accurate, well-cited, and useful. Revise using the numbered source passages in the "
    "context. Keep all claims that are supported or reasonably inferable from the sources. Remove "
    "only claims that are clearly fabricated (not present in any source). Never use outside knowledge. "
    "Do NOT add hedging language, epistemic disclaimers, or meta-sections about certainty levels. "
"Do NOT restructure a fluent answer into a bureaucratic format. Preserve the draft's tone and "
"flow. Your goal is to make the answer more accurate, not more cautious. "
"However, if the draft answer pivots to a different topic than what the user asked about "
"(e.g., the user asks about topic X and the draft answers about topic Y because the sources "
"cover Y but not X), flag this and revise the answer to clearly state that the sources do not "
"address the user's question, rather than letting the draft drift to an unrelated topic."
)
ANSWER_VALIDATION_USER_PROMPT = (
    "Question: {question}\n\n"
    "Context:\n\n{context}\n\n"
    "Draft answer:\n\n{draft_answer}\n\n"
    "Review and improve the answer:\n"
    "- Ensure each claim has at least one [N] citation. Add missing citations if the source is in the context.\n"
    "- Remove only claims that are clearly fabricated (no source supports them at all).\n"
    "- If a claim is reasonably supported or inferable, keep it as-is without adding hedging.\n"
    "- Do NOT add sections like \"Explicit in the sources\" or \"Cautious inference\".\n"
    "- Do NOT add disclaimers about what the sources don't cover unless critically relevant.\n"
"- Keep the answer fluent and natural. Do not make it more bureaucratic than the draft.\n"
"- If the draft answers a different question than what the user asked (topic drift), revise it to "
"clearly state what the sources do not cover instead of pivoting to unrelated content.\n"
"- Only if the sources contain absolutely no relevant information, return exactly: "
    "\"I do not have enough information in the sources to answer this question.\"\n\n"
    "Follow this response-style instruction too:\n{citation_instruction}"
)
SPANISH_QUERY_STOPWORDS = {
    "a", "al", "algo", "alguna", "alguno", "algunos", "ante", "aquel", "aquella", "aquellas",
    "aquello", "aquellos", "asi", "aun", "bajo", "cada", "como", "con", "contra", "cual",
    "cuales", "cuando", "de", "del", "desde", "donde", "dos", "el", "ella", "ellas", "ellos",
    "en", "entre", "era", "eran", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba",
    "estaban", "este", "esto", "estos", "fue", "fueron", "ha", "han", "hasta", "hay", "la",
    "las", "le", "les", "lo", "los", "mas", "más", "mi", "muy", "ni", "no", "o", "para", "pero",
    "por", "que", "qué", "quien", "quienes", "se", "ser", "si", "sin", "sobre", "son", "su",
    "sus", "tambien", "también", "te", "tiene", "tienen", "tu", "un", "una", "uno", "unos",
    "unas", "y", "ya",
}
QUERY_EXPANSION_RULES = {
    "antifranquista": [
        "contraria al franquismo",
        "contraria al régimen",
        "crítica del franquismo",
        "propaganda franquista",
        "triunfalismo del régimen",
        "denuncia social",
    ],
    "franquismo": [
        "régimen franquista",
        "propaganda del régimen",
        "triunfalismo franquista",
    ],
    "fotografía": [
        "fotografías",
        "imagen",
        "imágenes",
        "reportaje gráfico",
    ],
    "fotografías": [
        "fotografía",
        "imagen",
        "imágenes",
        "reportaje gráfico",
    ],
}
ESTIMATED_CHARS_PER_TOKEN = 4.0
DEFAULT_INDEX_SECONDS_PER_ITEM = 0.35
INDEX_CHUNK_SIZE = 500
INDEX_CHUNK_OVERLAP = 200
INDEX_EMBEDDING_BATCH_SIZE = 32
INDEX_DOCUMENT_VERSION = 3


def update_index_progress_state(**updates):
    with index_progress_lock:
        index_progress_state.update(updates)
        total_items = max(int(index_progress_state.get("total_items") or 0), 0)
        processed_items = max(int(index_progress_state.get("processed_items") or 0), 0)
        percent = 0.0

        if total_items > 0:
            percent = round(min(processed_items / total_items, 1) * 100, 1)
        elif index_progress_state.get("active") is False and index_progress_state.get("stage") == "completed":
            percent = 100.0

        index_progress_state["percent"] = percent
        return dict(index_progress_state)


def reset_index_progress_state():
    with index_progress_lock:
        index_progress_state.clear()
        index_progress_state.update(default_index_progress_state())


def get_index_progress_state():
    with index_progress_lock:
        return dict(index_progress_state)


def is_indexing_active():
    with index_progress_lock:
        return bool(index_progress_state.get("active"))


def get_current_config():
    return settings_store.get_config()


def public_error_message(exc, fallback_message):
    if isinstance(exc, (ProviderAPIError, ValueError)):
        return str(exc)

    app.logger.exception(fallback_message)
    return fallback_message


def error_response(exc, fallback_message):
    message = public_error_message(exc, fallback_message)
    if isinstance(exc, ProviderAPIError):
        status_code = exc.status_code if 400 <= exc.status_code < 600 else 502
        return jsonify({"error": message}), status_code
    if isinstance(exc, ValueError):
        return jsonify({"error": message}), 400
    return jsonify({"error": message}), 500


def require_active_provider():
    active_credential = settings_store.get_active_credential()
    if not active_credential:
        raise ValueError("No active API key configured. Save and activate a credential in Settings first.")

    client = get_provider(active_credential["provider"], active_credential["api_key"])
    return active_credential, client


def resolve_embeddings_model(client, current_config, model_catalog=None, provider_name=None):
    provider_name = (provider_name or current_config.get("provider") or "openrouter").lower()
    configured_model = (current_config.get("embeddings_model") or "").strip()

    try:
        if model_catalog is None:
            model_catalog = client.fetch_models()
    except Exception:
        if configured_model:
            return configured_model
        raise ValueError("Embeddings model required. Load models in Settings and choose an embeddings-capable model.")

    embedding_models = []
    for provider in model_catalog.get("providers", {}).values():
        for model in provider.get("models", []):
            model_id = model.get("id")
            if model.get("isEmbedding") and model_id:
                embedding_models.append(model_id)

    if configured_model and configured_model in embedding_models:
        return configured_model

    if not embedding_models:
        raise ValueError(
            "No embedding models are available for the configured provider. Reload models in Settings and choose an embeddings-capable model."
        )

    preferred_models = PREFERRED_EMBEDDINGS_MODELS.get(provider_name, [])
    fallback_model = next((model_id for model_id in preferred_models if model_id in embedding_models), embedding_models[0])
    settings_store.update_config({"embeddings_model": fallback_model})
    return fallback_model


def serialize_source(result):
    metadata = result.get("metadata", {})
    item_id = str(result.get("id", ""))
    zotero_targets = zotero.get_item_open_targets(item_id) if item_id else {}
    document_source = metadata.get("publication") or metadata.get("attachment_label") or metadata.get("section_heading") or metadata.get("content_type") or ""
    return {
        "source_id": metadata.get("key") or metadata.get("doi") or str(result.get("id", "")),
        "item_id": item_id,
        "key": metadata.get("key", ""),
        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", ""),
        "date": metadata.get("date", ""),
        "publication": metadata.get("publication", ""),
        "document_source": document_source,
        "attachment_label": metadata.get("attachment_label", ""),
        "section_heading": metadata.get("section_heading", ""),
        "content_type": metadata.get("content_type", ""),
        "full_reference": build_source_reference(metadata),
        "pages": metadata.get("pages", ""),
        "chunk_index": metadata.get("chunk_index"),
        "score": round(result.get("score", 0), 3),
        "retrieval_score": round(result.get("retrieval_score", 0), 3),
        "matched_queries": [normalize_text(item) for item in result.get("matched_queries", []) if normalize_text(item)],
        "text": result.get("text", "")[:4000],
        "zotero_open_uri": zotero_targets.get("zotero_open_uri", ""),
        "zotero_select_uri": zotero_targets.get("zotero_select_uri", ""),
    }


def build_source_reference(metadata):
    authors = str(metadata.get("authors") or "").strip()
    date = str(metadata.get("date") or "").strip()
    title = str(metadata.get("title") or "Untitled").strip() or "Untitled"
    publication = str(metadata.get("publication") or "").strip()

    lead = authors
    if date:
        lead = f"{lead} ({date})" if lead else date

    reference = f"{lead} — {title}" if lead else title
    if publication:
        reference = f"{reference}. {publication}"
    return reference


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_result_signature(result):
    metadata = result.get("metadata", {}) or {}
    item_id = str(result.get("id", "") or "").strip()
    chunk_index = metadata.get("chunk_index")
    section_heading = normalize_text(metadata.get("section_heading"))
    text = str(result.get("text", "") or "").strip()
    digest_source = f"{item_id}\n{chunk_index}\n{section_heading}\n{text[:400]}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]
    return f"{item_id}:{chunk_index}:{digest}"


def build_query_variants(question: str) -> List[str]:
    base = normalize_text(question)
    if not base:
        return []

    variants = []
    seen = set()

    def add_variant(value):
        normalized = normalize_text(value)
        if len(normalized) < 3:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        variants.append(normalized)

    add_variant(base)
    add_variant(base.replace('"', "").replace("“", "").replace("”", "").replace("'", ""))

    content_tokens = [
        token for token in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9][A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9'\-]{2,}", base)
        if token.casefold() not in SPANISH_QUERY_STOPWORDS
    ]
    if 3 <= len(content_tokens) <= 14:
        add_variant(" ".join(content_tokens))

    lowered = base.casefold()
    for needle, expansions in QUERY_EXPANSION_RULES.items():
        if needle not in lowered:
            continue
        for expansion in expansions:
            add_variant(re.sub(re.escape(needle), expansion, base, flags=re.IGNORECASE))
            if "las hurdes" in lowered:
                add_variant(f"Las Hurdes {expansion}")

    if "las hurdes" in lowered and any(term in lowered for term in ("franqu", "régimen", "regimen")):
        add_variant("Las Hurdes realismo crítico franquismo")
        add_variant("Las Hurdes literatura testimonial franquismo")
        add_variant("Las Hurdes denuncia social propaganda franquista")

    return variants[:6]


def merge_retrieval_result_sets(result_sets, top_k):
    merged = {}

    for variant, results in result_sets:
        for rank, result in enumerate(results):
            signature = build_result_signature(result)
            entry = merged.get(signature)
            if entry is None:
                entry = {
                    "result": dict(result),
                    "rrf_score": 0.0,
                    "best_score": float(result.get("score", 0) or 0),
                    "matched_queries": [],
                }
                merged[signature] = entry

            entry["rrf_score"] += 1.0 / (60 + rank + 1)
            entry["best_score"] = max(entry["best_score"], float(result.get("score", 0) or 0))
            if variant not in entry["matched_queries"]:
                entry["matched_queries"].append(variant)

    ordered = sorted(
        merged.values(),
        key=lambda item: (item["rrf_score"], item["best_score"]),
        reverse=True,
    )

    combined = []
    for entry in ordered[:top_k]:
        result = dict(entry["result"])
        result["score"] = round(entry["rrf_score"], 6)
        result["retrieval_score"] = round(entry["best_score"], 6)
        result["matched_queries"] = entry["matched_queries"][:4]
        combined.append(result)
    return combined


def adaptive_filter_results(query: str, results: List[Dict[str, Any]], limit: Optional[int] = None, max_per_source: int = 5) -> List[Dict[str, Any]]:
    if not results:
        return []

    if limit is None or limit <= 0:
        limit = len(results)
    else:
        limit = min(int(limit), len(results))

    scores = [r.get("score", 0) for r in results]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = variance ** 0.5
    adaptive_threshold = mean_score - 0.5 * std_score

    above_threshold = [r for r in results if r.get("score", 0) >= adaptive_threshold]
    above_threshold.sort(key=lambda r: r.get("score", 0), reverse=True)

    if len(above_threshold) < limit:
        relaxed_threshold = mean_score - 1.0 * std_score
        above_threshold = [r for r in results if r.get("score", 0) >= relaxed_threshold]
        above_threshold.sort(key=lambda r: r.get("score", 0), reverse=True)

    if len(above_threshold) < limit:
        above_threshold = sorted(results, key=lambda r: r.get("score", 0), reverse=True)

    selected = []
    seen_signatures = set()
    source_counts = {}
    preferred_unique_sources = max(1, min(limit, (limit + 1) // 2))

    for result in above_threshold:
        if len(selected) >= preferred_unique_sources:
            break

        metadata = result.get("metadata", {})
        source_id = metadata.get("key") or metadata.get("doi") or str(result.get("id", ""))
        text = str(result.get("text") or "")
        source_signature = (source_id, text[:120], text[-120:])

        if source_signature in seen_signatures:
            continue
        if source_id in source_counts:
            continue

        selected.append(result)
        seen_signatures.add(source_signature)
        source_counts[source_id] = 1

    for result in above_threshold:
        if len(selected) >= limit:
            break

        metadata = result.get("metadata", {})
        source_id = metadata.get("key") or metadata.get("doi") or str(result.get("id", ""))
        text = str(result.get("text") or "")
        source_signature = (source_id, text[:120], text[-120:])

        if source_signature in seen_signatures:
            continue

        current_count = source_counts.get(source_id, 0)
        if source_id and current_count >= max_per_source:
            continue
        if not source_id and len(selected) >= limit:
            continue

        selected.append(result)
        seen_signatures.add(source_signature)
        source_counts[source_id] = current_count + 1

    return selected[:limit]


def select_query_results(results, limit, max_per_source=3):
    if limit <= 0:
        return []

    ordered = sorted(results, key=lambda result: result.get("score", 0), reverse=True)

    selected = []
    seen_sources = set()
    seen_signatures = set()
    source_counts = {}

    for result in ordered:
        if len(selected) >= limit:
            return selected[:limit]

        metadata = result.get("metadata", {})
        source_id = metadata.get("key") or metadata.get("doi") or str(result.get("id", ""))
        source_signature = (source_id, str(result.get("text") or "")[:240])

        if source_signature in seen_signatures:
            continue

        seen_signatures.add(source_signature)

        if source_id and source_id not in seen_sources:
            selected.append(result)
            seen_sources.add(source_id)
            source_counts[source_id] = 1
        else:
            current_count = source_counts.get(source_id, 0)
            if source_id and current_count >= max_per_source:
                continue
            if not source_id and len(selected) >= limit:
                continue
            selected.append(result)
            source_counts[source_id] = current_count + 1

    return selected[:limit]


def build_query_context_block(index, result):
    metadata = result.get("metadata", {})
    parts = [f"[{index + 1}] {build_source_reference(metadata)}"]

    section_heading = str(metadata.get("section_heading") or "").strip()
    if section_heading:
        parts.append(f"Section: {section_heading}")

    pages = normalize_text(metadata.get("pages"))
    if pages:
        parts.append(f"Pages: {pages}")

    passage = str(result.get("text") or "").strip()
    if passage:
        parts.append(f"Passage:\n{passage}")

    return "\n".join(part for part in parts if part).strip()


def build_query_context(results):
    context_parts = [build_query_context_block(index, result) for index, result in enumerate(results)]
    return '\n\n'.join(context_parts) if context_parts else "No relevant documents found."


def build_conversation_messages(conversation_id):
    messages = []
    if not conversation_id:
        return messages

    try:
        conversation_detail = chat_store.get_conversation(conversation_id)
    except (ValueError, Exception):
        return messages

    history = conversation_detail.get("messages", [])
    recent_history = history[-8:] if len(history) > 8 else history
    latest_assistant_message = ""

    for msg in recent_history:
        role = msg.get("role")
        content = normalize_text(msg.get("content", ""))
        if not content:
            continue
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            latest_assistant_message = content

    if latest_assistant_message:
        assistant_summary = re.sub(r"\[[0-9]+\]", "", latest_assistant_message)
        assistant_summary = normalize_text(assistant_summary)
        if len(assistant_summary) > 1200:
            assistant_summary = f"{assistant_summary[:1200].rstrip()}..."
        messages.append({
            "role": "system",
            "content": (
                "Conversation continuity note: the last assistant answer was: "
                f"{assistant_summary}\nUse this only to resolve references in the follow-up. "
                "Do not treat it as evidence and do not preserve unsupported claims from it."
            ),
        })

    return messages


def expand_query_with_llm(client, model: str, question: str) -> List[str]:
    if not model or not question:
        return []
    try:
        result = client.chat_completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate 3 alternative phrasings of the research question below. "
                        "Use different vocabulary and structure but preserve the same intent. "
                        "Output only the reformulations, one per line, no numbering or explanations."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=120,
            temperature=0,
        )
        lines = (result.get("content") or "").split("\n")
        return [normalize_text(l) for l in lines if len(normalize_text(l)) > 8][:3]
    except Exception:
        return []


def detect_query_type(question: str) -> str:
    lowered = question.casefold()
    comparative_markers = [
        "compara", "diferencia", "distintos", "varios", "qué autores", "diferentes autores",
        " vs ", "versus", "perspectivas", "posiciones", "posturas", "corrientes", "debates",
        "compare", "contrast", "various", "different perspectives", "entre ",
    ]
    focused_markers = [
        "quién", "quiénes", "cuándo", "dónde", "qué es ", "define ", "cuál es",
        "fecha de", "año de", "nombre de", "who ", "when ", "where ", "what is ",
    ]
    if any(m in lowered for m in comparative_markers):
        return "comparative"
    if any(m in lowered for m in focused_markers):
        return "focused"
    return "overview"


def trim_by_score_gap(results: List[Dict[str, Any]], min_results: int = 3) -> List[Dict[str, Any]]:
    if len(results) <= min_results:
        return results
    scores = [r.get("score", 0.0) for r in results]
    score_range = scores[0] - scores[-1]
    if score_range < 1e-6:
        return results
    for i in range(min_results, len(results)):
        gap = scores[i - 1] - scores[i]
        if gap >= score_range * 0.35:
            return results[:i]
    return results


def retrieve_query_results(client, embedding_model, question, filter_ids, requested_top_k, chat_model="", reranker_mode="local"):
    question = normalize_text(question)
    rule_variants = build_query_variants(question) or [question]

    llm_variants: List[str] = []
    if chat_model and len(question.split()) >= 5:
        llm_variants = expand_query_with_llm(client, chat_model, question)

    seen: dict = {}
    for v in rule_variants + llm_variants:
        seen.setdefault(v.casefold(), v)
    query_variants = list(seen.values())[:8]

    query_embeddings = client.create_embeddings(embedding_model, query_variants)

    word_count = len(question.split())
    candidate_k = max(word_count * 3, requested_top_k * 4, 18)
    rerank_k = max(word_count * 3, requested_top_k * 4, 16)

    query_type = detect_query_type(question)
    max_per_source = {"focused": 5, "comparative": 2, "overview": 3}[query_type]

    result_sets = []
    for query_variant, query_embedding in zip(query_variants, query_embeddings):
        variant_results = vector_store.hybrid_search(
            query_embedding,
            query_variant,
            top_k=candidate_k,
            min_score=0.0,
            filter_ids=filter_ids,
        )
        result_sets.append((query_variant, variant_results))

    merged_results = merge_retrieval_result_sets(result_sets, top_k=max(candidate_k * 4, 40))
    current_reranker = get_reranker(client=client, chat_model=chat_model, reranker_mode=reranker_mode)
    try:
        reranked_results = current_reranker.rerank(question, merged_results, top_k=rerank_k)
    except Exception:
        app.logger.exception("Reranker failed; using retrieval results without reranking")
        reranked_results = sorted(
            merged_results,
            key=lambda result: result.get("score", 0.0),
            reverse=True,
        )[:rerank_k]
    reranked_results = trim_by_score_gap(reranked_results, min_results=min(3, requested_top_k))
    final_results = adaptive_filter_results(question, reranked_results, limit=requested_top_k, max_per_source=max_per_source)
    return final_results, query_variants


def validate_answer(client, model, question, draft_answer, context, citation_instruction, max_tokens):
    draft_text = normalize_text(draft_answer)
    if not draft_text:
        return draft_answer, {}

    try:
        validation = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": ANSWER_VALIDATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ANSWER_VALIDATION_USER_PROMPT.format(
                        question=question,
                        context=context,
                        draft_answer=draft_answer,
                        citation_instruction=citation_instruction,
                    ),
                },
            ],
            max_tokens=min(max(max_tokens, 512), 4096),
            temperature=0,
        )
    except Exception:
        app.logger.exception("Answer validation failed; returning draft answer")
        return draft_answer, {}

    validated_content = normalize_text(validation.get("content", ""))
    if not validated_content:
        return draft_answer, validation.get("usage", {})

    NO_INFO_MARKER = "no tengo información suficiente"
    draft_has_citations = bool(re.search(r'\[\d+\]', draft_text))
    if draft_has_citations and NO_INFO_MARKER in validated_content.lower():
        app.logger.warning("Validator discarded a cited draft answer; keeping the draft")
        return draft_answer, validation.get("usage", {})

    return validation["content"], validation.get("usage", {})


def merge_usage(*usages):
    merged = {}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
    return merged


def get_citation_instruction(config, request_data=None, conversation_mode=None):
    mode = None
    if request_data:
        mode = (request_data.get("response_mode") or "").strip().lower()
    if not mode and conversation_mode:
        mode = str(conversation_mode).strip().lower()
    if not mode:
        mode = (config.get("response_mode") or "synthesis").strip().lower()
    if mode == "paraphrase":
        return QUERY_PARAPHRASE_INSTRUCTION
    if mode == "literal":
        return QUERY_CITATION_INSTRUCTION
    return QUERY_SYNTHESIS_INSTRUCTION


def parse_optional_int(value, field_name):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def parse_optional_int_list(value, field_name):
    if value in (None, ""):
        return []

    candidates = value if isinstance(value, list) else [value]
    normalized = []
    seen = set()

    for candidate in candidates:
        if candidate in (None, ""):
            continue

        parts = candidate.split(',') if isinstance(candidate, str) else [candidate]
        for part in parts:
            text = str(part).strip()
            if not text:
                continue
            try:
                number = int(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must contain integers.") from exc
            if number in seen:
                continue
            seen.add(number)
            normalized.append(number)

    return normalized


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return default


def normalize_string_list(value):
    if value in (None, ""):
        return []

    candidates = value if isinstance(value, list) else [value]
    values = []
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        if isinstance(candidate, str):
            parts = candidate.split(',')
        else:
            parts = [candidate]
        for part in parts:
            text = str(part).strip()
            if text:
                values.append(text)

    seen = set()
    normalized = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized


def parse_index_filters(data):
    collection_ids = parse_optional_int_list(data.get("collection_ids"), "collection_ids")
    collection_id = parse_optional_int(data.get("collection_id"), "collection_id")

    filters = {
        "collection_id": collection_id,
        "collection_ids": collection_ids,
        "include_subcollections": parse_bool(data.get("include_subcollections"), default=True),
        "date_from": parse_optional_int(data.get("date_from"), "date_from"),
        "date_to": parse_optional_int(data.get("date_to"), "date_to"),
        "item_types": normalize_string_list(data.get("item_types")),
        "tags": normalize_string_list(data.get("tags")),
        "require_pdf_text": parse_bool(data.get("require_pdf_text"), default=False),
        "extract_pdf": parse_bool(data.get("extract_pdf"), default=False),
    }

    if filters["date_from"] and filters["date_to"] and filters["date_from"] > filters["date_to"]:
        raise ValueError("date_from must be less than or equal to date_to.")

    return filters


def summarize_index_filters(filters):
    collection_ids = filters.get("collection_ids") or []
    summary = {
        "collection_id": filters.get("collection_id"),
        "collection_ids": collection_ids,
        "collection_name": "",
        "collection_names": [],
        "collection_count": len(collection_ids),
        "include_subcollections": filters.get("include_subcollections", False),
        "date_from": filters.get("date_from"),
        "date_to": filters.get("date_to"),
        "item_types": filters.get("item_types", []),
        "tags": filters.get("tags", []),
        "require_pdf_text": filters.get("require_pdf_text", False),
        "extract_pdf": filters.get("extract_pdf", False),
    }

    if collection_ids:
        collection_paths = zotero.get_collection_path_map()
        collection_names = [
            collection_paths.get(collection_id) or zotero.get_collection_name(collection_id)
            for collection_id in collection_ids
        ]
        summary["collection_names"] = [name for name in collection_names if name]
        if len(summary["collection_names"]) == 1:
            summary["collection_name"] = summary["collection_names"][0]
    elif filters.get("collection_id"):
        summary["collection_name"] = zotero.get_collection_name(filters["collection_id"])

    return summary


def has_index_scope_filters(filters):
    return any([
        filters.get("collection_ids"),
        filters.get("collection_id"),
        filters.get("date_from"),
        filters.get("date_to"),
        filters.get("item_types"),
        filters.get("tags"),
        filters.get("require_pdf_text"),
    ])


def summarize_current_index(indexed_item_ids):
    summary = zotero.summarize_indexed_collections(list(indexed_item_ids))
    return {
        "collectionCount": summary.get("collection_count", 0),
        "collections": summary.get("collections", []),
        "unfiledItemCount": summary.get("unfiled_item_count", 0),
        "missingItemCount": summary.get("missing_item_count", 0),
        "matchedItemCount": summary.get("matched_item_count", 0),
    }


def filter_collections_tree(nodes, allowed_item_ids):
    allowed_ids = {str(item_id).strip() for item_id in allowed_item_ids if str(item_id).strip()}
    if not allowed_ids:
        return []

    filtered_nodes = []
    for node in nodes:
        children = filter_collections_tree(node.get("children") or [], allowed_ids)
        items = [
            dict(item)
            for item in (node.get("items") or [])
            if str(item.get("id") or "").strip() in allowed_ids
        ]
        if not children and not items:
            continue

        filtered_nodes.append({
            **node,
            "children": children,
            "items": items,
        })

    return filtered_nodes


def conversation_needs_auto_title(conversation_id):
    if not conversation_id:
        return False

    try:
        conversation = chat_store.get_conversation(conversation_id)
    except ValueError:
        return False

    return bool(conversation.get("title_auto", True)) and not (conversation.get("messages") or [])


def generate_conversation_title(client, model, question, answer):
    normalized_question = re.sub(r"\s+", " ", str(question or "")).strip()
    normalized_answer = re.sub(r"\s+", " ", str(answer or "")).strip()
    if not normalized_question:
        return ""

    title_messages = [
        {
            "role": "system",
            "content": (
                "Write a concise title for a research chat. "
                "Return only the title in the same language as the user. "
                "Use 3 to 7 words. No quotes. No markdown. No final punctuation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User question: {normalized_question}\n\n"
                f"Assistant answer summary: {normalized_answer[:1200]}\n\n"
                "Return only the title."
            ),
        },
    ]

    try:
        result = client.chat_completion(
            model=model,
            messages=title_messages,
            max_tokens=24,
            temperature=0.2,
        )
    except Exception:
        return ""

    title = re.sub(r"\s+", " ", str(result.get("content") or "")).strip()
    if not title:
        return ""

    if ":" in title:
        lead, tail = title.split(":", 1)
        if lead.casefold() in {"title", "titulo", "título"}:
            title = tail.strip()

    title = title.strip(" \t\n\r\"'`“”‘’.,;:!?")
    return title[:72].rstrip()


def extract_year(value):
    if not value:
        return None
    match = re.search(r"(?<!\d)(1\d{3}|20\d{2}|21\d{2})(?!\d)", str(value))
    if not match:
        return None
    return int(match.group(1))


def item_matches_index_filters(item, filters):
    if filters.get("item_types"):
        allowed_types = {item_type.casefold() for item_type in filters["item_types"]}
        if str(item.get("type") or "").casefold() not in allowed_types:
            return False

    if filters.get("tags"):
        filter_tags = {tag.casefold() for tag in filters["tags"]}
        item_tags = {tag.casefold() for tag in item.get("tags", [])}
        if not item_tags.intersection(filter_tags):
            return False

    if filters.get("require_pdf_text") and not item.get("hasPdfText"):
        return False

    item_year = extract_year(item.get("date"))
    if filters.get("date_from") is not None:
        if item_year is None or item_year < filters["date_from"]:
            return False
    if filters.get("date_to") is not None:
        if item_year is None or item_year > filters["date_to"]:
            return False

    return True


def estimate_text_tokens(text):
    if not text:
        return 0
    return max(1, math.ceil(len(text) / ESTIMATED_CHARS_PER_TOKEN))


def enrich_chunk(chunk_text, title="", authors="", year=None, section_heading=""):
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if authors:
        parts.append(f"Authors: {authors}")
    if year is not None:
        parts.append(f"Year: {year}")
    if section_heading:
        parts.append(f"Section: {section_heading}")
    prefix = '\n'.join(parts)
    return f"{prefix}\n{chunk_text}" if prefix else chunk_text


def build_chunk_prefix(item):
    parts = []
    if item['title']:
        parts.append(f"Title: {item['title']}")
    if item['creators']:
        creators = ', '.join([creator['name'] for creator in item['creators'] if creator['name']])
        if creators:
            parts.append(f"Authors: {creators}")
    if item['date']:
        parts.append(f"Date: {item['date']}")
    if item['publication']:
        parts.append(f"Publication: {item['publication']}")
    if item['tags']:
        parts.append(f"Tags: {', '.join(item['tags'])}")
    return '\n'.join(parts).strip()


def build_document_fingerprint(item, sections):
    payload = {
        "title": item.get("title", ""),
        "creators": [
            creator.get("name", "")
            for creator in item.get("creators", [])
            if creator.get("name")
        ],
        "date": item.get("date", ""),
        "publication": item.get("publication", ""),
        "type": item.get("type", ""),
        "pages": item.get("pages", ""),
        "url": item.get("URL", ""),
        "doi": item.get("DOI", ""),
        "sections": [
            {
                "heading": section.get("heading", ""),
                "label": section.get("label", ""),
                "content_type": section.get("content_type", ""),
                "text": str(section.get("text") or ""),
            }
            for section in sections
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_document_chunks(sections, item=None):
    item = item or {}
    title = str(item.get("title") or "")
    creators = item.get("creators") or []
    authors = ", ".join(c["name"] for c in creators if c.get("name"))
    year = extract_year(item.get("date"))

    chunks = []
    for section in sections:
        content = str(section.get("text") or "").strip()
        if not content:
            continue

        section_chunks = chunk_text(content, chunk_size=INDEX_CHUNK_SIZE, overlap=INDEX_CHUNK_OVERLAP)
        if not section_chunks:
            section_chunks = [content]

        for chunk in section_chunks:
            chunk_text_value = chunk.strip()
            if len(chunk_text_value) < 50:
                continue

            section_heading = str(section.get("heading") or "")
            enriched_text = enrich_chunk(
                chunk_text_value,
                title=title,
                authors=authors,
                year=year,
                section_heading=section_heading,
            )
            chunks.append({
                "text": chunk_text_value,
                "enriched_text": enriched_text,
                "section_heading": section_heading,
                "attachment_label": str(section.get("label") or ""),
                "content_type": str(section.get("content_type") or ""),
                "estimated_tokens": estimate_text_tokens(enriched_text),
                "embedding_chars": len(enriched_text),
            })

    return chunks


def build_index_document(item, extract_pdf=False):
    sections = []

    for attachment in zotero.get_item_attachment_text_sections(item.get('id'), extract_pdf=extract_pdf):
        label = attachment.get('label') or 'Attachment'
        content_type = (attachment.get('content_type') or '').strip()
        heading = f"Attachment: {label}"
        if content_type and content_type.casefold() not in label.casefold():
            heading = f"{heading} ({content_type})"
        sections.append({
            "heading": heading,
            "label": label,
            "content_type": content_type,
            "text": attachment.get('text', ''),
        })

    chunks = build_document_chunks(sections, item=item)
    fingerprint = build_document_fingerprint(item, sections) if sections else ""

    return {
        "chunks": chunks,
        "eligible": len(chunks) > 0,
        "estimated_tokens": sum(chunk["estimated_tokens"] for chunk in chunks),
        "embedding_chars": sum(chunk["embedding_chars"] for chunk in chunks),
        "embedding_calls": math.ceil(len(chunks) / INDEX_EMBEDDING_BATCH_SIZE) if chunks else 0,
        "fingerprint": fingerprint,
    }


def create_document_embeddings(client, model, document):
    chunks = list(document.get("chunks") or [])
    if not chunks:
        return []

    embeddings = []
    for start in range(0, len(chunks), INDEX_EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + INDEX_EMBEDDING_BATCH_SIZE]
        batch_embeddings = client.create_embeddings(
            model,
            [chunk["enriched_text"] for chunk in batch],
        )
        if len(batch_embeddings) != len(batch):
            raise ProviderAPIError(
                f"Embedding provider returned {len(batch_embeddings)} vectors for {len(batch)} chunks.",
                502,
            )
        embeddings.extend(batch_embeddings)

    return embeddings


def build_sync_config(embeddings_model):
    return {
        "embeddings_model": (embeddings_model or "").strip(),
        "chunk_size": INDEX_CHUNK_SIZE,
        "chunk_overlap": INDEX_CHUNK_OVERLAP,
        "document_version": INDEX_DOCUMENT_VERSION,
    }


def build_scope_key(filters):
    payload = {
        "collection_ids": sorted(filters.get("collection_ids", [])),
        "collection_id": filters.get("collection_id"),
        "include_subcollections": bool(filters.get("include_subcollections", False)),
        "date_from": filters.get("date_from"),
        "date_to": filters.get("date_to"),
        "item_types": sorted(filters.get("item_types", []), key=lambda value: value.casefold()),
        "tags": sorted(filters.get("tags", []), key=lambda value: value.casefold()),
        "require_pdf_text": bool(filters.get("require_pdf_text", False)),
        "extract_pdf": bool(filters.get("extract_pdf", False)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_item_sync_state(document, sync_config):
    return {
        "document_fingerprint": document.get("fingerprint", ""),
        "embeddings_model": sync_config["embeddings_model"],
        "chunk_size": sync_config["chunk_size"],
        "chunk_overlap": sync_config["chunk_overlap"],
        "document_version": sync_config["document_version"],
        "chunk_count": len(document.get("chunks", [])),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def classify_index_candidate(item_id, document, sync_state, indexed_ids, sync_config):
    is_indexed = item_id in indexed_ids
    if not is_indexed and not sync_state:
        return "new"

    if not is_indexed or not sync_state:
        return "changed"

    if sync_state.get("document_fingerprint") != document.get("fingerprint"):
        return "changed"

    if (sync_state.get("embeddings_model") or "") != sync_config["embeddings_model"]:
        return "changed"

    if int(sync_state.get("chunk_size") or 0) != sync_config["chunk_size"]:
        return "changed"

    if int(sync_state.get("chunk_overlap") or 0) != sync_config["chunk_overlap"]:
        return "changed"

    if int(sync_state.get("document_version") or 0) != sync_config["document_version"]:
        return "changed"

    return "unchanged"


def collect_index_candidates(filters, refresh=False, sync_config=None):
    if refresh:
        zotero.refresh()

    if sync_config is None:
        sync_config = build_sync_config("")

    items = zotero.get_items(
        collection_ids=filters.get("collection_ids"),
        collection_id=filters.get("collection_id"),
        include_subcollections=filters.get("include_subcollections", False),
    )
    matched_items = [item for item in items if item_matches_index_filters(item, filters)]
    indexed_ids = vector_store.get_indexed_item_ids()
    item_sync_states = vector_store.get_item_sync_states()
    scope_key = build_scope_key(filters)

    previous_scope_state = vector_store.get_scope_state(scope_key) or {}
    previous_scope_item_ids = {
        str(item_id).strip()
        for item_id in previous_scope_state.get("eligible_item_ids", [])
        if str(item_id).strip()
    }
    if not previous_scope_item_ids and not has_index_scope_filters(filters):
        previous_scope_item_ids = set(indexed_ids)

    candidates = []
    excluded_items = []
    total_tokens = 0
    total_chars = 0
    total_embedding_calls = 0
    eligible_item_ids = []
    unchanged_item_ids = []
    new_items = 0
    changed_items = 0

    extract_pdf = filters.get("extract_pdf", False)
    for item in matched_items:
        document = build_index_document(item, extract_pdf=extract_pdf)
        item_id = str(item.get('id'))
        if not document["eligible"]:
            excluded_items.append({
                "item_id": item_id,
                "title": item.get("title") or "Untitled",
                "reason": "insufficient_text",
            })
            continue

        eligible_item_ids.append(item_id)

        sync_action = classify_index_candidate(
            item_id,
            document,
            item_sync_states.get(item_id),
            indexed_ids,
            sync_config,
        )
        if sync_action == "unchanged":
            unchanged_item_ids.append(item_id)
            continue

        if sync_action == "new":
            new_items += 1
        else:
            changed_items += 1

        total_tokens += document["estimated_tokens"]
        total_chars += document["embedding_chars"]
        total_embedding_calls += document["embedding_calls"]
        candidates.append({
            "item": item,
            "document": document,
            "sync_action": sync_action,
            "was_indexed": item_id in indexed_ids,
        })

    orphaned_removed_item_ids = vector_store.predict_orphaned_scope_removals(scope_key, eligible_item_ids)

    return {
        "matched_items": matched_items,
        "candidates": candidates,
        "excluded_items": excluded_items,
        "estimated_tokens": total_tokens,
        "estimated_chars": total_chars,
        "estimated_embedding_calls": total_embedding_calls,
        "scope_key": scope_key,
        "previous_scope_item_ids": sorted(previous_scope_item_ids),
        "current_scope_item_ids": eligible_item_ids,
        "removed_scope_item_ids": sorted(previous_scope_item_ids - set(eligible_item_ids)),
        "orphaned_removed_item_ids": orphaned_removed_item_ids,
        "eligible_items_total": len(eligible_item_ids),
        "new_items": new_items,
        "changed_items": changed_items,
        "unchanged_items": len(unchanged_item_ids),
        "unchanged_item_ids": unchanged_item_ids,
    }


def find_model_details(model_catalog, model_id):
    if not model_catalog or not model_id:
        return None

    for provider in model_catalog.get("providers", {}).values():
        for model in provider.get("models", []):
            if model.get("id") == model_id:
                return model
    return None


def estimate_cost_from_model(model_details, estimated_tokens):
    if not model_details:
        return None

    prompt_price = model_details.get("promptPrice")
    if prompt_price in (None, 0):
        return None

    return round(float(prompt_price) * estimated_tokens, 6)


def get_index_model_context(current_config):
    context = {
        "provider": current_config.get("provider") or "",
        "embeddings_model": (current_config.get("embeddings_model") or "").strip(),
        "model_details": None,
        "model_catalog": None,
        "warning": "",
    }

    try:
        active_credential, client = require_active_provider()
        model_catalog = client.fetch_models()
        embeddings_model = resolve_embeddings_model(
            client,
            current_config,
            model_catalog=model_catalog,
            provider_name=active_credential.get("provider"),
        )
        context.update({
            "provider": active_credential.get("provider") or context["provider"],
            "embeddings_model": embeddings_model,
            "model_catalog": model_catalog,
            "model_details": find_model_details(model_catalog, embeddings_model),
        })
    except Exception as exc:
        context["warning"] = public_error_message(exc, "Unable to load embedding model details.")

    return context


def build_index_preview(filters, current_config, refresh=False):
    model_context = get_index_model_context(current_config)
    sync_config = build_sync_config(model_context.get("embeddings_model") or current_config.get("embeddings_model") or "")
    candidate_bundle = collect_index_candidates(filters, refresh=refresh, sync_config=sync_config)
    matched_items = candidate_bundle["matched_items"]
    candidates = candidate_bundle["candidates"]
    excluded_items = candidate_bundle["excluded_items"]
    estimated_cost = estimate_cost_from_model(model_context.get("model_details"), candidate_bundle["estimated_tokens"])

    seconds_per_item = vector_store.get_average_seconds_per_item() or DEFAULT_INDEX_SECONDS_PER_ITEM
    estimate_source = "history" if vector_store.get_average_seconds_per_item() else "default"

    warning_messages = []
    if model_context.get("warning"):
        warning_messages.append(model_context["warning"])
    elif estimated_cost is None:
        warning_messages.append("Cost estimate unavailable for the selected embeddings model.")

    return {
        "filters": summarize_index_filters(filters),
        "matched_items": len(matched_items),
        "eligible_items": candidate_bundle["eligible_items_total"],
        "excluded_items": len(excluded_items),
        "already_indexed_items": candidate_bundle["changed_items"] + candidate_bundle["unchanged_items"],
        "scheduled_items": len(candidates),
        "new_items": candidate_bundle["new_items"],
        "changed_items": candidate_bundle["changed_items"],
        "unchanged_items": candidate_bundle["unchanged_items"],
        "removed_items": len(candidate_bundle["orphaned_removed_item_ids"]),
        "estimated_embedding_calls": candidate_bundle["estimated_embedding_calls"],
        "estimated_tokens": candidate_bundle["estimated_tokens"],
        "estimated_characters": candidate_bundle["estimated_chars"],
        "estimated_cost_usd": estimated_cost,
        "pricing_available": estimated_cost is not None,
        "estimated_duration_seconds": round(len(candidates) * seconds_per_item, 1),
        "estimated_duration_source": estimate_source,
        "embeddings_model": model_context.get("embeddings_model") or "",
        "provider": model_context.get("provider") or "",
        "warnings": warning_messages,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/assets/<path:filename>')
def public_asset(filename):
    return send_from_directory(PUBLIC_DIR, filename)


@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        return jsonify(settings_store.get_public_state())

    try:
        data = request.get_json(silent=True) or {}
        settings_store.update_config(data)
        return jsonify(settings_store.get_public_state())
    except Exception as exc:
        return error_response(exc, 'Unable to save settings.')


@app.route('/api/chat/workspace')
def chat_workspace():
    try:
        return jsonify(chat_store.get_workspace_state())
    except Exception as exc:
        return error_response(exc, 'Unable to load the chat workspace.')


@app.route('/api/context-filter', methods=['GET', 'PUT'])
def handle_context_filter():
    if request.method == 'GET':
        try:
            return jsonify(chat_store.get_context_filter())
        except Exception as exc:
            return error_response(exc, 'Unable to load context filter.')

    try:
        data = request.get_json(silent=True) or {}
        result = chat_store.update_context_filter(data.get('selected_item_ids'))
        return jsonify(result)
    except Exception as exc:
        return error_response(exc, 'Unable to save context filter.')


@app.route('/api/chat/conversations', methods=['POST'])
def create_chat_conversation():
    try:
        data = request.get_json(silent=True) or {}
        conversation = chat_store.create_conversation(
            title=data.get('title', ''),
            selected_item_ids=data.get('selected_item_ids'),
        )
        return jsonify(conversation), 201
    except Exception as exc:
        return error_response(exc, 'Unable to create the conversation.')


@app.route('/api/chat/conversations/<conversation_id>', methods=['GET', 'PUT', 'DELETE'])
def handle_chat_conversation(conversation_id):
    try:
        if request.method == 'GET':
            return jsonify(chat_store.get_conversation(conversation_id))

        if request.method == 'DELETE':
            chat_store.delete_conversation(conversation_id)
            return jsonify({'success': True, 'trashed': True})

        data = request.get_json(silent=True) or {}
        return jsonify(chat_store.update_conversation(conversation_id, data))
    except Exception as exc:
        action = 'load' if request.method == 'GET' else ('delete' if request.method == 'DELETE' else 'update')
        return error_response(exc, f'Unable to {action} the conversation.')


@app.route('/api/chat/trash', methods=['GET'])
def list_trashed_conversations():
    try:
        return jsonify(chat_store.get_trashed_conversations())
    except Exception as exc:
        return error_response(exc, 'Unable to load trashed conversations.')


@app.route('/api/chat/trash/empty', methods=['POST'])
def empty_trash():
    try:
        chat_store.empty_trash()
        return jsonify({'success': True})
    except Exception as exc:
        return error_response(exc, 'Unable to empty trash.')


@app.route('/api/chat/trash/<conversation_id>', methods=['DELETE'])
def permanently_delete_trashed_conversation(conversation_id):
    try:
        chat_store.permanently_delete_conversation(conversation_id)
        return jsonify({'success': True})
    except Exception as exc:
        return error_response(exc, 'Unable to permanently delete conversation.')


@app.route('/api/chat/trash/<conversation_id>/restore', methods=['POST'])
def restore_trashed_conversation(conversation_id):
    try:
        summary = chat_store.restore_conversation(conversation_id)
        return jsonify(summary)
    except Exception as exc:
        return error_response(exc, 'Unable to restore conversation.')


@app.route('/api/chat/conversations/<conversation_id>/messages/<message_id>/pin', methods=['POST'])
def toggle_chat_message_pin(conversation_id, message_id):
    try:
        data = request.get_json(silent=True) or {}
        pinned = data.get('pinned')
        return jsonify(chat_store.toggle_message_pin(conversation_id, message_id, pinned=pinned))
    except Exception as exc:
        return error_response(exc, 'Unable to update the pinned state.')


@app.route('/api/chat/conversations/<conversation_id>/export', methods=['POST'])
def export_chat_conversation(conversation_id):
    try:
        return jsonify(chat_store.export_conversation_notes(conversation_id))
    except Exception as exc:
        return error_response(exc, 'Unable to export the conversation notes.')


@app.route('/api/chat/prompts', methods=['POST'])
def create_prompt():
    try:
        data = request.get_json(silent=True) or {}
        prompt = chat_store.create_prompt(
            title=data.get('title', ''),
            text=data.get('text', ''),
            length=data.get('length', 'default'),
        )
        return jsonify(prompt), 201
    except Exception as exc:
        return error_response(exc, 'Unable to save the prompt.')


@app.route('/api/chat/prompts/<prompt_id>', methods=['PUT', 'DELETE'])
def handle_prompt(prompt_id):
    try:
        if request.method == 'DELETE':
            chat_store.delete_prompt(prompt_id)
            return jsonify({'success': True})

        data = request.get_json(silent=True) or {}
        return jsonify(chat_store.update_prompt(prompt_id, data))
    except Exception as exc:
        action = 'delete' if request.method == 'DELETE' else 'update'
        return error_response(exc, f'Unable to {action} the prompt.')


@app.route('/api/credentials', methods=['GET', 'POST'])
def handle_credentials():
    if request.method == 'GET':
        state = settings_store.get_public_state()
        return jsonify({
            'credentials': state.get('credentials', []),
            'active_credential_id': state.get('active_credential_id'),
        })

    try:
        if not _check_credential_rate_limit():
            return jsonify({"error": "Too many requests. Please try again later."}), 429
        data = request.get_json(silent=True) or {}
        credential = settings_store.create_credential(
            provider=data.get('provider'),
            label=data.get('label', ''),
            api_key=data.get('api_key', ''),
            activate=bool(data.get('activate', True)),
        )
        return jsonify({
            'credential': credential,
            'config': settings_store.get_public_state(),
        }), 201
    except Exception as exc:
        return error_response(exc, 'Unable to save API key.')


@app.route('/api/credentials/<credential_id>', methods=['PUT', 'DELETE'])
def handle_credential(credential_id):
    try:
        if not _check_credential_rate_limit():
            return jsonify({"error": "Too many requests. Please try again later."}), 429
        if request.method == 'DELETE':
            settings_store.delete_credential(credential_id)
            return jsonify({
                'success': True,
                'config': settings_store.get_public_state(),
            })

        data = request.get_json(silent=True) or {}
        credential = settings_store.update_credential(credential_id, data)
        return jsonify({
            'credential': credential,
            'config': settings_store.get_public_state(),
        })
    except Exception as exc:
        action = 'delete' if request.method == 'DELETE' else 'update'
        return error_response(exc, f'Unable to {action} API key.')


@app.route('/api/credentials/<credential_id>/activate', methods=['POST'])
def activate_credential(credential_id):
    try:
        if not _check_credential_rate_limit():
            return jsonify({"error": "Too many requests. Please try again later."}), 429
        credential = settings_store.activate_credential(credential_id)
        return jsonify({
            'success': True,
            'credential': credential,
            'config': settings_store.get_public_state(),
        })
    except Exception as exc:
        return error_response(exc, 'Unable to activate API key.')


@app.route('/api/mcp/status')
def get_mcp_status():
    try:
        from mcp_server import get_server_status
        status = get_server_status()
        mcp_config = settings_store.get_mcp_config()
        keys = settings_store.list_mcp_keys()
        return jsonify({
            'server': status,
            'config': mcp_config,
            'has_keys': len(keys) > 0,
            'key_count': len(keys),
        })
    except Exception as exc:
        return jsonify({
            'server': {'status': 'unavailable', 'host': '127.0.0.1', 'port': 5001, 'local_ip': '127.0.0.1'},
            'config': settings_store.get_mcp_config(),
            'has_keys': len(settings_store.list_mcp_keys()) > 0,
            'key_count': len(settings_store.list_mcp_keys()),
        })


@app.route('/api/mcp/config', methods=['GET', 'PUT'])
def handle_mcp_config():
    if request.method == 'GET':
        return jsonify(settings_store.get_mcp_config())

    try:
        data = request.get_json(silent=True) or {}
        config = settings_store.update_mcp_config(data)
        return jsonify(config)
    except Exception as exc:
        return error_response(exc, 'Unable to save MCP configuration.')


@app.route('/api/mcp/keys', methods=['GET', 'POST'])
def handle_mcp_keys():
    if request.method == 'GET':
        return jsonify({'keys': settings_store.list_mcp_keys()})

    try:
        data = request.get_json(silent=True) or {}
        key_data = settings_store.create_mcp_key(label=data.get('label', ''))
        return jsonify(key_data), 201
    except Exception as exc:
        return error_response(exc, 'Unable to create MCP key.')


@app.route('/api/mcp/keys/<key_id>', methods=['DELETE'])
def delete_mcp_key(key_id):
    try:
        settings_store.delete_mcp_key(key_id)
        return jsonify({'success': True})
    except Exception as exc:
        return error_response(exc, 'Unable to delete MCP key.')


@app.route('/api/models')
def get_models():
    try:
        active_credential, client = require_active_provider()
        _model_catalog_cache.pop(f"{active_credential.get('provider')}:{active_credential.get('id', '')}", None)
        catalog = client.fetch_models()
        return jsonify(catalog)
    except Exception as exc:
        return error_response(exc, 'Unable to load models.')


@app.route('/api/rerank-models')
def get_rerank_models():
    try:
        active_credential, client = require_active_provider()
        if not isinstance(client, OpenRouterProvider):
            return jsonify({"providers": {}, "note": "Rerank models are only available via OpenRouter."})
        providers = client.fetch_rerank_models()
        return jsonify({"providers": providers})
    except Exception as exc:
        return error_response(exc, 'Unable to load rerank models.')


@app.route('/api/library/stats')
def get_library_stats():
    try:
        return jsonify({
            "totalItems": zotero.get_item_count(),
            "indexedItems": vector_store.get_stats()['count'],
        })
    except Exception as exc:
        return error_response(exc, 'Unable to load library stats.')


@app.route('/api/library/collections')
def get_collections():
    try:
        return jsonify(zotero.get_collections())
    except Exception as exc:
        return error_response(exc, 'Unable to load collections.')


@app.route('/api/library/items')
def get_items():
    try:
        collection_id = request.args.get('collection_id', type=int)
        limit = request.args.get('limit', 100, type=int)
        items = zotero.get_items(limit=limit, collection_id=collection_id)
        return jsonify(items)
    except Exception as exc:
        return error_response(exc, 'Unable to load items.')


@app.route('/api/library/collections-tree')
def get_collections_tree():
    try:
        tree = zotero.get_collections_tree()
        if parse_bool(request.args.get('indexed_only'), default=False):
            tree = filter_collections_tree(tree, vector_store.get_indexed_item_ids())
        return jsonify(tree)
    except Exception as exc:
        return error_response(exc, 'Unable to load the collections tree.')


@app.route('/api/index/options')
def get_index_options():
    try:
        return jsonify({
            "item_types": zotero.get_item_type_counts(),
            "tags": zotero.get_tag_counts(),
        })
    except Exception as exc:
        return error_response(exc, 'Unable to load indexing options.')


@app.route('/api/index/preview', methods=['POST'])
def preview_index():
    try:
        data = request.get_json(silent=True) or {}
        filters = parse_index_filters(data)
        current_config = get_current_config()
        return jsonify(build_index_preview(filters, current_config, refresh=False))
    except Exception as exc:
        return error_response(exc, 'Unable to prepare indexing preview.')


@app.route('/api/index/status')
def get_index_status():
    try:
        progress = get_index_progress_state()
        index_status = vector_store.get_index_status()
        current_index = dict(index_status.get("currentIndex") or {})
        if index_status.get("indexedCount") and not progress.get("active"):
            if "collectionCount" not in current_index or "collections" not in current_index:
                current_index.update(summarize_current_index(vector_store.get_indexed_item_ids()))
                index_status["currentIndex"] = current_index

        return jsonify({
            "index": index_status,
            "history": vector_store.get_history(limit=10),
            "progress": progress,
        })
    except Exception as exc:
        return error_response(exc, 'Unable to load indexing status.')


@app.route('/api/index/current', methods=['DELETE'])
def clear_current_index():
    try:
        if is_indexing_active():
            return jsonify({"error": "An indexing run is already in progress."}), 409

        cleared = vector_store.get_indexed_item_count()
        vector_store.clear()
        reset_index_progress_state()
        return jsonify({
            "success": True,
            "cleared": cleared,
        })
    except Exception as exc:
        return error_response(exc, 'Unable to clear the current index.')


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks, respecting sentence boundaries where possible."""
    text = text.strip()
    if not text:
        return []

    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p for p in (p.strip() for p in parts) if p]

    if len(parts) <= 2:
        words = text.split()
        if len(words) <= chunk_size:
            return [text] if words else []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(' '.join(words[start:end]))
            if end >= len(words):
                break
            start += chunk_size - overlap
        return chunks

    chunks = []
    current_parts = []
    current_count = 0

    for part in parts:
        part_count = len(part.split())

        if current_parts and current_count + part_count > chunk_size:
            chunks.append(' '.join(current_parts))

            overlap_parts = []
            overlap_count = 0
            for prev_part in reversed(current_parts):
                prev_count = len(prev_part.split())
                if overlap_count + prev_count > overlap:
                    break
                overlap_parts.insert(0, prev_part)
                overlap_count += prev_count

            current_parts = overlap_parts + [part]
            current_count = overlap_count + part_count
        else:
            current_parts.append(part)
            current_count += part_count

    if current_parts:
        chunks.append(' '.join(current_parts))

    return chunks if chunks else [text]


@app.route('/api/index', methods=['POST'])
def index_library():
    if is_indexing_active():
        return jsonify({"error": "An indexing run is already in progress."}), 409

    data = request.get_json(silent=True) or {}
    filters = parse_index_filters(data)
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    run_id = str(uuid.uuid4())
    run_summary = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": 0,
        "success": False,
        "provider": "",
        "embeddings_model": "",
        "scope": "partial" if has_index_scope_filters(filters) else "full",
        "filters": summarize_index_filters(filters),
        "matched_items": 0,
        "eligible_items": 0,
        "scheduled_items": 0,
        "excluded_items": 0,
        "already_indexed_items": 0,
        "new_items": 0,
        "changed_items": 0,
        "unchanged_items": 0,
        "removed_existing_items": 0,
        "removed_items": 0,
        "removed_scope_items": 0,
        "indexed_items": 0,
        "failed_items": 0,
        "indexed_chunks": 0,
        "estimated_embedding_calls": 0,
        "estimated_tokens": 0,
        "estimated_cost_usd": None,
        "pricing_available": False,
        "store_count_after": vector_store.get_stats()['count'],
        "failure_samples": [],
        "exclusion_samples": [],
        "error": "",
    }

    update_index_progress_state(
        active=True,
        stage="preparing",
        run_id=run_id,
        started_at=started_at,
        provider="",
        embeddings_model="",
        matched_items=0,
        excluded_items=0,
        total_items=0,
        processed_items=0,
        indexed_items=0,
        failed_items=0,
        current_item_title="",
        message="Preparing items for indexing...",
    )

    try:
        current_config = get_current_config()
        active_credential, client = require_active_provider()
        model_catalog = client.fetch_models()
        embedding_model = resolve_embeddings_model(
            client,
            current_config,
            model_catalog=model_catalog,
            provider_name=active_credential.get('provider'),
        )
        model_details = find_model_details(model_catalog, embedding_model)
        sync_config = build_sync_config(embedding_model)

        run_summary["provider"] = active_credential.get('provider') or ''
        run_summary["embeddings_model"] = embedding_model
        update_index_progress_state(
            provider=run_summary["provider"],
            embeddings_model=embedding_model,
            message="Preparing items for indexing...",
        )

        candidate_bundle = collect_index_candidates(filters, refresh=True, sync_config=sync_config)
        matched_items = candidate_bundle["matched_items"]
        candidates = candidate_bundle["candidates"]
        excluded_items = candidate_bundle["excluded_items"]
        unchanged_item_ids = set(candidate_bundle["unchanged_item_ids"])
        successful_item_ids = set()
        retained_failed_existing_item_ids = set()

        run_summary["matched_items"] = len(matched_items)
        run_summary["eligible_items"] = candidate_bundle["eligible_items_total"]
        run_summary["scheduled_items"] = len(candidates)
        run_summary["excluded_items"] = len(excluded_items)
        run_summary["already_indexed_items"] = candidate_bundle["changed_items"] + candidate_bundle["unchanged_items"]
        run_summary["new_items"] = candidate_bundle["new_items"]
        run_summary["changed_items"] = candidate_bundle["changed_items"]
        run_summary["unchanged_items"] = candidate_bundle["unchanged_items"]
        run_summary["removed_scope_items"] = len(candidate_bundle["removed_scope_item_ids"])
        run_summary["estimated_embedding_calls"] = candidate_bundle["estimated_embedding_calls"]
        run_summary["estimated_tokens"] = candidate_bundle["estimated_tokens"]
        run_summary["estimated_cost_usd"] = estimate_cost_from_model(model_details, candidate_bundle["estimated_tokens"])
        run_summary["pricing_available"] = run_summary["estimated_cost_usd"] is not None
        run_summary["exclusion_samples"] = excluded_items[:8]
        update_index_progress_state(
            stage="indexing" if candidates else "completed",
            matched_items=run_summary["matched_items"],
            excluded_items=run_summary["excluded_items"],
            total_items=run_summary["scheduled_items"],
            message=(
                "Selected scope is already in sync."
                if not candidates and len(candidate_bundle["orphaned_removed_item_ids"]) == 0
                else "Removing items that left the selected scope..."
                if not candidates
                else "Generating embeddings..."
            ),
        )

        for entry in candidates:
            item = entry["item"]
            document = entry["document"]
            item_id = str(item['id'])
            update_index_progress_state(
                current_item_title=item.get('title') or 'Untitled',
                message=(
                    f"Processing item {run_summary['indexed_items'] + run_summary['failed_items'] + 1} "
                    f"of {run_summary['scheduled_items']}"
                ),
            )
            try:
                chunk_embeddings = create_document_embeddings(client, embedding_model, document)

                base_metadata = {
                    "title": item['title'],
                    "authors": ', '.join([creator['name'] for creator in item['creators'] if creator['name']]),
                    "date": item['date'],
                    "publication": item['publication'],
                    "type": item['type'],
                    "key": item['key'],
                    "url": item['URL'],
                    "doi": item['DOI'],
                    "pages": item['pages'],
                    "collections": item['collections'],
                    "collectionIds": item['collectionIds'],
                }

                vector_store.remove_item_ids([item_id])

                for chunk_index, (chunk, embedding) in enumerate(zip(document["chunks"], chunk_embeddings)):
                    metadata = dict(base_metadata)
                    metadata["chunk_index"] = chunk_index
                    metadata["section_heading"] = chunk.get("section_heading", "")
                    metadata["attachment_label"] = chunk.get("attachment_label", "")
                    metadata["content_type"] = chunk.get("content_type", "")
                    metadata["document_version"] = sync_config["document_version"]
                    vector_store.add_item(
                        item_id=item_id,
                        text=chunk["text"],
                        metadata=metadata,
                        embedding=embedding,
                    )

                vector_store.set_item_sync_state(item_id, build_item_sync_state(document, sync_config))

                run_summary["indexed_items"] += 1
                run_summary["indexed_chunks"] += len(document["chunks"])
                successful_item_ids.add(item_id)
            except Exception as exc:
                run_summary["failed_items"] += 1
                if entry.get("was_indexed"):
                    retained_failed_existing_item_ids.add(item_id)
                if len(run_summary["failure_samples"]) < 8:
                    run_summary["failure_samples"].append({
                        "item_id": item_id,
                        "title": item.get('title') or 'Untitled',
                        "reason": public_error_message(exc, 'Unable to index item.'),
                    })
                app.logger.warning(
                    "Error indexing item %s: %s",
                    item['id'],
                    public_error_message(exc, 'Unable to index item.'),
                )

            update_index_progress_state(
                processed_items=run_summary["indexed_items"] + run_summary["failed_items"],
                indexed_items=run_summary["indexed_items"],
                failed_items=run_summary["failed_items"],
            )

        final_scope_item_ids = set(unchanged_item_ids)
        final_scope_item_ids.update(successful_item_ids)
        final_scope_item_ids.update(retained_failed_existing_item_ids)
        orphaned_removed_item_ids = vector_store.predict_orphaned_scope_removals(
            candidate_bundle["scope_key"],
            sorted(final_scope_item_ids),
        )
        run_summary["removed_items"] = vector_store.remove_item_ids(orphaned_removed_item_ids)
        run_summary["removed_existing_items"] = run_summary["removed_items"]
        vector_store.set_scope_state(
            candidate_bundle["scope_key"],
            summarize_index_filters(filters),
            sorted(final_scope_item_ids),
        )

        vector_store.save()

        run_summary["success"] = True
        run_summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_summary["duration_seconds"] = round(time.perf_counter() - started_clock, 2)
        run_summary["store_count_after"] = vector_store.get_stats()['count']
        run_summary["current_index"] = summarize_current_index(vector_store.get_indexed_item_ids())
        vector_store.record_run(run_summary)
        update_index_progress_state(
            active=False,
            stage="completed",
            processed_items=run_summary["indexed_items"] + run_summary["failed_items"],
            indexed_items=run_summary["indexed_items"],
            failed_items=run_summary["failed_items"],
            current_item_title="",
            message="Indexing completed.",
        )

        return jsonify({
            "success": True,
            "indexed": run_summary["indexed_items"],
            "scheduled": run_summary["scheduled_items"],
            "total": run_summary["matched_items"],
            "eligible": run_summary["eligible_items"],
            "excluded": run_summary["excluded_items"],
            "failed": run_summary["failed_items"],
            "already_indexed": run_summary["already_indexed_items"],
            "new_items": run_summary["new_items"],
            "changed_items": run_summary["changed_items"],
            "unchanged_items": run_summary["unchanged_items"],
            "removed": run_summary["removed_items"],
            "removed_existing": run_summary["removed_existing_items"],
            "embeddings_model": embedding_model,
            "history_entry": run_summary,
        })
    except Exception as exc:
        run_summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_summary["duration_seconds"] = round(time.perf_counter() - started_clock, 2)
        run_summary["error"] = public_error_message(exc, 'Unable to index the library.')
        try:
            vector_store.record_run(run_summary)
        except Exception:
            app.logger.exception("Failed to record run summary after indexing error")
        try:
            update_index_progress_state(
                active=False,
                stage="failed",
                current_item_title="",
                message=run_summary["error"],
            )
        except Exception:
            app.logger.exception("Failed to update progress state after indexing error")
        return error_response(exc, 'Unable to index the library.')


@app.route('/api/query', methods=['POST'])
def query():
    current_config = get_current_config()
    if not current_config['chat_model']:
        return jsonify({"error": "Chat model required. Load models in Settings first."}), 400

    data = request.get_json(silent=True) or {}
    question = data.get('question', '')
    selected_item_ids = data.get('selected_item_ids')
    conversation_id = str(data.get('conversation_id') or '').strip()
    active_prompt_id = str(data.get('active_prompt_id') or '').strip()
    if not question:
        return jsonify({"error": "Question required"}), 400

    try:
        if not conversation_id:
            conversation = chat_store.create_conversation(
                selected_item_ids=selected_item_ids,
            )
            conversation_id = conversation['id']

        should_generate_title = conversation_needs_auto_title(conversation_id)

        conversation = chat_store.get_conversation(conversation_id)
        request_mode = str(data.get("response_mode") or "").strip().lower()
        conversation_mode = str(conversation.get("response_mode") or "").strip().lower()
        if request_mode:
            if request_mode != conversation_mode:
                chat_store.update_conversation(conversation_id, {"response_mode": request_mode})
            effective_mode = request_mode
        else:
            effective_mode = conversation_mode or (current_config.get("response_mode") or "synthesis")

        active_credential, client = require_active_provider()
        model_catalog = _get_cached_model_catalog(client, active_credential)
        embedding_model = resolve_embeddings_model(client, current_config, model_catalog=model_catalog, provider_name=active_credential.get('provider'))

        effective_item_ids = selected_item_ids
        if not effective_item_ids:
            global_filter = chat_store.get_context_filter()
            if global_filter.get("selected_item_ids"):
                effective_item_ids = global_filter["selected_item_ids"]

        filter_ids = None
        if effective_item_ids:
            filter_ids = set(str(item_id) for item_id in effective_item_ids)
        requested_top_k = max(int(data.get('top_k') or current_config.get('top_k') or 5), 1)
        results, _ = retrieve_query_results(client, embedding_model, question, filter_ids, requested_top_k, chat_model=current_config.get('chat_model', ''), reranker_mode=current_config.get('reranker_mode', 'local'))
        context = build_query_context(results)

        system_prompt = current_config['system_prompt']
        active_prompt = None
        if active_prompt_id:
            try:
                prompt_data = chat_store._find_prompt(chat_store._read_store(), active_prompt_id)
                if prompt_data:
                    active_prompt = prompt_data
                    system_prompt = f"{system_prompt}\n\n{prompt_data['text']}"
            except Exception:
                pass

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(build_conversation_messages(conversation_id))
        citation_instruction = get_citation_instruction(current_config, data, effective_mode)

        messages.append({
            "role": "user",
            "content": f"Context from Zotero library:\n\n{context}\n\nQuestion: {question}\n\n{citation_instruction}",
        })

        max_tokens = current_config['max_tokens']
        if active_prompt:
            length_config = active_prompt.get("length", "default")
            if length_config == "shorter":
                max_tokens = max(1, int(max_tokens * 0.5))
            elif length_config == "longer":
                max_tokens = min(65536, int(max_tokens * 1.5))

        response = client.chat_completion(
            model=current_config['chat_model'],
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        validated_answer, validation_usage = validate_answer(
            client,
            current_config['chat_model'],
            question,
            response['content'],
            context,
            citation_instruction,
            max_tokens,
        )
        response['content'] = validated_answer
        response['usage'] = merge_usage(response.get('usage', {}), validation_usage)

        generated_title = ""
        if should_generate_title:
            generated_title = generate_conversation_title(
                client,
                current_config['chat_model'],
                question,
                response['content'],
            )

        sources = [serialize_source(result) for result in results]
        saved_exchange = chat_store.append_exchange(
            conversation_id,
            question=question,
            answer=response['content'],
            selected_item_ids=selected_item_ids,
            sources=sources,
            usage=response.get('usage', {}),
            generated_title=generated_title,
            response_mode=effective_mode,
        )

        return jsonify({
            "answer": response['content'],
            "embeddings_model": embedding_model,
            "sources": sources,
            "usage": response.get('usage', {}),
            "conversation_id": conversation_id,
            "conversation": saved_exchange['conversation'],
            "assistant_message": saved_exchange['assistant_message'],
            "user_message": saved_exchange['user_message'],
        })
    except Exception as exc:
        return error_response(exc, 'Unable to answer the question.')


@app.route('/api/query/stream', methods=['POST'])
def query_stream():
    current_config = get_current_config()
    if not current_config['chat_model']:
        return jsonify({"error": "Chat model required. Load models in Settings first."}), 400

    data = request.get_json(silent=True) or {}
    question = data.get('question', '')
    selected_item_ids = data.get('selected_item_ids')
    conversation_id = str(data.get('conversation_id') or '').strip()
    active_prompt_id = str(data.get('active_prompt_id') or '').strip()
    if not question:
        return jsonify({"error": "Question required"}), 400

    def generate():
        try:
            nonlocal conversation_id
            if not conversation_id:
                conversation = chat_store.create_conversation(
                    selected_item_ids=selected_item_ids,
                )
                conversation_id = conversation['id']

            should_generate_title = conversation_needs_auto_title(conversation_id)

            conversation = chat_store.get_conversation(conversation_id)
            request_mode = str(data.get("response_mode") or "").strip().lower()
            conversation_mode = str(conversation.get("response_mode") or "").strip().lower()
            if request_mode:
                if request_mode != conversation_mode:
                    chat_store.update_conversation(conversation_id, {"response_mode": request_mode})
                effective_mode = request_mode
            else:
                effective_mode = conversation_mode or (current_config.get("response_mode") or "synthesis")

            active_credential, client = require_active_provider()
            model_catalog = _get_cached_model_catalog(client, active_credential)
            embedding_model = resolve_embeddings_model(client, current_config, model_catalog=model_catalog, provider_name=active_credential.get('provider'))

            effective_item_ids = selected_item_ids
            if not effective_item_ids:
                global_filter = chat_store.get_context_filter()
                if global_filter.get("selected_item_ids"):
                    effective_item_ids = global_filter["selected_item_ids"]

            filter_ids = None
            if effective_item_ids:
                filter_ids = set(str(item_id) for item_id in effective_item_ids)

            requested_top_k = max(int(data.get('top_k') or current_config.get('top_k') or 5), 1)
            results, _ = retrieve_query_results(client, embedding_model, question, filter_ids, requested_top_k, chat_model=current_config.get('chat_model', ''), reranker_mode=current_config.get('reranker_mode', 'local'))
            context = build_query_context(results)

            system_prompt = current_config['system_prompt']
            active_prompt = None
            if active_prompt_id:
                try:
                    prompt_data = chat_store._find_prompt(chat_store._read_store(), active_prompt_id)
                    if prompt_data:
                        active_prompt = prompt_data
                        system_prompt = f"{system_prompt}\n\n{prompt_data['text']}"
                except Exception:
                    pass

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(build_conversation_messages(conversation_id))
            citation_instruction = get_citation_instruction(current_config, data, effective_mode)

            messages.append({
                "role": "user",
                "content": f"Context from Zotero library:\n\n{context}\n\nQuestion: {question}\n\n{citation_instruction}",
            })

            sources = [serialize_source(result) for result in results]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'conversation_id': conversation_id})}\n\n"

            answer_chunks = []

            max_tokens = current_config['max_tokens']
            if active_prompt:
                length_config = active_prompt.get("length", "default")
                if length_config == "shorter":
                    max_tokens = max(1, int(max_tokens * 0.5))
                elif length_config == "longer":
                    max_tokens = min(65536, int(max_tokens * 1.5))

            for chunk in client.stream_chat_completion(
                model=current_config['chat_model'],
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            ):
                answer_chunks.append(chunk)
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            answer_text = ''.join(answer_chunks)
            yield f"data: {json.dumps({'type': 'tokens_done'})}\n\n"
            yield f"data: {json.dumps({'type': 'validating'})}\n\n"
            try:
                answer_text, _ = validate_answer(
                    client,
                    current_config['chat_model'],
                    question,
                    answer_text,
                    context,
                    citation_instruction,
                    max_tokens,
                )
            except Exception:
                pass

            generated_title = ""
            if should_generate_title:
                try:
                    generated_title = generate_conversation_title(
                        client,
                        current_config['chat_model'],
                        question,
                        answer_text,
                    )
                except Exception:
                    pass

            saved_exchange = chat_store.append_exchange(
                conversation_id,
                question=question,
                answer=answer_text,
                selected_item_ids=selected_item_ids,
                sources=sources,
                usage={},
                generated_title=generated_title,
                response_mode=effective_mode,
            )

            done_payload = {
                'type': 'done',
                'conversation_id': conversation_id,
                'conversation': saved_exchange['conversation'],
                'assistant_message': saved_exchange['assistant_message'],
                'user_message': saved_exchange['user_message'],
                'embeddings_model': embedding_model,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
        except Exception as exc:
            message = public_error_message(exc, 'Unable to stream the answer.')
            yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"

    response = Response(stream_with_context(generate()), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/api/proxy/google/models', methods=['POST'])
def proxy_google_models():
    try:
        active_credential, client = require_active_provider()
        if active_credential.get("provider") != "google":
            return jsonify({"error": "Active credential must be a Google provider."}), 400
        if not isinstance(client, GoogleProvider):
            return jsonify({"error": "Active credential must be a Google provider."}), 400

        catalog = client.fetch_models()
        return jsonify(catalog)
    except Exception as exc:
        return error_response(exc, 'Unable to fetch Google models.')


@app.route('/api/proxy/google/embedContent', methods=['POST'])
def proxy_google_embed_content():
    try:
        active_credential, client = require_active_provider()
        if active_credential.get("provider") != "google":
            return jsonify({"error": "Active credential must be a Google provider."}), 400
        if not isinstance(client, GoogleProvider):
            return jsonify({"error": "Active credential must be a Google provider."}), 400

        data = request.get_json(silent=True) or {}
        model = data.get("model", "")
        text = data.get("text", "")
        if not model or not text:
            return jsonify({"error": "model and text are required."}), 400

        embedding = client.create_embedding(model, text)
        return jsonify({"embedding": embedding, "model": model})
    except Exception as exc:
        return error_response(exc, 'Unable to create embedding.')


@app.route('/api/proxy/google/generateContent', methods=['POST'])
def proxy_google_generate_content():
    try:
        active_credential, client = require_active_provider()
        if active_credential.get("provider") != "google":
            return jsonify({"error": "Active credential must be a Google provider."}), 400
        if not isinstance(client, GoogleProvider):
            return jsonify({"error": "Active credential must be a Google provider."}), 400

        data = request.get_json(silent=True) or {}
        model = data.get("model", "")
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 4096)
        temperature = data.get("temperature", 0.7)
        if not model or not messages:
            return jsonify({"error": "model and messages are required."}), 400

        result = client.chat_completion(model, messages, max_tokens=max_tokens, temperature=temperature)
        return jsonify(result)
    except Exception as exc:
        return error_response(exc, 'Unable to generate content.')


@app.route('/api/proxy/google/streamGenerateContent', methods=['POST'])
def proxy_google_stream_generate_content():
    try:
        active_credential, client = require_active_provider()
        if active_credential.get("provider") != "google":
            return jsonify({"error": "Active credential must be a Google provider."}), 400
        if not isinstance(client, GoogleProvider):
            return jsonify({"error": "Active credential must be a Google provider."}), 400

        data = request.get_json(silent=True) or {}
        model = data.get("model", "")
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 4096)
        temperature = data.get("temperature", 0.7)
        if not model or not messages:
            return jsonify({"error": "model and messages are required."}), 400

        def generate():
            for chunk in client.stream_chat_completion(model, messages, max_tokens=max_tokens, temperature=temperature):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        response = Response(stream_with_context(generate()), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        return response
    except Exception as exc:
        return error_response(exc, 'Unable to stream content.')


if __name__ == '__main__':
    debug_enabled = os.environ.get('FLASK_DEBUG') == '1'
    host = (os.environ.get('HOST') or '127.0.0.1').strip() or '127.0.0.1'
    port = int(os.environ.get('PORT', '5000'))
    app.run(host=host, debug=debug_enabled, port=port)
