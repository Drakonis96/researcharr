import json
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

RERANKER_MODE_LOCAL = "local"
RERANKER_MODE_API = "api"
RERANKER_MODE_OFF = "off"
VALID_RERANKER_MODES = {RERANKER_MODE_LOCAL, RERANKER_MODE_API, RERANKER_MODE_OFF}

LOCAL_MODEL_CANDIDATES = [
    os.environ.get("RESEARCHARR_RERANKER_MODEL", "").strip(),
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]

API_RERANKER_SYSTEM_PROMPT = (
    "You are a relevance scorer. Given a research question and a passage, "
    "score how relevant the passage is to the question on a scale of 0 to 10. "
    "Output ONLY a single integer between 0 and 10. No explanation, no other text."
)


class LocalReranker:
    def __init__(self, model_name: str = ""):
        self._model = None
        self._model_name = ""
        self._init_model_name = model_name

    def _ensure_loaded(self):
        if self._model is not None:
            return
        candidates = [
            self._init_model_name,
        ] + LOCAL_MODEL_CANDIDATES
        errors = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading local reranker model: %s", candidate)
                self._model = CrossEncoder(candidate)
                self._model_name = candidate
                logger.info("Loaded local reranker model: %s", candidate)
                return
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                logger.warning("Failed to load reranker model %s: %s", candidate, exc)
        raise RuntimeError(
            f"Unable to initialize local reranker: {'; '.join(errors)}"
        )

    @property
    def model_name(self):
        return self._model_name

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        if not results:
            return []
        self._ensure_loaded()
        pairs = [(query, _build_rerank_text(result)) for result in results]
        scores = self._model.predict(pairs)
        scored_results = []
        for result, score in zip(results, scores):
            new_result = dict(result)
            new_result["score"] = float(score)
            new_result["reranker_model"] = self._model_name
            scored_results.append(new_result)
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]


class OpenRouterReranker:
    def __init__(self, client, reranker_model: str):
        self._client = client
        self._reranker_model = reranker_model

    @property
    def model_name(self):
        return f"rerank:{self._reranker_model}"

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        if not results:
            return []

        documents = [_build_rerank_text(result) for result in results]

        try:
            ranked = self._client.rerank(
                model=self._reranker_model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(results)),
            )
        except Exception as exc:
            logger.warning("OpenRouter rerank API failed: %s", exc)
            return list(results[:top_k])

        scored = []
        for item in ranked:
            idx = item.get("index", 0)
            if 0 <= idx < len(results):
                new_result = dict(results[idx])
                new_result["score"] = item.get("relevance_score", 0.5)
                new_result["reranker_model"] = f"rerank:{self._reranker_model}"
                scored.append(new_result)

        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored[:top_k]


class ChatReranker:
    def __init__(self, client, reranker_model: str):
        self._client = client
        self._reranker_model = reranker_model

    @property
    def model_name(self):
        return f"api:{self._reranker_model}"

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        if not results:
            return []

        scored_results = []
        batch_size = min(8, max(1, len(results)))
        for start in range(0, len(results), batch_size):
            batch = results[start:start + batch_size]
            scored_results.extend(self._score_batch(query, batch))

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def _score_batch(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        passages_text = []
        for i, result in enumerate(results, start=1):
            passage = _build_rerank_text(result)
            passages_text.append(f"[{i}] {passage}")

        user_content = (
            f"Question: {query}\n\n"
            f"Passages:\n{chr(10).join(passages_text)}\n\n"
            f"Score each passage's relevance to the question. "
            f"Output one line per passage with the format: [N] score"
        )

        try:
            response = self._client.chat_completion(
                model=self._reranker_model,
                messages=[
                    {"role": "system", "content": API_RERANKER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=min(128, len(results) * 10),
                temperature=0,
            )
            content = (response.get("content") or "").strip()
            scores = self._parse_scores(content, len(results))
        except Exception as exc:
            logger.warning("API reranker failed: %s", exc)
            scores = [0.5] * len(results)

        scored = []
        for result, score in zip(results, scores):
            new_result = dict(result)
            new_result["score"] = score
            new_result["reranker_model"] = f"api:{self._reranker_model}"
            scored.append(new_result)
        return scored

    def _parse_scores(self, content: str, expected_count: int) -> List[float]:
        scores = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            import re
            match = re.search(r"(\d+(?:\.\d+)?)\s*$", line)
            if match:
                try:
                    val = float(match.group(1))
                    scores.append(min(max(val, 0.0), 10.0))
                except ValueError:
                    scores.append(5.0)
            else:
                scores.append(5.0)

        while len(scores) < expected_count:
            scores.append(5.0)
        return scores[:expected_count]


class NoopReranker:
    @property
    def model_name(self):
        return "none"

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        return list(results[:top_k])


class Reranker:
    def __init__(self, mode: str = "", client=None, reranker_model: str = ""):
        mode = (mode or os.environ.get("RESEARCHARR_RERANKER_MODE", "") or RERANKER_MODE_LOCAL).strip().lower()
        if mode not in VALID_RERANKER_MODES:
            mode = RERANKER_MODE_LOCAL

        self._mode = mode
        if mode == RERANKER_MODE_OFF:
            self._impl = NoopReranker()
        elif mode == RERANKER_MODE_API:
            if not client or not reranker_model:
                logger.warning("API reranker requires a client and reranker_model; falling back to local")
                self._mode = RERANKER_MODE_LOCAL
                self._impl = LocalReranker()
            else:
                provider_name = getattr(client, '__class__', type(client)).__name__
                if provider_name == "OpenRouterProvider":
                    self._impl = OpenRouterReranker(client, reranker_model)
                else:
                    self._impl = ChatReranker(client, reranker_model)
        else:
            self._impl = LocalReranker()

    @property
    def mode(self):
        return self._mode

    @property
    def model_name(self):
        return self._impl.model_name

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        return self._impl.rerank(query, results, top_k)


def _build_rerank_text(result: Dict[str, Any]) -> str:
    metadata = result.get("metadata", {})
    parts = []
    for label, value in (
        ("Title", metadata.get("title")),
        ("Authors", metadata.get("authors")),
        ("Date", metadata.get("date")),
        ("Publication", metadata.get("publication")),
        ("Section", metadata.get("section_heading")),
    ):
        normalized = str(value or "").strip()
        if normalized:
            parts.append(f"{label}: {normalized}")
    passage = str(result.get("text") or "").strip()
    if passage:
        parts.append(passage[:4000])
    return "\n".join(parts)