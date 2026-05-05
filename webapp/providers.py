import requests
import json
import re
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class ProviderAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def _sanitize_error_text(text: str) -> str:
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [redacted]", text, flags=re.IGNORECASE)
    sanitized = re.sub(r"([?&](?:key|api[_-]?key|token|secret)=)[^&\s]+", r"\1[redacted]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "sk-[redacted]", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized.strip()


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    message = ""
    if isinstance(payload, dict):
        error_payload = payload.get("error")
        if isinstance(error_payload, dict):
            message = str(error_payload.get("message") or error_payload.get("status") or "")
        elif isinstance(error_payload, str):
            message = error_payload

        if not message:
            message = str(payload.get("message") or payload.get("detail") or "")

    if not message:
        message = response.text[:300]

    return _sanitize_error_text(message)


def _extract_payload_error(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"message": "", "status_code": None}

    error_payload = payload.get("error")
    message = ""
    status_code = None
    if isinstance(error_payload, dict):
        message = str(
            error_payload.get("message")
            or error_payload.get("status")
            or error_payload.get("code")
            or ""
        )
        code = error_payload.get("code")
        if isinstance(code, int):
            status_code = code
        elif isinstance(code, str) and code.isdigit():
            status_code = int(code)
    elif isinstance(error_payload, str):
        message = error_payload

    if not message:
        message = str(payload.get("message") or payload.get("detail") or "")

    return {
        "message": _sanitize_error_text(message),
        "status_code": status_code,
    }


def _embedding_retry_delay_seconds(message: str, attempt: int) -> float:
    retry_after = re.search(r"retry after\s+(\d+(?:\.\d+)?)\s*second", message, flags=re.IGNORECASE)
    if retry_after:
        return max(float(retry_after.group(1)), 0.5)
    return min(1.0 * (2 ** attempt), 8.0)


def _extract_embeddings_from_payload(payload: Any) -> Optional[List[List[float]]]:
    if not isinstance(payload, dict):
        return None

    data_entries = payload.get("data")
    if isinstance(data_entries, list) and data_entries:
        embeddings = []
        for entry in data_entries:
            if isinstance(entry, dict) and isinstance(entry.get("embedding"), list):
                embeddings.append(entry["embedding"])
                continue
            if isinstance(entry, list):
                embeddings.append(entry)
                continue
            return None
        return embeddings

    direct_embedding = payload.get("embedding")
    if isinstance(direct_embedding, list):
        return [direct_embedding]

    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        normalized_embeddings = []
        for entry in embeddings:
            if isinstance(entry, list):
                normalized_embeddings.append(entry)
                continue
            if isinstance(entry, dict) and isinstance(entry.get("embedding"), list):
                normalized_embeddings.append(entry["embedding"])
                continue
            return None
        return normalized_embeddings

    return None


def _extract_embedding_from_payload(payload: Any) -> Optional[List[float]]:
    embeddings = _extract_embeddings_from_payload(payload)
    if not embeddings:
        return None
    return embeddings[0]


def _raise_for_status(response: requests.Response, provider_name: str) -> None:
    if response.ok:
        return

    message = _extract_error_message(response)
    if not message:
        message = f"{provider_name} request failed with status {response.status_code}."
    raise ProviderAPIError(message, response.status_code)


def _looks_like_embedding_model(model: Dict[str, Any]) -> bool:
    model_id = str(model.get("id", "")).lower()
    model_name = str(model.get("name", "")).lower()
    haystack = f"{model_id} {model_name}"
    return any(keyword in haystack for keyword in ("embedding", "embed"))


class BaseProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def fetch_models(self) -> Dict[str, Any]:
        """Fetch available models. Must return {providers: {...}, allModels: [...]}."""
        pass

    @abstractmethod
    def create_embedding(self, model: str, text: str) -> List[float]:
        pass

    def create_embeddings(self, model: str, texts: List[str]) -> List[List[float]]:
        return [self.create_embedding(model, text) for text in texts]

    @abstractmethod
    def chat_completion(self, model: str, messages: List[Dict[str, str]],
                        max_tokens: int = 4096, temperature: float = 0.7) -> Dict[str, Any]:
        pass

    @abstractmethod
    def stream_chat_completion(self, model: str, messages: List[Dict[str, str]],
                               max_tokens: int = 4096, temperature: float = 0.7):
        pass


class OpenRouterProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Researcharr"
        }

    def fetch_rerank_models(self) -> Dict[str, Any]:
        """Fetch available rerank models from OpenRouter."""
        resp = requests.get(
            f"{self.base_url}/models",
            params={"output_modalities": "rerank"},
            headers=self.headers,
            timeout=30,
        )
        _raise_for_status(resp, "OpenRouter")
        data = resp.json()
        providers = {}
        for model in data.get("data", []):
            provider_id = model["id"].split("/")[0]
            if provider_id not in providers:
                providers[provider_id] = {
                    "id": provider_id,
                    "name": provider_id,
                    "models": [],
                }
            providers[provider_id]["models"].append({
                "id": model["id"],
                "name": model.get("name", model["id"]),
                "contextLength": model.get("context_length", 0),
                "promptPrice": float(model.get("pricing", {}).get("prompt", "0")),
                "completionPrice": float(model.get("pricing", {}).get("completion", "0")),
            })
        return providers

    def rerank(self, model: str, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """Use the OpenRouter rerank API to rerank documents."""
        response = requests.post(
            f"{self.base_url}/rerank",
            headers=self.headers,
            json={
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
            timeout=120,
        )
        _raise_for_status(response, "OpenRouter")
        data = response.json()
        results = []
        for item in data.get("results", []):
            results.append({
                "index": item.get("index", 0),
                "relevance_score": item.get("relevance_score", 0.0),
                "text": item.get("document", {}).get("text", ""),
            })
        results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return results

    def fetch_models(self) -> Dict[str, Any]:
        """Fetch available chat and embedding models from OpenRouter."""
        # Fetch chat/completion models
        chat_resp = requests.get(
            f"{self.base_url}/models",
            headers=self.headers,
            timeout=30
        )
        _raise_for_status(chat_resp, "OpenRouter")
        chat_data = chat_resp.json()

        # Fetch embedding models from the dedicated endpoint
        emb_resp = requests.get(
            f"{self.base_url}/embeddings/models",
            headers=self.headers,
            timeout=30
        )
        emb_data = None
        if emb_resp.status_code == 200:
            try:
                emb_data = emb_resp.json()
            except Exception:
                pass

        providers = {}
        all_models = []
        seen_model_ids = set()

        def add_model(model: Dict[str, Any], is_embedding: bool) -> None:
            provider_id = model["id"].split("/")[0]
            if provider_id not in providers:
                providers[provider_id] = {
                    "id": provider_id,
                    "name": provider_id,
                    "models": []
                }

            model_entry = {
                "id": model["id"],
                "name": model.get("name", model["id"]),
                "contextLength": model.get("context_length", 0),
                "promptPrice": float(model.get("pricing", {}).get("prompt", "0")),
                "completionPrice": float(model.get("pricing", {}).get("completion", "0")),
                "isEmbedding": is_embedding
            }

            existing = next(
                (item for item in providers[provider_id]["models"] if item["id"] == model_entry["id"]),
                None
            )
            if existing:
                existing.update(model_entry)
            else:
                providers[provider_id]["models"].append(model_entry)

            if model["id"] not in seen_model_ids:
                all_models.append(model)
                seen_model_ids.add(model["id"])

        # Process chat models
        for model in chat_data.get("data", []):
            add_model(model, _looks_like_embedding_model(model))

        # Process embedding models
        if emb_data and "data" in emb_data:
            for model in emb_data["data"]:
                add_model(model, True)

        return {"providers": providers, "allModels": all_models}

    def create_embeddings(self, model: str, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        last_error_message = ""
        last_status_code = 500

        for attempt in range(5):
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json={"model": model, "input": texts},
                timeout=120
            )
            _raise_for_status(response, "OpenRouter")

            try:
                data = response.json()
            except ValueError:
                data = None

            payload_error = _extract_payload_error(data)
            if payload_error["message"]:
                last_error_message = payload_error["message"]
                last_status_code = payload_error["status_code"] or response.status_code or 500
                is_rate_limited = (
                    last_status_code == 429
                    or "rate limit" in last_error_message.casefold()
                    or "ratelimit" in last_error_message.casefold()
                )
                if is_rate_limited and attempt < 4:
                    time.sleep(_embedding_retry_delay_seconds(last_error_message, attempt))
                    continue
                raise ProviderAPIError(last_error_message, last_status_code)

            embeddings = _extract_embeddings_from_payload(data)
            if embeddings is not None and len(embeddings) == len(texts):
                return embeddings

            last_error_message = (
                "OpenRouter returned an unexpected embedding payload. "
                f"Expected {len(texts)} embeddings."
            )
            last_status_code = response.status_code or 500
            break

        raise ProviderAPIError(last_error_message, last_status_code)

    def create_embedding(self, model: str, text: str) -> List[float]:
        return self.create_embeddings(model, [text])[0]

    def chat_completion(self, model: str, messages: List[Dict[str, str]],
                        max_tokens: int = 4096, temperature: float = 0.7) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=120
        )
        _raise_for_status(response, "OpenRouter")
        data = response.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
            "model": data.get("model", model)
        }

    def stream_chat_completion(self, model: str, messages: List[Dict[str, str]],
                               max_tokens: int = 4096, temperature: float = 0.7):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            },
            stream=True,
            timeout=120
        )
        _raise_for_status(response, "OpenRouter")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except Exception:
                        pass


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def fetch_models(self) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/models",
            headers=self.headers,
            timeout=30
        )
        _raise_for_status(response, "OpenAI")
        data = response.json()

        providers = {}
        provider_id = "openai"
        providers[provider_id] = {"id": provider_id, "name": "OpenAI", "models": []}

        for model in data.get("data", []):
            model_id = model["id"]
            providers[provider_id]["models"].append({
                "id": model_id,
                "name": model_id,
                "contextLength": 0,
                "promptPrice": 0,
                "completionPrice": 0,
                "isEmbedding": "embedding" in model_id.lower()
            })

        return {"providers": providers, "allModels": data.get("data", [])}

    def create_embeddings(self, model: str, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json={"model": model, "input": texts},
            timeout=120
        )
        _raise_for_status(response, "OpenAI")
        data = response.json()
        embeddings = _extract_embeddings_from_payload(data)
        if embeddings is None or len(embeddings) != len(texts):
            raise ProviderAPIError(
                f"OpenAI returned an unexpected embedding payload. Expected {len(texts)} embeddings.",
                response.status_code or 500,
            )
        return embeddings

    def create_embedding(self, model: str, text: str) -> List[float]:
        return self.create_embeddings(model, [text])[0]

    def chat_completion(self, model: str, messages: List[Dict[str, str]],
                        max_tokens: int = 4096, temperature: float = 0.7) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=120
        )
        _raise_for_status(response, "OpenAI")
        data = response.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
            "model": data.get("model", model)
        }

    def stream_chat_completion(self, model: str, messages: List[Dict[str, str]],
                               max_tokens: int = 4096, temperature: float = 0.7):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            },
            stream=True,
            timeout=120
        )
        _raise_for_status(response, "OpenAI")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except Exception:
                        pass


class GoogleProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def fetch_models(self) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/models?key={self.api_key}",
            timeout=30
        )
        _raise_for_status(response, "Google")
        data = response.json()

        providers = {}
        provider_id = "google"
        providers[provider_id] = {"id": provider_id, "name": "Google", "models": []}

        for model in data.get("models", []):
            name = model["name"].replace("models/", "")
            methods = model.get("supportedGenerationMethods", [])
            is_embedding = (
                "embedding" in name.lower()
                or "embed" in name.lower()
                or "embedContent" in methods
            )
            providers[provider_id]["models"].append({
                "id": name,
                "name": model.get("displayName", name),
                "contextLength": model.get("inputTokenLimit", 0),
                "promptPrice": 0,
                "completionPrice": 0,
                "isEmbedding": is_embedding
            })

        return {"providers": providers, "allModels": data.get("models", [])}

    def create_embedding(self, model: str, text: str) -> List[float]:
        body = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]}
        }
        response = requests.post(
            f"{self.base_url}/models/{model}:embedContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=60
        )
        _raise_for_status(response, "Google")
        data = response.json()
        return data.get("embedding", {}).get("values", [])

    def chat_completion(self, model: str, messages: List[Dict[str, str]],
                        max_tokens: int = 4096, temperature: float = 0.7) -> Dict[str, Any]:
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        user_msgs = [m for m in messages if m["role"] != "system"]

        contents = []
        for msg in user_msgs:
            contents.append({
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": [{"text": msg["content"]}]
            })

        body = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        if system_msg:
            body["systemInstruction"] = {"parts": [{"text": system_msg["content"]}]}

        response = requests.post(
            f"{self.base_url}/models/{model}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=120
        )
        _raise_for_status(response, "Google")
        data = response.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return {
            "content": text,
            "usage": data.get("usageMetadata", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "model": model
        }

    def stream_chat_completion(self, model: str, messages: List[Dict[str, str]],
                               max_tokens: int = 4096, temperature: float = 0.7):
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        user_msgs = [m for m in messages if m["role"] != "system"]

        contents = []
        for msg in user_msgs:
            contents.append({
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": [{"text": msg["content"]}]
            })

        body = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        if system_msg:
            body["systemInstruction"] = {"parts": [{"text": system_msg["content"]}]}

        response = requests.post(
            f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json=body,
            stream=True,
            timeout=120
        )
        _raise_for_status(response, "Google")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        parsed = json.loads(data)
                        text = parsed.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text:
                            yield text
                    except Exception:
                        pass


def get_provider(provider_name: str, api_key: str) -> BaseProvider:
    """Factory to instantiate the correct provider."""
    provider_name = (provider_name or "openrouter").lower()
    if provider_name == "openrouter":
        return OpenRouterProvider(api_key)
    if provider_name == "openai":
        return OpenAIProvider(api_key)
    if provider_name == "google":
        return GoogleProvider(api_key)
    raise ValueError(f"Unknown provider: {provider_name}")
