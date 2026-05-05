import hashlib
import json
import os
import secrets
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet


LEGACY_DEFAULT_SYSTEM_PROMPT = (
    "<context>\n"
    "You are a research assistant with access to the user's Zotero library. "
    "Your only source of knowledge is the full-text passages from the attachments "
    "provided in the context. You have no general knowledge or prior training.\n"
    "</context>\n"
    "\n"
    "<sources>\n"
    "Each source in the context has an identification number [N]. You may only use these "
    "numbered sources to respond. Do not treat titles, abstracts, notes, or bibliographic "
    "metadata as evidentiary passages.\n"
    "</sources>\n"
    "\n"
    "<instructions>\n"
    "Follow this step-by-step process for each response:\n"
    "\n"
    "1. IDENTIFY — Examine the provided context. Identify which numbered sources [N] "
    "are relevant to the user's question.\n"
    "\n"
    "2. EXTRACT — Extract all relevant information from the identified sources, even if "
    "partial or indirect. Read each passage carefully looking for mentions, references, or "
    "passages that relate to the question, even if they do not directly answer it. "
    "Passages may contain relevant information even if they do not explicitly mention "
    "the exact terms of the question.\n"
    "\n"
    "3. VERIFY — If you found relevant information in step 2, draft the response "
    "with what is available. If a source provides partial or tangential information on the topic, "
    "present it indicating which aspects it covers and which remain unanswered. NEVER respond that "
    "you have no information if any source contains data related to the topic of the question, "
    "even if incomplete or indirect. Only if truly no source contains "
    "information related to the question, respond: "
    "\"I do not have enough information in the sources to answer this question.\" "
    "Do not use your general knowledge under any circumstances.\n"
    "\n"
    "4. DRAFT — Draft a response that cites each claim "
    "with its source [N]. Use double quotes only to reproduce exact passages, and place "
    "the passage immediately before its citation marker (e.g., \"textual passage\"[1]). "
    "Paraphrases may be cited without quotes. When helpful to the user, use Markdown to "
    "structure the response, including lists, tables, blockquotes, and code blocks. "
    "Be precise and concise.\n"
    "</instructions>"
)

DEFAULT_SYSTEM_PROMPT = (
    "<context>\n"
    "You are a research assistant with access to the user's Zotero library. "
    "Respond based on the text passages provided in the context. "
    "Do not invent data or add information that is not in the sources.\n"
    "</context>\n"
    "\n"
    "<sources>\n"
    "Each source in the context has an identification number [N]. Use those sources to "
    "respond. Do not treat titles, abstracts, notes, or bibliographic metadata as "
    "evidentiary passages.\n"
    "</sources>\n"
    "\n"
    "<instructions>\n"
    "1. RESPOND DIRECTLY — Read the sources and answer the user's question clearly "
    "and fluently. Prioritize giving a useful and complete answer with what the sources provide.\n"
    "\n"
"2. USE EVERYTHING AVAILABLE — If the sources contain relevant information, even if "
"partial or indirect, use it to build the best possible answer. Connect information "
"from different sources when reasonable. Do not discard useful information for not being "
"literally exact regarding the question. If the user asks for a table, list, or other "
"format and you have partial information, build it with what is available and mark empty "
"cells with \"\xe2\x80\x94\" rather than refusing to create it.\n"
"\n"
"3. BE HONEST WITHOUT BEING EVASIVE — If a specific piece of data does not appear in the "
"sources, briefly indicate it with \"\xe2\x80\x94\" or a footnote, but do not abandon the answer "
"or turn it into an inventory of what is missing. Only respond \"I do not have enough "
"information in the sources to answer this question.\" when truly no source contains "
"anything about the specific topic asked.\n"
    "\n"
"4. DO NOT CHANGE THE SUBJECT \xe2\x80\x94 If the sources do not contain information about what "
"the user is asking, do not answer about a different topic that does appear in the sources. "
"Clearly indicate what is not found. Only if appropriate, briefly mention what the sources "
"do cover, but do not build the answer around that alternative content.\n"
    "\n"
    "5. CITE NATURALLY — Cite each claim with its source [N]. Use double quotes only "
    "for exact textual fragments (e.g., \"textual passage\"[1]). Paraphrases are cited "
    "without quotes. When helpful to the user, use Markdown to structure the response.\n"
    "\n"
    "IMPORTANT: Your goal is to be helpful, not defensive. Do not add metacognitive sections like "
    "\"Explicit in the sources\" or \"Cautious inference\". Write like a researcher "
    "reporting to a colleague: with clarity, precision, and fluency.\n"
    "</instructions>"
)

PREVIOUS_DEFAULT_SYSTEM_PROMPT = (
    "<context>\n"
    "You are a research assistant with access to the user's Zotero library. "
    "Your only source of knowledge is the full-text passages from the attachments "
    "provided in the context. Do not use general knowledge or fill gaps with intuition.\n"
    "</context>\n"
    "\n"
    "<sources>\n"
    "Each source in the context has an identification number [N]. You may only use those "
    "numbered sources to respond. Do not treat titles, abstracts, notes, or bibliographic "
    "metadata as evidentiary passages.\n"
    "</sources>\n"
    "\n"
    "<instructions>\n"
    "Follow this step-by-step process for each response:\n"
    "\n"
    "1. IDENTIFY — Examine the context and detect which numbered sources [N] provide explicit "
    "answers, which only provide partial information, and which are not relevant.\n"
    "\n"
    "2. EXTRACT — Extract the relevant information from each pertinent source. If a response "
    "requires connecting two passages from the same document or different documents, do so only "
    "when the connection is justified by the text.\n"
    "\n"
    "3. VERIFY — Always distinguish between: a) what is explicit in the sources; b) what can "
    "only be inferred cautiously. Do not promote an inference to fact. If a person, work, or "
    "concept is not sufficiently supported by the sources, omit it or present it expressly as "
    "a tentative case. Only respond \"I do not have enough information in the sources to answer "
    "this question.\" when truly no source contains related information.\n"
    "\n"
    "4. DRAFT \xe2\x80\x94 Write a precise and concise response. Cite each claim with its source [N]. "
    "Use double quotes only to reproduce exact passages, and place the passage "
    "immediately before its citation marker (e.g., \"textual passage\"[1]). Paraphrases "
    "may be cited without quotes. If the evidence is mixed or partial, indicate it clearly using "
    "brief sections like \"Explicit in the sources\" and \"Cautious inference\" when "
    "needed. When helpful to the user, use Markdown to structure the response.\n"
    "</instructions>"
)

LEGACY_SYSTEM_PROMPTS = {
    LEGACY_DEFAULT_SYSTEM_PROMPT,
    PREVIOUS_DEFAULT_SYSTEM_PROMPT,
    "You are a research assistant with access to the user's Zotero library. Answer questions only from the provided attachment full-text passages. Do not treat titles, abstracts, notes, or bibliography metadata as evidentiary passages. For each claim you make, quote the exact supporting fragment in double quotes immediately before its citation marker, and use [N] format for citations (e.g., \"supporting text\"[1]). Cite only from the numbered sources provided in the context. Be precise and concise.",
    "You are a research assistant with access to the user's Zotero library. Answer questions based on the provided context. For each claim, cite the source item and include the exact fragment of text that supports your answer. Format citations as [Source: Author (Year) — Title]. Be precise and concise.",
}

_LEGACY_MARKERS = (
    "No tienes conocimiento general",
    "Inferencia cautelosa",
    "lo explícito en las fuentes",
    "inferirse con cautela",
    "caso tentativo",
)


def _is_legacy_system_prompt(prompt: str) -> bool:
    if not prompt:
        return False
    if prompt in LEGACY_SYSTEM_PROMPTS:
        return True
    if prompt == DEFAULT_SYSTEM_PROMPT:
        return False
    lowered = prompt.casefold()
    if lowered.startswith("<contexto>") and any(marker.casefold() in lowered for marker in _LEGACY_MARKERS):
        return True
    return False


DEFAULT_CONFIG = {
    "provider": "openrouter",
    "chat_model": "",
    "embeddings_model": "",
    "chunk_size": 500,
    "chunk_overlap": 200,
    "top_k": 5,
    "max_tokens": 4096,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "response_mode": "synthesis",
    "reranker_mode": "local",
    "active_credential_id": None,
}

SUPPORTED_PROVIDERS = {"openrouter", "openai", "google"}


class SettingsStore:
    def __init__(self, instance_path: str):
        self.instance_path = os.path.abspath(instance_path)
        self.config_path = os.path.join(self.instance_path, "researcharr-settings.json")
        self.master_key_path = os.path.join(self.instance_path, "researcharr-secrets.key")
        self._lock = threading.RLock()

        self._ensure_directory(self.instance_path, 0o700)
        self._fernet = Fernet(self._load_or_create_master_key())
        self._ensure_store_exists()

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._read_store()["config"])

    def get_public_state(self) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            active_id = store["config"].get("active_credential_id")
            credentials = [self._public_credential(item, active_id) for item in store["credentials"]]
            active_credential = next(
                (credential for credential in credentials if credential["id"] == active_id),
                None,
            )
            return {
                **dict(store["config"]),
                "has_active_api_key": bool(active_credential),
                "credentials": credentials,
                "active_credential": active_credential,
                "mcp_config": dict(store.get("mcp_config", {"enabled": True, "host": "127.0.0.1", "port": 5001})),
            }

    def get_active_credential(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            store = self._read_store()
            active_id = store["config"].get("active_credential_id")
            if not active_id:
                return None

            credential = self._find_credential(store, active_id)
            if credential is None:
                return None

            payload = self._public_credential(credential, active_id)
            payload["api_key"] = self._decrypt_secret(credential["secret"])
            return payload

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            active_credential = self._active_credential(store)

            for key in ("chat_model", "embeddings_model"):
                if key in updates:
                    store["config"][key] = str(updates.get(key, "") or "").strip()

            if "response_mode" in updates:
                mode = str(updates.get("response_mode", "") or "").strip().lower()
                store["config"]["response_mode"] = mode if mode in ("literal", "paraphrase", "synthesis") else "synthesis"

            if "reranker_mode" in updates:
                reranker_mode = str(updates.get("reranker_mode", "") or "").strip().lower()
                store["config"]["reranker_mode"] = reranker_mode if reranker_mode in ("local", "api", "off") else "local"

            if "system_prompt" in updates:
                store["config"]["system_prompt"] = str(updates.get("system_prompt", "") or "").strip()

            for key, minimum, maximum in (
                ("chunk_size", 1, 100000),
                ("chunk_overlap", 0, 100000),
                ("max_tokens", 1, 65536),
            ):
                if key in updates:
                    store["config"][key] = self._coerce_int(key, updates[key], minimum, maximum)

            if "top_k" in updates:
                store["config"]["top_k"] = self._coerce_positive_int("Top k", updates["top_k"])

            if active_credential is not None:
                requested_provider = str(updates.get("provider", active_credential["provider"]) or "").strip().lower()
                if requested_provider and requested_provider != active_credential["provider"]:
                    raise ValueError("Provider is controlled by the active credential. Activate a credential for that provider first.")
                store["config"]["provider"] = active_credential["provider"]
            elif "provider" in updates:
                store["config"]["provider"] = self._normalize_provider(updates["provider"])

            self._write_store(store)
            return dict(store["config"])

    def create_credential(
        self,
        provider: Any,
        label: Any,
        api_key: Any,
        activate: bool = True,
    ) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            provider_name = self._normalize_provider(provider)
            secret_value = str(api_key or "").strip()
            if not secret_value:
                raise ValueError("API key required.")

            label_value = str(label or "").strip() or self._default_label(store["credentials"], provider_name)
            timestamp = self._timestamp()
            credential = {
                "id": str(uuid.uuid4()),
                "provider": provider_name,
                "label": label_value,
                "secret": self._encrypt_secret(secret_value),
                "created_at": timestamp,
                "updated_at": timestamp,
            }

            store["credentials"].append(credential)
            should_activate = activate or len(store["credentials"]) == 1
            if should_activate:
                self._activate_store_credential(store, credential)

            self._write_store(store)
            return self._public_credential(credential, store["config"].get("active_credential_id"))

    def update_credential(self, credential_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            credential = self._find_credential(store, credential_id)
            if credential is None:
                raise ValueError("Credential not found.")

            previous_provider = credential["provider"]
            if "provider" in updates and updates["provider"] not in (None, ""):
                credential["provider"] = self._normalize_provider(updates["provider"])

            if "label" in updates:
                label_value = str(updates.get("label", "") or "").strip()
                credential["label"] = label_value or credential["label"]

            if "api_key" in updates:
                secret_value = str(updates.get("api_key", "") or "").strip()
                if secret_value:
                    credential["secret"] = self._encrypt_secret(secret_value)

            credential["updated_at"] = self._timestamp()

            active_id = store["config"].get("active_credential_id")
            if active_id == credential_id:
                self._activate_store_credential(store, credential, previous_provider=previous_provider)

            if bool(updates.get("activate")):
                self._activate_store_credential(store, credential, previous_provider=previous_provider)

            self._write_store(store)
            return self._public_credential(credential, store["config"].get("active_credential_id"))

    def activate_credential(self, credential_id: str) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            credential = self._find_credential(store, credential_id)
            if credential is None:
                raise ValueError("Credential not found.")

            self._activate_store_credential(store, credential)
            self._write_store(store)
            return self._public_credential(credential, store["config"].get("active_credential_id"))

    def delete_credential(self, credential_id: str) -> None:
        with self._lock:
            store = self._read_store()
            credential = self._find_credential(store, credential_id)
            if credential is None:
                raise ValueError("Credential not found.")

            store["credentials"] = [item for item in store["credentials"] if item["id"] != credential_id]
            if store["config"].get("active_credential_id") == credential_id:
                store["config"]["active_credential_id"] = None
                store["config"]["chat_model"] = ""
                store["config"]["embeddings_model"] = ""
                if store["credentials"]:
                    store["config"]["provider"] = store["credentials"][0]["provider"]

            self._write_store(store)

    def _ensure_store_exists(self) -> None:
        if os.path.exists(self.config_path):
            return
        self._write_store(self._default_store())

    def _default_store(self) -> Dict[str, Any]:
        return {
            "config": dict(DEFAULT_CONFIG),
            "credentials": [],
            "mcp_keys": [],
            "mcp_config": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 5001,
            },
        }

    def _read_store(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            return self._default_store()

        with open(self.config_path, "r", encoding="utf-8") as handle:
            return self._normalize_store(json.load(handle))

    def _write_store(self, store: Dict[str, Any]) -> None:
        normalized = self._normalize_store(store)
        descriptor, temp_path = tempfile.mkstemp(dir=self.instance_path)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(normalized, handle, indent=2)
                handle.write("\n")
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.config_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _normalize_store(self, store: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._default_store()

        config = store.get("config", {}) if isinstance(store, dict) else {}
        for key, default_value in DEFAULT_CONFIG.items():
            normalized["config"][key] = config.get(key, default_value)

        if _is_legacy_system_prompt(normalized["config"].get("system_prompt", "")):
            normalized["config"]["system_prompt"] = DEFAULT_SYSTEM_PROMPT

        credentials = []
        for credential in store.get("credentials", []):
            if not isinstance(credential, dict):
                continue
            if not credential.get("id") or not credential.get("provider") or not credential.get("label") or not credential.get("secret"):
                continue
            credentials.append({
                "id": str(credential["id"]),
                "provider": self._normalize_provider(credential["provider"]),
                "label": str(credential["label"]),
                "secret": str(credential["secret"]),
                "created_at": str(credential.get("created_at") or self._timestamp()),
                "updated_at": str(credential.get("updated_at") or credential.get("created_at") or self._timestamp()),
            })

        normalized["credentials"] = credentials
        active_id = normalized["config"].get("active_credential_id")
        valid_ids = {credential["id"] for credential in credentials}
        if active_id not in valid_ids:
            normalized["config"]["active_credential_id"] = None

        if normalized["config"].get("active_credential_id"):
            active_credential = self._active_credential(normalized)
            if active_credential:
                normalized["config"]["provider"] = active_credential["provider"]

        mcp_keys = []
        for key in store.get("mcp_keys", []):
            if not isinstance(key, dict) or not key.get("id") or not key.get("key_hash"):
                continue
            mcp_keys.append({
                "id": str(key["id"]),
                "label": str(key.get("label", "")),
                "key_hash": str(key["key_hash"]),
                "key_prefix": str(key.get("key_prefix", "")),
                "created_at": str(key.get("created_at") or self._timestamp()),
                "updated_at": str(key.get("updated_at") or key.get("created_at") or self._timestamp()),
            })
        normalized["mcp_keys"] = mcp_keys

        mcp_config = store.get("mcp_config", {})
        if isinstance(mcp_config, dict):
            normalized["mcp_config"] = {
                "enabled": bool(mcp_config.get("enabled", True)),
                "host": str(mcp_config.get("host", "127.0.0.1")),
                "port": int(mcp_config.get("port", 5001)),
            }
        else:
            normalized["mcp_config"] = {"enabled": True, "host": "127.0.0.1", "port": 5001}

        return normalized

    def _load_or_create_master_key(self) -> bytes:
        env_key = os.environ.get("RESEARCHARR_SECRET_KEY")
        if env_key:
            key_bytes = env_key.encode("utf-8") if isinstance(env_key, str) else env_key
            try:
                Fernet(key_bytes)
                return key_bytes
            except Exception:
                pass

        if os.path.exists(self.master_key_path):
            with open(self.master_key_path, "rb") as handle:
                return handle.read().strip()

        key = Fernet.generate_key()
        descriptor, temp_path = tempfile.mkstemp(dir=self.instance_path)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(key)
                handle.write(b"\n")
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.master_key_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return key

    def _encrypt_secret(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")

    def _decrypt_secret(self, encrypted_value: str) -> str:
        return self._fernet.decrypt(encrypted_value.encode("utf-8")).decode("utf-8")

    def _public_credential(self, credential: Dict[str, Any], active_id: Optional[str]) -> Dict[str, Any]:
        return {
            "id": credential["id"],
            "provider": credential["provider"],
            "label": credential["label"],
            "masked_key": self._mask_secret(self._decrypt_secret(credential["secret"])),
            "created_at": credential["created_at"],
            "updated_at": credential["updated_at"],
            "is_active": credential["id"] == active_id,
        }

    def _activate_store_credential(
        self,
        store: Dict[str, Any],
        credential: Dict[str, Any],
        previous_provider: Optional[str] = None,
    ) -> None:
        current_provider = previous_provider or store["config"].get("provider")
        store["config"]["active_credential_id"] = credential["id"]
        store["config"]["provider"] = credential["provider"]
        if current_provider != credential["provider"]:
            store["config"]["chat_model"] = ""
            store["config"]["embeddings_model"] = ""

    def _find_credential(self, store: Dict[str, Any], credential_id: str) -> Optional[Dict[str, Any]]:
        return next((item for item in store["credentials"] if item["id"] == credential_id), None)

    def _active_credential(self, store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        active_id = store["config"].get("active_credential_id")
        if not active_id:
            return None
        return self._find_credential(store, active_id)

    def _default_label(self, credentials: List[Dict[str, Any]], provider: str) -> str:
        current_count = sum(1 for item in credentials if item["provider"] == provider)
        return f"{provider.title()} key {current_count + 1}"

    def _normalize_provider(self, provider: Any) -> str:
        provider_name = str(provider or "").strip().lower()
        if provider_name not in SUPPORTED_PROVIDERS:
            raise ValueError("Unsupported provider.")
        return provider_name

    def _coerce_int(self, key: str, value: Any, minimum: int, maximum: int) -> int:
        try:
            integer_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key.replace('_', ' ').title()} must be an integer.") from exc

        if integer_value < minimum or integer_value > maximum:
            raise ValueError(f"{key.replace('_', ' ').title()} must be between {minimum} and {maximum}.")
        return integer_value

    def _coerce_positive_int(self, label: str, value: Any) -> int:
        try:
            integer_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer.") from exc

        if integer_value < 1:
            raise ValueError(f"{label} must be at least 1.")
        return integer_value

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _mask_secret(self, secret: str) -> str:
        if len(secret) <= 8:
            return "*" * len(secret)
        return f"{secret[:4]}...{secret[-4:]}"

    def create_mcp_key(self, label: Any = "") -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            raw_key = f"researcharr_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
            label_value = str(label or "").strip() or f"Key {len(store.get('mcp_keys', [])) + 1}"
            timestamp = self._timestamp()
            key_entry = {
                "id": str(uuid.uuid4()),
                "label": label_value,
                "key_hash": key_hash,
                "key_prefix": raw_key[:14],
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            store.setdefault("mcp_keys", []).append(key_entry)
            self._write_store(store)
            result = dict(key_entry)
            result["key"] = raw_key
            return result

    def list_mcp_keys(self) -> List[Dict[str, Any]]:
        with self._lock:
            store = self._read_store()
            return [
                {
                    "id": k["id"],
                    "label": k["label"],
                    "key_prefix": k.get("key_prefix", ""),
                    "created_at": k["created_at"],
                    "updated_at": k["updated_at"],
                }
                for k in store.get("mcp_keys", [])
            ]

    def delete_mcp_key(self, key_id: str) -> None:
        with self._lock:
            store = self._read_store()
            existing = next((k for k in store.get("mcp_keys", []) if k["id"] == key_id), None)
            if existing is None:
                raise ValueError("MCP key not found.")
            store["mcp_keys"] = [k for k in store.get("mcp_keys", []) if k["id"] != key_id]
            self._write_store(store)

    def validate_mcp_key(self, raw_key: str) -> bool:
        if not raw_key:
            return False
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        with self._lock:
            store = self._read_store()
            return any(k["key_hash"] == key_hash for k in store.get("mcp_keys", []))

    def has_mcp_keys(self) -> bool:
        with self._lock:
            store = self._read_store()
            return len(store.get("mcp_keys", [])) > 0

    def get_mcp_config(self) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            return dict(store.get("mcp_config", {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 5001,
            }))

    def update_mcp_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            current = store.get("mcp_config", {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 5001,
            })
            if "enabled" in updates:
                current["enabled"] = bool(updates["enabled"])
            if "host" in updates:
                host = str(updates["host"] or "").strip()
                if host:
                    current["host"] = host
            if "port" in updates:
                try:
                    port = int(updates["port"])
                    if 1 <= port <= 65535:
                        current["port"] = port
                except (TypeError, ValueError):
                    pass
            store["mcp_config"] = current
            self._write_store(store)
            return dict(current)

    def _ensure_directory(self, path: str, mode: int) -> None:
        os.makedirs(path, exist_ok=True)
        try:
            os.chmod(path, mode)
        except PermissionError:
            pass
