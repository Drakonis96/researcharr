import json
import os
import re
import tempfile
import threading
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


VALID_LENGTH_CONFIGS = {"shorter", "default", "longer"}

DEFAULT_STORE = {
    "conversations": [],
    "prompts": [],
}

DEFAULT_STOP_WORDS = {
    "a", "about", "after", "against", "all", "also", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "before", "between", "both", "but", "by", "can", "como", "con", "del", "desde", "did", "do", "does",
    "during", "each", "el", "ella", "ellas", "ellos", "en", "entre", "es", "esta", "este", "for", "from", "had",
    "has", "have", "how", "into", "its", "la", "las", "los", "más", "más", "may", "much", "near", "not", "of",
    "on", "or", "para", "por", "qué", "que", "ser", "sobre", "some", "such", "than", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "through", "to", "un", "una", "was", "what", "when", "which", "who",
    "why", "with", "would", "y",
}


class ChatStore:
    def __init__(self, instance_path: str):
        self.instance_path = os.path.abspath(instance_path)
        self.store_path = os.path.join(self.instance_path, "researcharr-conversations.json")
        self._context_filter_path = os.path.join(self.instance_path, "researcharr-context-filter.json")
        self._lock = threading.RLock()
        self._ensure_directory(self.instance_path, 0o700)
        self._ensure_store_exists()

    def get_workspace_state(self) -> Dict[str, Any]:
        with self._lock:
            self._purge_expired_trash()
            store = self._read_store()
            active = [c for c in store["conversations"] if not c.get("deleted_at")]
            conversations = sorted(
                (self._conversation_summary(conversation) for conversation in active),
                key=lambda item: item.get("updated_at") or "",
                reverse=True,
            )
            prompts = sorted(
                (dict(prompt) for prompt in store["prompts"]),
                key=lambda item: item.get("updated_at") or item.get("created_at") or "",
                reverse=True,
            )
            return {
                "conversations": conversations,
                "prompts": prompts,
            }

    def create_conversation(
        self,
        title: str = "",
        selected_item_ids: Optional[List[str]] = None,
        response_mode: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            timestamp = self._timestamp()
            conversation = {
                "id": str(uuid.uuid4()),
                "title": self._normalize_title(title) or "Untitled workspace",
                "title_auto": not bool(self._normalize_title(title)),
                "tags": [],
                "created_at": timestamp,
                "updated_at": timestamp,
                "messages": [],
                "selected_item_ids": self._normalize_id_list(selected_item_ids),
                "export_count": 0,
                "response_mode": self._normalize_response_mode(response_mode),
            }
            store["conversations"].append(conversation)
            self._write_store(store)
            return self._conversation_summary(conversation)

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            conversation = self._find_conversation(store, conversation_id)
            if conversation is None:
                raise ValueError("Conversation not found.")
            return self._conversation_detail(conversation)

    def update_conversation(self, conversation_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            conversation = self._find_conversation(store, conversation_id)
            if conversation is None:
                raise ValueError("Conversation not found.")

            if "title" in updates:
                title = self._normalize_title(updates.get("title"))
                if not title:
                    raise ValueError("Title required.")
                conversation["title"] = title
                conversation["title_auto"] = False

            if "selected_item_ids" in updates:
                conversation["selected_item_ids"] = self._normalize_id_list(updates.get("selected_item_ids"))

            if "response_mode" in updates:
                conversation["response_mode"] = self._normalize_response_mode(updates.get("response_mode"))

            conversation["updated_at"] = self._timestamp()
            self._write_store(store)
            return self._conversation_summary(conversation)

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock:
            store = self._read_store()
            existing = self._find_conversation(store, conversation_id)
            if existing is None:
                raise ValueError("Conversation not found.")
            existing["deleted_at"] = self._timestamp()
            self._write_store(store)

    def permanently_delete_conversation(self, conversation_id: str) -> None:
        with self._lock:
            store = self._read_store()
            existing = self._find_conversation(store, conversation_id)
            if existing is None:
                raise ValueError("Conversation not found.")
            store["conversations"] = [item for item in store["conversations"] if item["id"] != conversation_id]
            self._write_store(store)

    def get_trashed_conversations(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._purge_expired_trash()
            store = self._read_store()
            trashed = [c for c in store["conversations"] if c.get("deleted_at")]
            trashed.sort(key=lambda c: c["deleted_at"], reverse=True)
            return [self._conversation_summary(c) for c in trashed]

    def restore_conversation(self, conversation_id: str) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            existing = self._find_conversation(store, conversation_id)
            if existing is None:
                raise ValueError("Conversation not found.")
            existing.pop("deleted_at", None)
            existing["updated_at"] = self._timestamp()
            self._write_store(store)
            return self._conversation_summary(existing)

    def empty_trash(self) -> None:
        with self._lock:
            store = self._read_store()
            store["conversations"] = [c for c in store["conversations"] if not c.get("deleted_at")]
            self._write_store(store)

    def _purge_expired_trash(self) -> None:
        store = self._read_store()
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        changed = False
        remaining = []
        for c in store["conversations"]:
            deleted_at = c.get("deleted_at")
            if deleted_at:
                try:
                    dt = datetime.fromisoformat(deleted_at)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < cutoff:
                        changed = True
                        continue
                except (ValueError, TypeError):
                    changed = True
                    continue
            remaining.append(c)
        if changed:
            store["conversations"] = remaining
            self._write_store(store)

    def append_exchange(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        selected_item_ids: Optional[List[str]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, Any]] = None,
        generated_title: str = "",
        response_mode: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            conversation = self._find_conversation(store, conversation_id)
            if conversation is None:
                raise ValueError("Conversation not found.")

            timestamp = self._timestamp()
            normalized_sources = [self._normalize_source(source) for source in (sources or [])]
            selected_ids = self._normalize_id_list(selected_item_ids)
            had_existing_messages = bool(conversation.get("messages"))
            user_message = {
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": str(question or "").strip(),
                "created_at": timestamp,
                "pinned": False,
                "selected_item_ids": selected_ids,
            }
            assistant_message = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": str(answer or "").strip(),
                "created_at": timestamp,
                "pinned": False,
                "sources": normalized_sources,
                "usage": dict(usage or {}),
            }
            conversation["messages"].extend([user_message, assistant_message])
            conversation["selected_item_ids"] = selected_ids or conversation.get("selected_item_ids", [])
            if response_mode:
                conversation["response_mode"] = self._normalize_response_mode(response_mode)
            if conversation.get("title_auto", True) and not had_existing_messages and question:
                conversation["title"] = self._normalize_title(generated_title) or self._build_auto_title(question)
                conversation["title_auto"] = True
            conversation["tags"] = self._derive_tags(conversation, normalized_sources)
            conversation["updated_at"] = timestamp

            self._write_store(store)
            return {
                "conversation": self._conversation_summary(conversation),
                "user_message": self._public_message(user_message),
                "assistant_message": self._public_message(assistant_message),
            }

    def toggle_message_pin(self, conversation_id: str, message_id: str, pinned: Optional[bool] = None) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            conversation = self._find_conversation(store, conversation_id)
            if conversation is None:
                raise ValueError("Conversation not found.")

            message = self._find_message(conversation, message_id)
            if message is None:
                raise ValueError("Message not found.")

            next_value = (not bool(message.get("pinned"))) if pinned is None else bool(pinned)
            message["pinned"] = next_value
            conversation["updated_at"] = self._timestamp()
            self._write_store(store)
            return {
                "conversation": self._conversation_summary(conversation),
                "message": self._public_message(message),
            }

    def create_prompt(self, title: str, text: str, length: str = "default") -> Dict[str, Any]:
        with self._lock:
            normalized_title = self._normalize_title(title)
            normalized_text = str(text or "").strip()
            normalized_length = self._normalize_length(length)
            if not normalized_title:
                raise ValueError("Prompt title required.")
            if not normalized_text:
                raise ValueError("Prompt text required.")
            if len(normalized_text) > 1000:
                raise ValueError("Prompt text must not exceed 1000 characters.")

            store = self._read_store()
            timestamp = self._timestamp()
            prompt = {
                "id": str(uuid.uuid4()),
                "title": normalized_title,
                "text": normalized_text,
                "length": normalized_length,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            store["prompts"].append(prompt)
            self._write_store(store)
            return dict(prompt)

    def update_prompt(self, prompt_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            prompt = self._find_prompt(store, prompt_id)
            if prompt is None:
                raise ValueError("Prompt not found.")

            if "title" in updates:
                normalized_title = self._normalize_title(updates.get("title"))
                if not normalized_title:
                    raise ValueError("Prompt title required.")
                prompt["title"] = normalized_title

            if "text" in updates:
                normalized_text = str(updates.get("text") or "").strip()
                if not normalized_text:
                    raise ValueError("Prompt text required.")
                if len(normalized_text) > 1000:
                    raise ValueError("Prompt text must not exceed 1000 characters.")
                prompt["text"] = normalized_text

            if "length" in updates:
                prompt["length"] = self._normalize_length(updates.get("length"))

            prompt["updated_at"] = self._timestamp()
            self._write_store(store)
            return dict(prompt)

    def delete_prompt(self, prompt_id: str) -> None:
        with self._lock:
            store = self._read_store()
            prompt = self._find_prompt(store, prompt_id)
            if prompt is None:
                raise ValueError("Prompt not found.")
            store["prompts"] = [item for item in store["prompts"] if item["id"] != prompt_id]
            self._write_store(store)

    def export_conversation_notes(self, conversation_id: str) -> Dict[str, Any]:
        with self._lock:
            store = self._read_store()
            conversation = self._find_conversation(store, conversation_id)
            if conversation is None:
                raise ValueError("Conversation not found.")

            conversation["export_count"] = int(conversation.get("export_count") or 0) + 1
            conversation["updated_at"] = self._timestamp()
            markdown = self._conversation_to_markdown(conversation)
            self._write_store(store)
            return {
                "filename": self._export_filename(conversation),
                "content": markdown,
            }

    def _conversation_to_markdown(self, conversation: Dict[str, Any]) -> str:
        lines = [
            f"# {conversation.get('title') or 'Untitled workspace'}",
            "",
        ]
        tags = conversation.get("tags") or []
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
            lines.append("")

        pinned_messages = [message for message in conversation.get("messages", []) if message.get("pinned")]
        if pinned_messages:
            lines.extend([
                "## Pinned Notes",
                "",
            ])
            for message in pinned_messages:
                lines.append(f"- {self._single_line(message.get('content') or '')}")
            lines.append("")

        lines.extend([
            "## Conversation",
            "",
        ])
        for message in conversation.get("messages", []):
            role = "Researcher" if message.get("role") == "user" else "Assistant"
            lines.append(f"### {role}")
            lines.append("")
            lines.append(str(message.get("content") or "").strip() or "(empty)")
            lines.append("")
            sources = message.get("sources") or []
            if sources:
                lines.append("Sources:")
                for index, source in enumerate(sources, start=1):
                    lines.append(f"- [{index}] {self._build_source_reference(source)}")
                lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _conversation_summary(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        messages = conversation.get("messages", [])
        last_message = messages[-1] if messages else None
        pinned_count = sum(1 for message in messages if message.get("pinned"))
        return {
            "id": conversation["id"],
            "title": conversation.get("title") or "Untitled workspace",
            "title_auto": bool(conversation.get("title_auto", True)),
            "tags": list(conversation.get("tags") or []),
            "created_at": conversation.get("created_at"),
            "updated_at": conversation.get("updated_at"),
            "deleted_at": conversation.get("deleted_at"),
            "message_count": len(messages),
            "pinned_count": pinned_count,
            "last_message_preview": self._single_line(last_message.get("content") if last_message else ""),
            "selected_item_ids": list(conversation.get("selected_item_ids") or []),
            "response_mode": conversation.get("response_mode") or "",
        }

    def _conversation_detail(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        detail = self._conversation_summary(conversation)
        detail["messages"] = [self._public_message(message) for message in conversation.get("messages", [])]
        return detail

    def _public_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "id": message["id"],
            "role": message.get("role") or "assistant",
            "content": str(message.get("content") or ""),
            "created_at": message.get("created_at"),
            "pinned": bool(message.get("pinned")),
            "selected_item_ids": list(message.get("selected_item_ids") or []),
        }
        if payload["role"] == "assistant":
            payload["sources"] = [self._normalize_source(source) for source in message.get("sources") or []]
            payload["usage"] = dict(message.get("usage") or {})
        return payload

    def get_context_filter(self) -> Dict[str, Any]:
        with self._lock:
            if not os.path.exists(self._context_filter_path):
                return {"selected_item_ids": []}
            try:
                with open(self._context_filter_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return {"selected_item_ids": self._normalize_id_list(data.get("selected_item_ids"))}
            except Exception:
                return {"selected_item_ids": []}

    def update_context_filter(self, selected_item_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        with self._lock:
            normalized = self._normalize_id_list(selected_item_ids)
            data = {"selected_item_ids": normalized}
            descriptor, temp_path = tempfile.mkstemp(dir=self.instance_path)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
                os.chmod(temp_path, 0o600)
                os.replace(temp_path, self._context_filter_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            return data

    def _normalize_store(self, store: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "conversations": [],
            "prompts": [],
        }

        if not isinstance(store, dict):
            return normalized

        for raw_conversation in store.get("conversations", []):
            conversation = self._normalize_conversation(raw_conversation)
            if conversation:
                normalized["conversations"].append(conversation)

        for raw_prompt in store.get("prompts", []):
            prompt = self._normalize_prompt(raw_prompt)
            if prompt:
                normalized["prompts"].append(prompt)

        return normalized

    def _normalize_conversation(self, raw_conversation: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_conversation, dict):
            return None

        conversation_id = str(raw_conversation.get("id") or "").strip()
        if not conversation_id:
            return None

        created_at = str(raw_conversation.get("created_at") or self._timestamp())
        updated_at = str(raw_conversation.get("updated_at") or created_at)
        messages = []
        for raw_message in raw_conversation.get("messages", []):
            message = self._normalize_message(raw_message)
            if message:
                messages.append(message)

        title = self._normalize_title(raw_conversation.get("title"))
        if not title:
            first_user_message = next((message for message in messages if message.get("role") == "user" and message.get("content")), None)
            title = self._build_auto_title(first_user_message.get("content") if first_user_message else "")

        conversation = {
            "id": conversation_id,
            "title": title or "Untitled workspace",
            "title_auto": bool(raw_conversation.get("title_auto", True)),
            "tags": self._normalize_tags(raw_conversation.get("tags")),
            "created_at": created_at,
            "updated_at": updated_at,
            "deleted_at": raw_conversation.get("deleted_at"),
            "messages": messages,
            "selected_item_ids": self._normalize_id_list(raw_conversation.get("selected_item_ids")),
            "export_count": max(int(raw_conversation.get("export_count") or 0), 0),
            "response_mode": self._normalize_response_mode(raw_conversation.get("response_mode")),
        }
        if not conversation["tags"]:
            assistant_sources = []
            for message in messages:
                assistant_sources.extend(message.get("sources") or [])
            conversation["tags"] = self._derive_tags(conversation, assistant_sources)
        return conversation

    def _normalize_message(self, raw_message: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_message, dict):
            return None

        message_id = str(raw_message.get("id") or "").strip()
        role = str(raw_message.get("role") or "").strip().lower()
        if role not in {"user", "assistant"} or not message_id:
            return None

        message = {
            "id": message_id,
            "role": role,
            "content": str(raw_message.get("content") or "").strip(),
            "created_at": str(raw_message.get("created_at") or self._timestamp()),
            "pinned": bool(raw_message.get("pinned")),
            "selected_item_ids": self._normalize_id_list(raw_message.get("selected_item_ids")),
        }
        if role == "assistant":
            message["sources"] = [self._normalize_source(source) for source in raw_message.get("sources") or []]
            message["usage"] = dict(raw_message.get("usage") or {})
        return message

    def _normalize_prompt(self, raw_prompt: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_prompt, dict):
            return None

        prompt_id = str(raw_prompt.get("id") or "").strip()
        title = self._normalize_title(raw_prompt.get("title"))
        text = str(raw_prompt.get("text") or "").strip()
        length = self._normalize_length(raw_prompt.get("length"))
        if not prompt_id or not title or not text:
            return None

        created_at = str(raw_prompt.get("created_at") or self._timestamp())
        updated_at = str(raw_prompt.get("updated_at") or created_at)
        return {
            "id": prompt_id,
            "title": title,
            "text": text[:1000],
            "length": length,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def _normalize_length(self, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in VALID_LENGTH_CONFIGS else "default"

    def _normalize_response_mode(self, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in ("literal", "paraphrase", "synthesis") else ""

    def _normalize_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(source or {})
        return {
            "source_id": str(normalized.get("source_id") or normalized.get("key") or normalized.get("item_id") or ""),
            "item_id": str(normalized.get("item_id") or ""),
            "key": str(normalized.get("key") or ""),
            "title": str(normalized.get("title") or "Untitled source"),
            "authors": str(normalized.get("authors") or ""),
            "date": str(normalized.get("date") or ""),
            "publication": str(normalized.get("publication") or ""),
            "document_source": str(normalized.get("document_source") or ""),
            "attachment_label": str(normalized.get("attachment_label") or ""),
            "section_heading": str(normalized.get("section_heading") or ""),
            "content_type": str(normalized.get("content_type") or ""),
            "full_reference": str(normalized.get("full_reference") or ""),
            "pages": str(normalized.get("pages") or ""),
            "score": normalized.get("score"),
            "text": str(normalized.get("text") or ""),
            "zotero_open_uri": str(normalized.get("zotero_open_uri") or ""),
            "zotero_select_uri": str(normalized.get("zotero_select_uri") or ""),
        }

    def _derive_tags(self, conversation: Dict[str, Any], sources: List[Dict[str, Any]]) -> List[str]:
        text_fragments = []
        for message in conversation.get("messages", []):
            if message.get("role") == "user":
                text_fragments.append(message.get("content") or "")
        for source in sources:
            text_fragments.extend([
                source.get("title") or "",
                source.get("authors") or "",
                source.get("publication") or "",
                source.get("document_source") or "",
            ])

        tokens = []
        for fragment in text_fragments:
            for token in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9][A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9'\-]{2,}", str(fragment or "")):
                folded = token.casefold()
                if folded in DEFAULT_STOP_WORDS or folded.isdigit():
                    continue
                tokens.append(token)

        counts = Counter(tokens)
        tags = []
        for token, _ in counts.most_common(8):
            cleaned = token.strip("- '")
            if not cleaned:
                continue
            normalized = cleaned[0].upper() + cleaned[1:]
            if normalized.casefold() in {tag.casefold() for tag in tags}:
                continue
            tags.append(normalized)
            if len(tags) >= 4:
                break

        return tags

    def _build_auto_title(self, question: str) -> str:
        text = self._single_line(question)
        if not text:
            return "Untitled workspace"

        sentence_break = re.search(r"[.?!]\s", text)
        if sentence_break:
            text = text[:sentence_break.start() + 1]

        if len(text) > 72:
            text = text[:69].rstrip() + "..."

        return text or "Untitled workspace"

    def _build_source_reference(self, source: Dict[str, Any]) -> str:
        authors = str(source.get("authors") or "").strip()
        date = str(source.get("date") or "").strip()
        title = str(source.get("title") or "Untitled source").strip() or "Untitled source"
        publication = str(source.get("publication") or "").strip()

        lead = authors
        if date:
            lead = f"{lead} ({date})" if lead else date
        reference = f"{lead} — {title}" if lead else title
        if publication:
            reference = f"{reference}. {publication}"
        return reference

    def _single_line(self, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) > 140:
            return text[:137].rstrip() + "..."
        return text

    def _normalize_title(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _normalize_tags(self, tags: Any) -> List[str]:
        if tags in (None, ""):
            return []
        candidates = tags if isinstance(tags, list) else [tags]
        normalized = []
        seen = set()
        for candidate in candidates:
            tag = self._normalize_title(candidate)
            if not tag:
                continue
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(tag)
        return normalized[:6]

    def _normalize_id_list(self, item_ids: Any) -> List[str]:
        if item_ids in (None, ""):
            return []
        values = item_ids if isinstance(item_ids, list) else [item_ids]
        normalized = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    def _export_filename(self, conversation: Dict[str, Any]) -> str:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", conversation.get("title") or "researcharr-notes").strip("-") or "researcharr-notes"
        return f"{stem}.md"

    def _find_conversation(self, store: Dict[str, Any], conversation_id: str) -> Optional[Dict[str, Any]]:
        return next((item for item in store["conversations"] if item["id"] == conversation_id), None)

    def _find_prompt(self, store: Dict[str, Any], prompt_id: str) -> Optional[Dict[str, Any]]:
        return next((item for item in store["prompts"] if item["id"] == prompt_id), None)

    def _find_message(self, conversation: Dict[str, Any], message_id: str) -> Optional[Dict[str, Any]]:
        return next((item for item in conversation.get("messages", []) if item["id"] == message_id), None)

    def _ensure_store_exists(self) -> None:
        if os.path.exists(self.store_path):
            return
        self._write_store(dict(DEFAULT_STORE))

    def _read_store(self) -> Dict[str, Any]:
        if not os.path.exists(self.store_path):
            return dict(DEFAULT_STORE)

        with open(self.store_path, "r", encoding="utf-8") as handle:
            return self._normalize_store(json.load(handle))

    def _write_store(self, store: Dict[str, Any]) -> None:
        normalized = self._normalize_store(store)
        descriptor, temp_path = tempfile.mkstemp(dir=self.instance_path)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(normalized, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.store_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _ensure_directory(self, path: str, mode: int) -> None:
        os.makedirs(path, exist_ok=True)
        try:
            os.chmod(path, mode)
        except PermissionError:
            pass