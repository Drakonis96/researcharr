import asyncio
import os
import json
import re
import socket
import subprocess
import sys
import signal
import traceback
from typing import Any, Dict, List, Optional

from zotero_reader import ZoteroReader
from vector_store import VectorStore
from chat_store import ChatStore
from settings_store import SettingsStore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')

FLASK_HOST = (os.environ.get('HOST') or '127.0.0.1').strip() or '127.0.0.1'
FLASK_PORT = int(os.environ.get('PORT', '5000'))
MCP_HOST = (os.environ.get('MCP_HOST') or '127.0.0.1').strip() or '127.0.0.1'
MCP_PORT = int(os.environ.get('MCP_PORT', '5001'))

zotero = ZoteroReader()
vector_store = VectorStore()
settings_store = SettingsStore(INSTANCE_DIR)
chat_store = ChatStore(INSTANCE_DIR)

PID_FILE = os.path.join(INSTANCE_DIR, 'mcp_server.pid')
STATUS_FILE = os.path.join(INSTANCE_DIR, 'mcp_server.json')
MCP_SSE_PATH = '/sse'
MCP_MESSAGE_PATH = '/messages/'
DEFAULT_MCP_RESPONSE_MODE = 'summary'
CITATION_MARKER_PATTERN = re.compile(r'\[(\d+)\]')


def write_status(host, port, pid=None):
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    pid = pid or os.getpid()
    status = {
        'host': host,
        'port': port,
        'pid': pid,
        'status': 'running',
    }
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)
    with open(PID_FILE, 'w') as f:
        f.write(str(pid))


def _flask_url(path):
    return f"http://{FLASK_HOST}:{FLASK_PORT}{path}"


def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return MCP_HOST


def normalize_mcp_response_mode(value: Any) -> tuple[str, str]:
    normalized = str(value or '').strip().lower()
    if normalized == 'literal':
        return 'literal', 'literal'
    if normalized == 'paraphrase':
        return 'paraphrase', 'paraphrase'
    return 'synthesis', DEFAULT_MCP_RESPONSE_MODE


def public_mcp_response_mode(value: Any) -> str:
    return normalize_mcp_response_mode(value)[1]


def build_mcp_citation_metadata(answer_text: Any, sources: Any) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    normalized_answer = str(answer_text or '')
    normalized_sources = [dict(source or {}) for source in (sources or []) if isinstance(source, dict)]
    citations: list[Dict[str, Any]] = []
    cited_sources: list[Dict[str, Any]] = []
    seen_sources = set()

    for match in CITATION_MARKER_PATTERN.finditer(normalized_answer):
        source_number = int(match.group(1))
        if source_number < 1 or source_number > len(normalized_sources):
            continue

        source = dict(normalized_sources[source_number - 1])
        source_identifier = str(source.get('source_id') or source.get('key') or source.get('item_id') or source_number)
        citations.append({
            'occurrence_index': len(citations),
            'occurrence_number': len(citations) + 1,
            'source_number': source_number,
            'marker': match.group(0),
            'source': source,
        })

        dedupe_key = (source_number, source_identifier)
        if dedupe_key in seen_sources:
            continue

        seen_sources.add(dedupe_key)
        cited_source = dict(source)
        cited_source['source_number'] = source_number
        cited_sources.append(cited_source)

    return citations, cited_sources


class MCPBearerAuthMiddleware:
    def __init__(self, app):
        self.app = app

    def _requires_auth(self, scope):
        path = scope.get('path') or ''
        normalized_message_path = MCP_MESSAGE_PATH.rstrip('/')
        return path == MCP_SSE_PATH or path == normalized_message_path or path.startswith(f'{normalized_message_path}/')

    def _extract_bearer_token(self, scope):
        for header_name, header_value in scope.get('headers', []):
            if header_name.decode('latin-1').lower() != 'authorization':
                continue
            value = header_value.decode('latin-1').strip()
            if not value.lower().startswith('bearer '):
                return ''
            return value[7:].strip()
        return ''

    async def _send_unauthorized(self, send):
        body = json.dumps({'error': 'Unauthorized MCP request. Provide Authorization: Bearer <api-key>.'}).encode('utf-8')
        headers = [
            (b'content-type', b'application/json; charset=utf-8'),
            (b'content-length', str(len(body)).encode('ascii')),
            (b'cache-control', b'no-store'),
            (b'www-authenticate', b'Bearer'),
        ]
        await send({
            'type': 'http.response.start',
            'status': 401,
            'headers': headers,
        })
        await send({
            'type': 'http.response.body',
            'body': body,
        })

    async def __call__(self, scope, receive, send):
        if scope.get('type') != 'http' or not self._requires_auth(scope):
            await self.app(scope, receive, send)
            return

        if not settings_store.has_mcp_keys():
            await self.app(scope, receive, send)
            return

        token = self._extract_bearer_token(scope)
        if token and settings_store.validate_mcp_key(token):
            await self.app(scope, receive, send)
            return

        await self._send_unauthorized(send)


def create_mcp_server():
    try:
        from mcp.server.fastmcp import Context, FastMCP
    except ImportError:
        print("MCP package not installed. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    mcp = FastMCP(
        "Researcharr",
        instructions=(
            "Researcharr MCP Server — access your Zotero research library. "
            "Use list_collections to browse the library structure, get_collection to explore items, "
            "query_library to ask questions about indexed documents, and chat tools to manage conversations."
        ),
        host=MCP_HOST,
        port=MCP_PORT,
        sse_path=MCP_SSE_PATH,
        message_path=MCP_MESSAGE_PATH,
    )

    @mcp.tool()
    def list_collections() -> str:
        """List all collections in the Zotero library with their IDs, names, parent collections, and item counts."""
        collections = zotero.get_collections()
        result = []
        for c in collections:
            result.append({
                "id": c["id"],
                "name": c["name"],
                "parentId": c.get("parentId"),
                "itemCount": c.get("itemCount", 0),
            })
        return json.dumps(result, indent=2, ensure_ascii=False)

    @mcp.tool()
    def get_collection(collection_id: int, include_subcollections: bool = False) -> str:
        """Get details and items of a specific collection by ID. Set include_subcollections=True to include items from nested subcollections."""
        try:
            items = zotero.get_items(
                collection_id=collection_id,
                include_subcollections=include_subcollections,
                limit=100,
            )
            collections = zotero.get_collections()
            collection = next((c for c in collections if c["id"] == collection_id), None)
            subcollections = [c for c in collections if c.get("parentId") == collection_id]

            result = {
                "collection": collection,
                "subcollections": subcollections,
                "items": [
                    {
                        "id": item["id"],
                        "key": item.get("key", ""),
                        "title": item.get("title", ""),
                        "type": item.get("type", ""),
                        "date": item.get("date", ""),
                        "creators": item.get("creators", []),
                        "tags": item.get("tags", []),
                        "hasPdfAttachment": item.get("hasPdfAttachment", False),
                    }
                    for item in items
                ],
            }
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool()
    def get_item(item_id: int) -> str:
        """Get detailed information about a specific Zotero item by its ID, including metadata, creators, and attachments."""
        try:
            items = zotero.get_items(limit=10000)
            item = next((i for i in items if i["id"] == item_id), None)
            if not item:
                return json.dumps({"error": f"Item {item_id} not found."})

            targets = zotero.get_item_open_targets(item_id)
            result = {
                "id": item["id"],
                "key": item.get("key", ""),
                "title": item.get("title", ""),
                "type": item.get("type", ""),
                "date": item.get("date", ""),
                "publication": item.get("publication", ""),
                "abstract": item.get("abstractNote", ""),
                "DOI": item.get("DOI", ""),
                "URL": item.get("URL", ""),
                "creators": item.get("creators", []),
                "tags": item.get("tags", []),
                "collections": item.get("collections", []),
                "hasPdfAttachment": item.get("hasPdfAttachment", False),
                "hasPdfText": item.get("hasPdfText", False),
                "zotero_open_uri": targets.get("zotero_open_uri", ""),
                "zotero_select_uri": targets.get("zotero_select_uri", ""),
            }
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool()
    def list_subcollections(parent_id: int) -> str:
        """List all direct subcollections of a given collection ID."""
        collections = zotero.get_collections()
        children = [c for c in collections if c.get("parentId") == parent_id]
        result = [
            {"id": c["id"], "name": c["name"], "parentId": c.get("parentId"), "itemCount": c.get("itemCount", 0)}
            for c in children
        ]
        return json.dumps(result, indent=2, ensure_ascii=False)

    @mcp.tool()
    def search_items(query: str, limit: int = 20) -> str:
        """Search for items in the Zotero library by title. Returns items whose title contains the search query (case-insensitive)."""
        try:
            all_items = zotero.get_items(limit=10000)
            query_lower = query.lower()
            matched = [i for i in all_items if query_lower in (i.get("title") or "").lower()]
            result = [
                {
                    "id": item["id"],
                    "key": item.get("key", ""),
                    "title": item.get("title", ""),
                    "type": item.get("type", ""),
                    "date": item.get("date", ""),
                    "creators": item.get("creators", []),
                    "collections": item.get("collections", []),
                }
                for item in matched[:limit]
            ]
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool()
    def get_library_stats() -> str:
        """Get statistics about the Zotero library: total items, indexed items, and collection count."""
        try:
            total = zotero.get_item_count()
            indexed = vector_store.get_stats()['count']
            collections = zotero.get_collections()
            return json.dumps({
                "totalItems": total,
                "indexedItems": indexed,
                "collectionCount": len(collections),
            }, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool()
    async def query_library(
        question: str,
        response_mode: str = DEFAULT_MCP_RESPONSE_MODE,
        top_k: int = 5,
        conversation_id: str = "",
        active_prompt_id: str = "",
        ctx: Context | None = None,
    ) -> str:
        """Ask a question about your indexed Zotero library. The server retrieves relevant documents and generates an answer with citations.

        Args:
            question: The research question to ask about your library.
            response_mode: How to format citations. One of: summary (default), synthesis, literal, paraphrase.
            top_k: Number of sources to retrieve (default 5).
            conversation_id: Optional existing conversation id to append the exchange to.
            active_prompt_id: Optional saved prompt id to apply to this request.
        """
        import requests as http_requests

        try:
            active_credential = settings_store.get_active_credential()
            if not active_credential:
                return json.dumps({"error": "No active API key configured. Save and activate a credential in Settings first."})

            config = settings_store.get_config()
            if not config.get("chat_model"):
                return json.dumps({"error": "No chat model configured. Configure one in Settings."})

            embedding_model = config.get("embeddings_model", "")
            if not embedding_model:
                return json.dumps({"error": "No embeddings model configured."})

            indexed_ids = vector_store.get_indexed_item_ids()
            if not indexed_ids:
                return json.dumps({"error": "No documents indexed yet. Run indexing first."})

            question_clean = question.strip()
            if not question_clean:
                return json.dumps({"error": "Question is required."})

            effective_mode, public_mode = normalize_mcp_response_mode(response_mode)
            requested_top_k = max(int(top_k or 5), 1)
            normalized_conversation_id = str(conversation_id or '').strip()
            normalized_prompt_id = str(active_prompt_id or '').strip()

            flask_url = _flask_url("/api/query")
            payload = {
                "question": question_clean,
                "response_mode": effective_mode,
                "top_k": requested_top_k,
            }
            if normalized_conversation_id:
                payload["conversation_id"] = normalized_conversation_id
            if normalized_prompt_id:
                payload["active_prompt_id"] = normalized_prompt_id

            request_task = asyncio.create_task(
                asyncio.to_thread(
                    http_requests.post,
                    flask_url,
                    json=payload,
                    timeout=120,
                )
            )

            heartbeat_count = 0
            while not request_task.done():
                done, _ = await asyncio.wait({request_task}, timeout=15)
                if request_task in done:
                    break
                heartbeat_count += 1
                if ctx is not None:
                    message = "Researcharr is still retrieving sources and generating the answer."
                    await ctx.info(message)
                    await ctx.report_progress(heartbeat_count, heartbeat_count + 1, message=message)

            response = await request_task

            if response.status_code != 200:
                try:
                    error_data = response.json()
                    return json.dumps({"error": error_data.get("error", f"HTTP {response.status_code}")})
                except Exception:
                    return json.dumps({"error": f"HTTP {response.status_code}"})

            data = response.json()
            if "error" in data:
                return json.dumps({"error": data["error"]})

            sources = data.get("sources", [])
            citations, cited_sources = build_mcp_citation_metadata(data.get("answer", ""), sources)

            result = {
                "question": question_clean,
                "answer": data.get("answer", ""),
                "sources": sources,
                "citations": citations,
                "cited_sources": cited_sources,
                "citation_count": len(citations),
                "source_count": len(sources),
                "conversation_id": data.get("conversation_id", ""),
                "response_mode": public_mode,
            }
            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return json.dumps({"error": str(exc)})

    @mcp.tool()
    def list_chats(limit: int = 20) -> str:
        """List recent chat conversations in Researcharr."""
        try:
            state = chat_store.get_workspace_state()
            conversations = state.get("conversations", [])[:limit]
            result = [
                {
                    "id": c.get("id", ""),
                    "title": c.get("title", ""),
                    "message_count": c.get("message_count", 0),
                    "created_at": c.get("created_at", ""),
                    "updated_at": c.get("updated_at", ""),
                    "response_mode": public_mcp_response_mode(c.get("response_mode", "")),
                }
                for c in conversations
            ]
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool()
    def create_chat(title: str = "") -> str:
        """Create a new empty chat conversation in Researcharr.

        Args:
            title: Optional title for the conversation. Leave empty for auto-generated title.
        """
        try:
            conversation = chat_store.create_conversation(title=title)
            conversation["response_mode"] = public_mcp_response_mode(conversation.get("response_mode", ""))
            conversation["default_response_mode"] = DEFAULT_MCP_RESPONSE_MODE
            return json.dumps(conversation, indent=2, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool()
    def get_chat(conversation_id: str) -> str:
        """Get the full conversation including all messages and sources for a specific chat by ID."""
        try:
            conversation = chat_store.get_conversation(conversation_id)
            messages = []
            for msg in conversation.get("messages", []):
                entry = {
                    "id": msg.get("id", ""),
                    "role": msg.get("role", ""),
                    "content": msg.get("content", ""),
                    "created_at": msg.get("created_at", ""),
                }
                if msg.get("role") == "assistant" and msg.get("sources"):
                    sources = msg["sources"]
                    citations, cited_sources = build_mcp_citation_metadata(msg.get("content", ""), sources)
                    entry["sources"] = sources
                    entry["citations"] = citations
                    entry["cited_sources"] = cited_sources
                messages.append(entry)

            result = {
                "id": conversation.get("id", ""),
                "title": conversation.get("title", ""),
                "message_count": len(messages),
                "created_at": conversation.get("created_at", ""),
                "updated_at": conversation.get("updated_at", ""),
                "response_mode": public_mcp_response_mode(conversation.get("response_mode", "")),
                "default_response_mode": DEFAULT_MCP_RESPONSE_MODE,
                "messages": messages,
            }
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool()
    def get_collections_tree() -> str:
        """Get the full hierarchical tree of collections with their nested subcollections."""
        try:
            tree = zotero.get_collections_tree()
            return json.dumps(tree, indent=2, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return mcp


def run_server():
    mcp = create_mcp_server()
    write_status(MCP_HOST, MCP_PORT)

    local_ip = _get_local_ip()
    print(f"Researcharr MCP Server starting on {MCP_HOST}:{MCP_PORT}", file=sys.stderr)
    print(f"Local IP: {local_ip}", file=sys.stderr)
    print(f"Flask API: http://{FLASK_HOST}:{FLASK_PORT}", file=sys.stderr)

    try:
        import uvicorn

        app = MCPBearerAuthMiddleware(mcp.sse_app())
        config = uvicorn.Config(
            app,
            host=MCP_HOST,
            port=MCP_PORT,
            log_level=mcp.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        pass
    finally:
            for cleanup_file in (STATUS_FILE, PID_FILE):
                if os.path.exists(cleanup_file):
                    os.remove(cleanup_file)


def get_server_status():
    status = {
        'host': MCP_HOST,
        'port': MCP_PORT,
        'local_ip': _get_local_ip(),
        'status': 'unknown',
        'pid': None,
    }

    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            status.update(saved)
        except Exception:
            pass

    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            status['pid'] = pid
            os.kill(pid, 0)
            status['status'] = 'running'
        except (ProcessLookupError, ValueError, OSError):
            status['status'] = 'stopped'
    elif os.path.exists(STATUS_FILE):
        pid = status.get('pid')
        if pid:
            try:
                os.kill(pid, 0)
                status['status'] = 'running'
            except (ProcessLookupError, OSError):
                status['status'] = 'stopped'
        else:
            status['status'] = 'unknown'
    else:
        status['status'] = 'not_configured'

    return status


if __name__ == '__main__':
    run_server()