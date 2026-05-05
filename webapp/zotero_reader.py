import sqlite3
import shutil
import os
import re
import threading
from html import unescape
from functools import wraps
from typing import List, Dict, Any

try:
    import fitz as _fitz
except ImportError:
    _fitz = None


REGULAR_ITEMS_CLAUSE = (
    "{alias}.itemID NOT IN (SELECT itemID FROM itemAttachments) "
    "AND {alias}.itemID NOT IN (SELECT itemID FROM itemNotes) "
    "AND {alias}.itemID NOT IN (SELECT itemID FROM itemAnnotations)"
)

TEXT_ATTACHMENT_EXTENSIONS = {".htm", ".html", ".md", ".markdown", ".txt", ".xhtml", ".xml"}
TEXT_ATTACHMENT_CONTENT_TYPES = {
    "application/xhtml+xml",
    "application/xml",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/xml",
}
HTML_ATTACHMENT_EXTENSIONS = {".htm", ".html", ".xhtml", ".xml"}
HTML_ATTACHMENT_CONTENT_TYPES = {
    "application/xhtml+xml",
    "application/xml",
    "text/html",
    "text/xml",
}
DOCUMENT_ATTACHMENT_EXTENSIONS = {".doc", ".docx", ".epub", ".md", ".markdown", ".odt", ".pdf", ".rtf", ".txt"}
DOCUMENT_ATTACHMENT_CONTENT_TYPES = {
    "application/epub+zip",
    "application/msword",
    "application/pdf",
    "application/rtf",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/markdown",
    "text/plain",
}


def with_db_lock(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._db_lock:
            return method(self, *args, **kwargs)

    return wrapper

class ZoteroReader:
    def __init__(self, db_path: str = None):
        default_db_path = os.path.expanduser("~/Zotero/zotero.sqlite")
        resolved_db_path = db_path or os.environ.get("ZOTERO_DB_PATH") or default_db_path
        resolved_db_path = os.path.expanduser(resolved_db_path)

        default_storage_root = os.path.join(os.path.dirname(resolved_db_path), "storage")
        resolved_storage_root = os.environ.get("ZOTERO_STORAGE_DIR") or default_storage_root

        self.original_db = resolved_db_path
        self.working_db = "/tmp/zotero-researcharr.sqlite"
        self.storage_root = os.path.expanduser(resolved_storage_root)
        self._attachment_text_cache = {}
        self._db_lock = threading.RLock()
        self._copy_db()
        self.conn = sqlite3.connect(self.working_db, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
    def _copy_db(self):
        """Copy database to avoid lock conflicts with running Zotero"""
        shutil.copy2(self.original_db, self.working_db)
        
    @with_db_lock
    def refresh(self):
        """Refresh the copy from the original database"""
        self._copy_db()
        self.conn.close()
        self._attachment_text_cache = {}
        self.conn = sqlite3.connect(self.working_db, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def _normalize_attachment_text(self, text: str) -> str:
        if not text:
            return ""

        normalized = unescape(text).replace("\x00", " ")
        normalized = re.sub(r"\r\n?", "\n", normalized)
        normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _strip_html(self, text: str) -> str:
        if not text:
            return ""

        stripped = re.sub(r"<script[^>]*?>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<style[^>]*?>.*?</style>", " ", stripped, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<br\s*/?>", "\n", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"</(p|div|section|article|li|tr|td|h[1-6])>", "\n", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        return self._normalize_attachment_text(stripped)

    def _read_text_file(self, file_path: str) -> str:
        if not file_path or not os.path.isfile(file_path):
            return ""

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file_handle:
                raw_text = file_handle.read()
        except OSError:
            return ""

        suffix = os.path.splitext(file_path)[1].lower()
        if suffix in {".htm", ".html", ".xhtml", ".xml"}:
            return self._strip_html(raw_text)
        return self._normalize_attachment_text(raw_text)

    def _attachment_storage_dir(self, attachment_key: str) -> str:
        return os.path.join(self.storage_root, attachment_key)

    def _resolve_attachment_file_path(self, attachment_key: str, attachment_path: str) -> str:
        raw_path = (attachment_path or "").strip()
        if not raw_path:
            return ""

        if raw_path.startswith("storage:"):
            relative_name = raw_path.split(":", 1)[1]
            if not relative_name:
                return ""
            return os.path.join(self._attachment_storage_dir(attachment_key), relative_name)

        if os.path.isabs(raw_path):
            return raw_path

        return ""

    def _attachment_filename(self, attachment_path: str) -> str:
        raw_path = (attachment_path or "").strip()
        if not raw_path:
            return ""
        return os.path.basename(raw_path.split(":", 1)[-1])

    def _is_html_attachment(self, attachment_path: str, content_type: str) -> bool:
        suffix = os.path.splitext(self._attachment_filename(attachment_path))[1].lower()
        normalized_content_type = (content_type or "").lower()
        return (
            normalized_content_type in HTML_ATTACHMENT_CONTENT_TYPES
            or suffix in HTML_ATTACHMENT_EXTENSIONS
        )

    def _attachment_open_priority(self, attachment_path: str, content_type: str) -> int:
        suffix = os.path.splitext(self._attachment_filename(attachment_path))[1].lower()
        normalized_content_type = (content_type or "").lower()

        if normalized_content_type == "application/pdf" or suffix == ".pdf":
            return 0
        if normalized_content_type == "application/epub+zip" or suffix == ".epub":
            return 1
        if normalized_content_type in DOCUMENT_ATTACHMENT_CONTENT_TYPES or suffix in DOCUMENT_ATTACHMENT_EXTENSIONS:
            return 2
        if normalized_content_type in TEXT_ATTACHMENT_CONTENT_TYPES or suffix in TEXT_ATTACHMENT_EXTENSIONS:
            return 3
        if self._is_html_attachment(attachment_path, content_type):
            return 100
        return 50

    def _extract_pdf_text(self, file_path: str) -> str:
        if _fitz is None or not file_path or not os.path.isfile(file_path):
            return ""
        try:
            doc = _fitz.open(file_path)
            pages = []
            for page in doc:
                page_width = page.rect.width
                blocks = page.get_text("blocks")
                text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
                if not text_blocks:
                    pages.append(page.get_text())
                    continue

                mid_x = page_width / 2
                left_blocks = [b for b in text_blocks if b[2] <= mid_x + page_width * 0.1]
                right_blocks = [b for b in text_blocks if b[0] >= mid_x - page_width * 0.1]
                is_two_column = (
                    len(left_blocks) >= 3
                    and len(right_blocks) >= 3
                    and (len(left_blocks) + len(right_blocks)) >= len(text_blocks)
                )

                if is_two_column:
                    left_col = sorted([b for b in text_blocks if b[0] < mid_x], key=lambda b: b[1])
                    right_col = sorted([b for b in text_blocks if b[0] >= mid_x], key=lambda b: b[1])
                    ordered = left_col + right_col
                else:
                    ordered = sorted(text_blocks, key=lambda b: (b[1], b[0]))

                pages.append("\n".join(b[4].strip() for b in ordered))

            doc.close()
            return self._normalize_attachment_text("\n\n".join(pages))
        except Exception:
            return ""

    def _read_attachment_text(self, attachment_key: str, attachment_path: str, content_type: str, extract_pdf: bool = False) -> str:
        cache_path = os.path.join(self._attachment_storage_dir(attachment_key), ".zotero-ft-cache")
        cached_text = self._read_text_file(cache_path)
        if cached_text:
            return cached_text

        resolved_path = self._resolve_attachment_file_path(attachment_key, attachment_path)
        if not resolved_path:
            return ""

        suffix = os.path.splitext(resolved_path)[1].lower()
        normalized_content_type = (content_type or "").lower()
        if normalized_content_type in TEXT_ATTACHMENT_CONTENT_TYPES or suffix in TEXT_ATTACHMENT_EXTENSIONS:
            return self._read_text_file(resolved_path)

        if extract_pdf and (suffix == ".pdf" or normalized_content_type == "application/pdf"):
            return self._extract_pdf_text(resolved_path)

        return ""

    @with_db_lock
    def get_item_attachment_text_sections(self, item_id: Any, extract_pdf: bool = False) -> List[Dict[str, str]]:
        cache_key = (str(item_id), extract_pdf)
        cached_sections = self._attachment_text_cache.get(cache_key)
        if cached_sections is not None:
            return [dict(section) for section in cached_sections]

        cursor = self.conn.execute(
            """
            SELECT child.key AS attachmentKey, ia.contentType, ia.path
            FROM itemAttachments ia
            JOIN items child ON child.itemID = ia.itemID
            WHERE ia.parentItemID = ?
            ORDER BY CASE
                WHEN LOWER(COALESCE(ia.contentType, '')) = 'application/pdf' THEN 0
                ELSE 1
            END, ia.itemID
            """,
            (item_id,),
        )

        sections = []
        seen_signatures = set()
        for row in cursor.fetchall():
            attachment_key = row["attachmentKey"] or ""
            attachment_path = row["path"] or ""
            content_type = row["contentType"] or ""
            text = self._read_attachment_text(attachment_key, attachment_path, content_type, extract_pdf=extract_pdf)
            if not text:
                continue

            signature = (len(text), text[:1000])
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            filename = os.path.basename(attachment_path.split(":", 1)[-1]) if attachment_path else ""
            sections.append({
                "label": filename or content_type or attachment_key,
                "content_type": content_type,
                "text": text,
            })

        if any((section.get("content_type") or "").lower() == "application/pdf" for section in sections):
            sections = [
                section
                for section in sections
                if (section.get("content_type") or "").lower() not in HTML_ATTACHMENT_CONTENT_TYPES
                and os.path.splitext(str(section.get("label") or ""))[1].lower() not in HTML_ATTACHMENT_EXTENSIONS
            ]

        self._attachment_text_cache[cache_key] = [dict(section) for section in sections]
        return sections
        
    @with_db_lock
    def get_field_map(self) -> Dict[int, str]:
        """Map fieldID to fieldName"""
        cursor = self.conn.execute("SELECT fieldID, fieldName FROM fieldsCombined")
        return {row["fieldID"]: row["fieldName"] for row in cursor.fetchall()}
        
    @with_db_lock
    def get_item_type_map(self) -> Dict[int, str]:
        """Map itemTypeID to typeName"""
        cursor = self.conn.execute("SELECT itemTypeID, typeName FROM itemTypesCombined")
        return {row["itemTypeID"]: row["typeName"] for row in cursor.fetchall()}

    @with_db_lock
    def get_item_open_targets(self, item_id: Any) -> Dict[str, str]:
        """Return Zotero protocol URLs for selecting or opening the item's attachment."""
        parent_row = self.conn.execute(
            """
            SELECT key
            FROM items
            WHERE itemID = ?
            """,
            (item_id,),
        ).fetchone()
        parent_key = parent_row["key"] if parent_row else ""

        attachment_rows = self.conn.execute(
            """
            SELECT i.key AS attachmentKey, ia.contentType AS contentType, ia.path AS path
            FROM itemAttachments ia
            JOIN items i ON i.itemID = ia.itemID
            WHERE ia.parentItemID = ?
            ORDER BY ia.itemID
            """,
            (item_id,),
        ).fetchall()

        select_uri = f"zotero://select/library/items/{parent_key}" if parent_key else ""
        open_uri = select_uri

        preferred_attachment = None
        preferred_priority = None
        for row in attachment_rows:
            attachment_key = row["attachmentKey"] or ""
            if not attachment_key:
                continue

            attachment_path = row["path"] or ""
            content_type = row["contentType"] or ""
            priority = self._attachment_open_priority(attachment_path, content_type)
            if priority >= 100:
                continue

            if preferred_attachment is None or priority < preferred_priority:
                preferred_attachment = row
                preferred_priority = priority

        if preferred_attachment and preferred_attachment["attachmentKey"]:
            attachment_key = preferred_attachment["attachmentKey"]
            attachment_path = preferred_attachment["path"] or ""
            content_type = (preferred_attachment["contentType"] or "").lower()
            suffix = os.path.splitext(self._attachment_filename(attachment_path))[1].lower()

            if content_type == "application/pdf" or suffix == ".pdf":
                open_uri = f"zotero://open-pdf/library/items/{attachment_key}"
            else:
                open_uri = f"zotero://open/library/items/{attachment_key}"

        return {
            "zotero_open_uri": open_uri,
            "zotero_select_uri": select_uri,
        }
        
    @with_db_lock
    def get_creator_type_map(self) -> Dict[int, str]:
        """Map creatorTypeID to creatorType"""
        cursor = self.conn.execute("SELECT creatorTypeID, creatorType FROM creatorTypes")
        return {row["creatorTypeID"]: row["creatorType"] for row in cursor.fetchall()}

    @with_db_lock
    def get_collection_name(self, collection_id: int) -> str:
        row = self.conn.execute(
            "SELECT collectionName FROM collections WHERE collectionID = ?",
            (collection_id,),
        ).fetchone()
        return row["collectionName"] if row else ""

    def get_collection_descendant_ids(self, collection_id: int) -> List[int]:
        collections = self.get_collections()
        children_by_parent = {}
        for collection in collections:
            children_by_parent.setdefault(collection["parentId"], []).append(collection["id"])

        descendants = []
        stack = [collection_id]
        while stack:
            current_id = stack.pop()
            descendants.append(current_id)
            stack.extend(children_by_parent.get(current_id, []))

        return descendants

    def get_collection_path_map(self) -> Dict[int, str]:
        collections = self.get_collections()
        collection_by_id = {collection["id"]: collection for collection in collections}
        path_map = {}

        def resolve_path(collection_id: int) -> str:
            if collection_id in path_map:
                return path_map[collection_id]

            collection = collection_by_id.get(collection_id)
            if not collection:
                return ""

            parent_id = collection.get("parentId")
            parent_path = resolve_path(parent_id) if parent_id else ""
            path = f"{parent_path} / {collection['name']}" if parent_path else collection["name"]
            path_map[collection_id] = path
            return path

        for collection_id in collection_by_id:
            resolve_path(collection_id)

        return path_map

    @with_db_lock
    def summarize_indexed_collections(self, item_ids: List[Any]) -> Dict[str, Any]:
        normalized_ids = []
        seen_ids = set()
        for item_id in item_ids:
            text = str(item_id).strip()
            if not text or text in seen_ids:
                continue
            try:
                normalized_id = int(text)
            except (TypeError, ValueError):
                continue
            normalized_ids.append(normalized_id)
            seen_ids.add(text)

        if not normalized_ids:
            return {
                "collection_count": 0,
                "collections": [],
                "unfiled_item_count": 0,
                "missing_item_count": 0,
                "matched_item_count": 0,
            }

        placeholders = ", ".join("?" for _ in normalized_ids)
        collection_paths = self.get_collection_path_map()
        cursor = self.conn.execute(
            f"""
            SELECT i.itemID, c.collectionID, c.collectionName
            FROM items i
            LEFT JOIN collectionItems ci ON ci.itemID = i.itemID
            LEFT JOIN collections c ON c.collectionID = ci.collectionID
            WHERE i.itemID IN ({placeholders})
            ORDER BY c.collectionName COLLATE NOCASE, c.collectionID
            """,
            tuple(normalized_ids),
        )

        matched_item_ids = set()
        item_collection_ids = {}
        collection_counts = {}

        for row in cursor.fetchall():
            item_id = row["itemID"]
            matched_item_ids.add(item_id)
            item_collection_ids.setdefault(item_id, set())

            collection_id = row["collectionID"]
            collection_name = row["collectionName"]
            if collection_id is None or collection_name is None:
                continue

            if collection_id in item_collection_ids[item_id]:
                continue

            item_collection_ids[item_id].add(collection_id)
            entry = collection_counts.setdefault(collection_id, {
                "id": collection_id,
                "name": collection_paths.get(collection_id) or collection_name,
                "itemCount": 0,
            })
            entry["itemCount"] += 1

        unfiled_item_count = sum(1 for item_id in matched_item_ids if not item_collection_ids.get(item_id))
        collections = sorted(
            collection_counts.values(),
            key=lambda collection: (-collection["itemCount"], collection["name"].casefold()),
        )

        return {
            "collection_count": len(collections),
            "collections": collections,
            "unfiled_item_count": unfiled_item_count,
            "missing_item_count": max(0, len(normalized_ids) - len(matched_item_ids)),
            "matched_item_count": len(matched_item_ids),
        }

    @with_db_lock
    def get_item_type_counts(self) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            f"""
            SELECT itc.typeName AS typeName, COUNT(*) AS itemCount
            FROM items i
            JOIN itemTypesCombined itc ON i.itemTypeID = itc.itemTypeID
            WHERE {REGULAR_ITEMS_CLAUSE.format(alias='i')}
            GROUP BY itc.typeName
            ORDER BY itemCount DESC, itc.typeName COLLATE NOCASE
            """
        )
        return [
            {
                "value": row["typeName"],
                "count": row["itemCount"],
            }
            for row in cursor.fetchall()
        ]

    @with_db_lock
    def get_tag_counts(self, limit: int = 250) -> List[Dict[str, Any]]:
        query = f"""
            SELECT t.name AS tagName, COUNT(*) AS itemCount
            FROM itemTags it
            JOIN tags t ON it.tagID = t.tagID
            JOIN items i ON i.itemID = it.itemID
            WHERE {REGULAR_ITEMS_CLAUSE.format(alias='i')}
            GROUP BY t.name
            ORDER BY itemCount DESC, t.name COLLATE NOCASE
        """
        params = ()
        if limit:
            query += " LIMIT ?"
            params = (limit,)

        cursor = self.conn.execute(query, params)
        return [
            {
                "value": row["tagName"],
                "count": row["itemCount"],
            }
            for row in cursor.fetchall()
        ]
        
    @with_db_lock
    def get_items(
        self,
        limit: int = None,
        collection_id: int = None,
        include_subcollections: bool = False,
        collection_ids: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get all regular items with metadata"""
        field_map = self.get_field_map()
        type_map = self.get_item_type_map()
        creator_type_map = self.get_creator_type_map()
        
        # Get all item data values
        item_data = {}
        cursor = self.conn.execute("""
            SELECT id.itemID, id.fieldID, idv.value 
            FROM itemData id 
            JOIN itemDataValues idv ON id.valueID = idv.valueID
        """)
        for row in cursor.fetchall():
            item_id = row["itemID"]
            field_name = field_map.get(row["fieldID"], "unknown")
            if item_id not in item_data:
                item_data[item_id] = {}
            item_data[item_id][field_name] = row["value"]
            
        # Get creators per item
        creators = {}
        cursor = self.conn.execute("""
            SELECT ic.itemID, ic.creatorTypeID, c.firstName, c.lastName, ic.orderIndex
            FROM itemCreators ic
            JOIN creators c ON ic.creatorID = c.creatorID
            ORDER BY ic.itemID, ic.orderIndex
        """)
        for row in cursor.fetchall():
            item_id = row["itemID"]
            if item_id not in creators:
                creators[item_id] = []
            name = f"{row['firstName'] or ''} {row['lastName'] or ''}".strip()
            creator_type = creator_type_map.get(row["creatorTypeID"], "author")
            creators[item_id].append({
                "name": name,
                "type": creator_type
            })
            
        # Get tags per item
        tags = {}
        cursor = self.conn.execute("""
            SELECT it.itemID, t.name
            FROM itemTags it
            JOIN tags t ON it.tagID = t.tagID
        """)
        for row in cursor.fetchall():
            item_id = row["itemID"]
            if item_id not in tags:
                tags[item_id] = []
            tags[item_id].append(row["name"])
            
        # Get notes per item
        notes = {}
        cursor = self.conn.execute("""
            SELECT itemID, note, title FROM itemNotes WHERE parentItemID IS NOT NULL
        """)
        for row in cursor.fetchall():
            item_id = row["itemID"]
            # Note is stored on the note item itself, need to map by parentItemID
            pass
            
        # Get notes by parentItemID
        notes = {}
        cursor = self.conn.execute("""
            SELECT parentItemID, note, title FROM itemNotes WHERE parentItemID IS NOT NULL
        """)
        for row in cursor.fetchall():
            parent_id = row["parentItemID"]
            if parent_id not in notes:
                notes[parent_id] = []
            note_text = row["note"] or ""
            # Strip HTML if present
            import re
            note_text = re.sub(r'<[^>]+>', ' ', note_text)
            note_text = re.sub(r'\s+', ' ', note_text).strip()
            if note_text:
                notes[parent_id].append(note_text)
                
        # Get collections
        collections = {}
        item_collection_ids = {}
        cursor = self.conn.execute("""
            SELECT ci.itemID, c.collectionID, c.collectionName
            FROM collectionItems ci
            JOIN collections c ON ci.collectionID = c.collectionID
        """)
        for row in cursor.fetchall():
            item_id = row["itemID"]
            if item_id not in collections:
                collections[item_id] = []
            if item_id not in item_collection_ids:
                item_collection_ids[item_id] = []
            collections[item_id].append(row["collectionName"])
            item_collection_ids[item_id].append(row["collectionID"])

        attachment_stats = {}
        cursor = self.conn.execute("""
            SELECT ia.parentItemID, LOWER(COALESCE(ia.contentType, '')) AS contentType,
                   COALESCE(fi.indexedPages, 0) AS indexedPages,
                   COALESCE(fi.totalPages, 0) AS totalPages,
                   COALESCE(fi.indexedChars, 0) AS indexedChars,
                   COALESCE(fi.totalChars, 0) AS totalChars
            FROM itemAttachments ia
            LEFT JOIN fulltextItems fi ON fi.itemID = ia.itemID
            WHERE ia.parentItemID IS NOT NULL
        """)
        for row in cursor.fetchall():
            parent_id = row["parentItemID"]
            stats = attachment_stats.setdefault(parent_id, {
                "hasPdfAttachment": False,
                "hasPdfText": False,
                "pdfIndexedPages": 0,
                "pdfTotalPages": 0,
                "pdfIndexedChars": 0,
                "pdfTotalChars": 0,
            })
            if row["contentType"] == "application/pdf":
                stats["hasPdfAttachment"] = True
                stats["pdfIndexedPages"] = max(stats["pdfIndexedPages"], row["indexedPages"] or 0)
                stats["pdfTotalPages"] = max(stats["pdfTotalPages"], row["totalPages"] or 0)
                stats["pdfIndexedChars"] = max(stats["pdfIndexedChars"], row["indexedChars"] or 0)
                stats["pdfTotalChars"] = max(stats["pdfTotalChars"], row["totalChars"] or 0)
                if (row["indexedPages"] or 0) > 0 or (row["indexedChars"] or 0) > 0:
                    stats["hasPdfText"] = True
            
        # Build item list
        query = """
            SELECT i.itemID, i.itemTypeID, i.dateAdded, i.dateModified, i.key
            FROM items i
            WHERE {regular_items_clause}
        """
        normalized_collection_ids = []
        seen_collection_ids = set()
        for selected_collection_id in collection_ids or []:
            try:
                normalized_collection_id = int(selected_collection_id)
            except (TypeError, ValueError):
                continue
            collection_ids_to_match = [normalized_collection_id]
            if include_subcollections:
                collection_ids_to_match = self.get_collection_descendant_ids(normalized_collection_id)

            for collection_id_to_match in collection_ids_to_match:
                if collection_id_to_match in seen_collection_ids:
                    continue
                seen_collection_ids.add(collection_id_to_match)
                normalized_collection_ids.append(collection_id_to_match)

        if normalized_collection_ids:
            placeholders = ", ".join("?" for _ in normalized_collection_ids)
            query = f"""
                SELECT DISTINCT i.itemID, i.itemTypeID, i.dateAdded, i.dateModified, i.key
                FROM items i
                JOIN collectionItems ci ON i.itemID = ci.itemID
                WHERE ci.collectionID IN ({placeholders})
                AND {REGULAR_ITEMS_CLAUSE.format(alias='i')}
            """
            cursor = self.conn.execute(query, tuple(normalized_collection_ids))
        elif collection_id:
            collection_ids_to_match = [collection_id]
            if include_subcollections:
                collection_ids_to_match = self.get_collection_descendant_ids(collection_id)

            placeholders = ", ".join("?" for _ in collection_ids_to_match)
            query = f"""
                SELECT DISTINCT i.itemID, i.itemTypeID, i.dateAdded, i.dateModified, i.key
                FROM items i
                JOIN collectionItems ci ON i.itemID = ci.itemID
                WHERE ci.collectionID IN ({placeholders})
                AND {REGULAR_ITEMS_CLAUSE.format(alias='i')}
            """
            cursor = self.conn.execute(query, tuple(collection_ids_to_match))
        else:
            cursor = self.conn.execute(query.format(regular_items_clause=REGULAR_ITEMS_CLAUSE.format(alias='i')))
            
        items = []
        for row in cursor.fetchall():
            item_id = row["itemID"]
            data = item_data.get(item_id, {})
            stats = attachment_stats.get(item_id, {})
            
            item = {
                "id": item_id,
                "key": row["key"],
                "type": type_map.get(row["itemTypeID"], "unknown"),
                "dateAdded": row["dateAdded"],
                "dateModified": row["dateModified"],
                "title": data.get("title", ""),
                "abstract": data.get("abstractNote", ""),
                "date": data.get("date", ""),
                "publication": data.get("publicationTitle", ""),
                "journal": data.get("journalAbbreviation", ""),
                "volume": data.get("volume", ""),
                "issue": data.get("issue", ""),
                "pages": data.get("pages", ""),
                "DOI": data.get("DOI", ""),
                "URL": data.get("url", ""),
                "creators": creators.get(item_id, []),
                "tags": tags.get(item_id, []),
                "notes": notes.get(item_id, []),
                "collections": collections.get(item_id, []),
                "collectionIds": item_collection_ids.get(item_id, []),
                "hasPdfAttachment": stats.get("hasPdfAttachment", False),
                "hasPdfText": stats.get("hasPdfText", False),
                "pdfIndexedPages": stats.get("pdfIndexedPages", 0),
                "pdfTotalPages": stats.get("pdfTotalPages", 0),
                "pdfIndexedChars": stats.get("pdfIndexedChars", 0),
                "pdfTotalChars": stats.get("pdfTotalChars", 0),
            }
            items.append(item)
            
        if limit:
            items = items[:limit]
            
        return items
        
    @with_db_lock
    def get_collections(self) -> List[Dict[str, Any]]:
        """Get all collections"""
        cursor = self.conn.execute("""
            SELECT c.collectionID, c.collectionName, c.parentCollectionID,
                   COALESCE(collection_counts.itemCount, 0) AS itemCount
            FROM collections c
            LEFT JOIN (
                SELECT ci.collectionID, COUNT(DISTINCT ci.itemID) AS itemCount
                FROM collectionItems ci
                JOIN items i ON i.itemID = ci.itemID
                WHERE {regular_items_clause}
                GROUP BY ci.collectionID
            ) AS collection_counts ON collection_counts.collectionID = c.collectionID
            ORDER BY c.collectionName
        """.format(regular_items_clause=REGULAR_ITEMS_CLAUSE.format(alias='i')))
        return [
            {
                "id": row["collectionID"],
                "name": row["collectionName"],
                "parentId": row["parentCollectionID"],
                "itemCount": row["itemCount"] or 0,
            }
            for row in cursor.fetchall()
        ]

    @with_db_lock
    def get_collection_items(self, collection_id: int) -> List[Dict[str, Any]]:
        """Get items belonging to a specific collection"""
        field_map = self.get_field_map()
        type_map = self.get_item_type_map()
        creator_type_map = self.get_creator_type_map()

        item_data = {}
        cursor = self.conn.execute("""
            SELECT id.itemID, id.fieldID, idv.value
            FROM itemData id
            JOIN itemDataValues idv ON id.valueID = idv.valueID
        """)
        for row in cursor.fetchall():
            item_id = row["itemID"]
            field_name = field_map.get(row["fieldID"], "unknown")
            if item_id not in item_data:
                item_data[item_id] = {}
            item_data[item_id][field_name] = row["value"]

        creators = {}
        cursor = self.conn.execute("""
            SELECT ic.itemID, ic.creatorTypeID, c.firstName, c.lastName, ic.orderIndex
            FROM itemCreators ic
            JOIN creators c ON ic.creatorID = c.creatorID
            ORDER BY ic.itemID, ic.orderIndex
        """)
        for row in cursor.fetchall():
            item_id = row["itemID"]
            if item_id not in creators:
                creators[item_id] = []
            name = f"{row['firstName'] or ''} {row['lastName'] or ''}".strip()
            if not name:
                continue
            creators[item_id].append({
                "name": name,
                "type": creator_type_map.get(row["creatorTypeID"], "author")
            })

        tags = {}
        cursor = self.conn.execute("""
            SELECT it.itemID, t.name
            FROM itemTags it
            JOIN tags t ON it.tagID = t.tagID
        """)
        for row in cursor.fetchall():
            item_id = row["itemID"]
            if item_id not in tags:
                tags[item_id] = []
            tags[item_id].append(row["name"])

        cursor = self.conn.execute("""
            SELECT i.itemID, i.itemTypeID, i.key
            FROM items i
            JOIN collectionItems ci ON i.itemID = ci.itemID
            WHERE ci.collectionID = ?
            AND i.itemID NOT IN (SELECT itemID FROM itemAttachments WHERE parentItemID IS NULL)
            AND i.itemID NOT IN (SELECT itemID FROM itemNotes WHERE parentItemID IS NULL)
        """, (collection_id,))

        items = []
        for row in cursor.fetchall():
            item_id = row["itemID"]
            data = item_data.get(item_id, {})
            item_creators = creators.get(item_id, [])
            items.append({
                "id": item_id,
                "key": row["key"],
                "type": type_map.get(row["itemTypeID"], "unknown"),
                "title": data.get("title", ""),
                "date": data.get("date", ""),
                "publication": data.get("publicationTitle", ""),
                "authors": ', '.join([creator["name"] for creator in item_creators if creator.get("name")]),
                "creators": item_creators,
                "tags": tags.get(item_id, []),
            })
        return items

    @with_db_lock
    def get_collections_tree(self) -> List[Dict[str, Any]]:
        """Get hierarchical collections tree with items"""
        collections = self.get_collections()
        # Build id -> collection map
        coll_map = {c["id"]: c for c in collections}

        # Add children list and items to each collection
        for c in collections:
            c["children"] = []
            c["items"] = self.get_collection_items(c["id"])

        # Build hierarchy
        roots = []
        for c in collections:
            pid = c["parentId"]
            if pid and pid in coll_map:
                coll_map[pid]["children"].append(c)
            else:
                roots.append(c)

        return roots
        
    @with_db_lock
    def get_item_count(self) -> int:
        """Get total number of regular items"""
        cursor = self.conn.execute("""
            SELECT COUNT(*) as count FROM items
            WHERE {regular_items_clause}
        """.format(regular_items_clause=REGULAR_ITEMS_CLAUSE.format(alias='items')))
        return cursor.fetchone()["count"]
        
    @with_db_lock
    def close(self):
        self._attachment_text_cache = {}
        self.conn.close()
