import json
import re
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import os


class VectorStore:
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.environ.get("RESEARCHARR_VECTOR_STORE_DIR") or "/tmp/researcharr-vectors"
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "index.json")
        self.vectors_path = os.path.join(storage_dir, "vectors.npy")
        self.manifest_path = os.path.join(storage_dir, "manifest.json")
        self.items = []
        self.vectors = None
        self.dimensions = 0
        self.bm25 = None
        self.bm25_corpus_tokens = []
        self._bm25_stale = True
        self.manifest = {
            "history": [],
            "currentIndex": {},
            "lastSuccessfulRun": None,
            "itemSyncState": {},
            "scopeState": {},
        }
        os.makedirs(storage_dir, exist_ok=True)
        self.load()

    def _normalize_item_id(self, item_id: Any) -> str:
        return str(item_id).strip()
        
    def load(self):
        """Load index from disk"""
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                data = json.load(f)
                self.items = data.get("items", [])
                self.dimensions = data.get("dimensions", 0)
        if os.path.exists(self.vectors_path):
            self.vectors = np.load(self.vectors_path)
        else:
            self.vectors = np.array([])
        self._load_manifest()

    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return

        with open(self.manifest_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            self.manifest.update({
                "history": data.get("history", []),
                "currentIndex": data.get("currentIndex", {}),
                "lastSuccessfulRun": data.get("lastSuccessfulRun"),
                "itemSyncState": data.get("itemSyncState", {}),
                "scopeState": data.get("scopeState", {}),
            })

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f)
            
    def save(self):
        """Save index to disk"""
        with open(self.index_path, 'w') as f:
            json.dump({
                "items": self.items,
                "dimensions": self.dimensions,
                "count": self.get_indexed_item_count(),
                "entryCount": len(self.items)
            }, f)
        if self.vectors is not None and len(self.vectors) > 0:
            np.save(self.vectors_path, self.vectors)
        elif os.path.exists(self.vectors_path):
            os.remove(self.vectors_path)

        current_index = self.manifest.get("currentIndex", {})
        if current_index:
            current_index["indexedCount"] = self.get_indexed_item_count()
            current_index["entryCount"] = len(self.items)
            current_index["dimensions"] = self.dimensions
        self._save_manifest()
            
    def add_item(self, item_id: str, text: str, metadata: Dict[str, Any],
                 embedding: List[float]):
        """Add an item with its embedding"""
        self.items.append({
            "id": item_id,
            "text": text,
            "metadata": metadata
        })

        vector = np.array(embedding)
        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector])
        self.dimensions = len(embedding)
        self._bm25_stale = True

    def get_item_sync_states(self) -> Dict[str, Dict[str, Any]]:
        state_map = self.manifest.get("itemSyncState", {}) or {}
        return {
            self._normalize_item_id(item_id): dict(state)
            for item_id, state in state_map.items()
            if isinstance(state, dict)
        }

    def get_item_sync_state(self, item_id: Any) -> Optional[Dict[str, Any]]:
        normalized_id = self._normalize_item_id(item_id)
        state = self.manifest.get("itemSyncState", {}).get(normalized_id)
        return dict(state) if isinstance(state, dict) else None

    def set_item_sync_state(self, item_id: Any, state: Dict[str, Any]):
        normalized_id = self._normalize_item_id(item_id)
        self.manifest.setdefault("itemSyncState", {})[normalized_id] = dict(state)

    def remove_item_sync_states(self, item_ids: List[Any]) -> int:
        if not item_ids:
            return 0

        removed = 0
        state_map = self.manifest.setdefault("itemSyncState", {})
        for item_id in item_ids:
            normalized_id = self._normalize_item_id(item_id)
            if normalized_id in state_map:
                del state_map[normalized_id]
                removed += 1
        return removed

    def get_scope_state(self, scope_key: str) -> Optional[Dict[str, Any]]:
        state = self.manifest.get("scopeState", {}).get(scope_key)
        return dict(state) if isinstance(state, dict) else None

    def set_scope_state(self, scope_key: str, filters: Dict[str, Any], eligible_item_ids: List[Any]):
        normalized_ids = []
        seen_ids = set()
        for item_id in eligible_item_ids:
            normalized_id = self._normalize_item_id(item_id)
            if not normalized_id or normalized_id in seen_ids:
                continue
            seen_ids.add(normalized_id)
            normalized_ids.append(normalized_id)

        scope_state = self.manifest.setdefault("scopeState", {})
        if not normalized_ids:
            scope_state.pop(scope_key, None)
            return

        scope_state[scope_key] = {
            "filters": dict(filters),
            "eligible_item_ids": normalized_ids,
        }

    def predict_orphaned_scope_removals(self, scope_key: str, next_item_ids: List[Any]) -> List[str]:
        scope_state = self.manifest.get("scopeState", {}) or {}
        previous_ids = {
            self._normalize_item_id(item_id)
            for item_id in (scope_state.get(scope_key, {}) or {}).get("eligible_item_ids", [])
            if self._normalize_item_id(item_id)
        }
        next_ids = {
            self._normalize_item_id(item_id)
            for item_id in next_item_ids
            if self._normalize_item_id(item_id)
        }

        candidate_ids = previous_ids - next_ids
        if not candidate_ids:
            return []

        retained_ids = set()
        for other_scope_key, other_scope in scope_state.items():
            if not isinstance(other_scope, dict):
                continue

            referenced_ids = {
                self._normalize_item_id(item_id)
                for item_id in other_scope.get("eligible_item_ids", [])
                if self._normalize_item_id(item_id)
            }
            if other_scope_key == scope_key:
                referenced_ids = next_ids

            retained_ids.update(candidate_ids.intersection(referenced_ids))

        return sorted(candidate_ids - retained_ids)

    def remove_item_ids(self, item_ids: List[str]) -> int:
        """Remove all entries whose item IDs match the provided list."""
        if not item_ids or len(self.items) == 0:
            return 0

        target_ids = {self._normalize_item_id(item_id) for item_id in item_ids if self._normalize_item_id(item_id)}
        keep_indices = [index for index, item in enumerate(self.items) if str(item.get("id")) not in target_ids]
        removed_ids = {str(item.get("id")) for item in self.items if str(item.get("id")) in target_ids}
        removed = len(removed_ids)

        if removed == 0:
            return 0

        self.items = [self.items[index] for index in keep_indices]
        if self.vectors is not None and len(self.vectors) > 0:
            if keep_indices:
                self.vectors = self.vectors[keep_indices]
            else:
                self.vectors = np.array([])

        if len(self.items) == 0:
            self.dimensions = 0

        self.remove_item_sync_states(list(removed_ids))
        self._bm25_stale = True

        return removed

    def get_indexed_item_ids(self) -> set:
        return {str(item.get("id")) for item in self.items}

    def get_indexed_item_count(self) -> int:
        return len(self.get_indexed_item_ids())

    def record_run(self, run_summary: Dict[str, Any]):
        history = [run_summary]
        history.extend(self.manifest.get("history", []))
        self.manifest["history"] = history[:20]

        if run_summary.get("success"):
            current_index = run_summary.get("current_index") or {}
            self.manifest["lastSuccessfulRun"] = run_summary
            self.manifest["currentIndex"] = {
                "runId": run_summary.get("run_id"),
                "updatedAt": run_summary.get("finished_at"),
                "provider": run_summary.get("provider"),
                "embeddingsModel": run_summary.get("embeddings_model"),
                "indexedCount": self.get_indexed_item_count(),
                "entryCount": len(self.items),
                "dimensions": self.dimensions,
                "scope": run_summary.get("scope"),
                "filters": run_summary.get("filters", {}),
                "lastRunScope": run_summary.get("scope"),
                "lastRunFilters": run_summary.get("filters", {}),
                "collectionCount": current_index.get("collectionCount", 0),
                "collections": current_index.get("collections", []),
                "unfiledItemCount": current_index.get("unfiledItemCount", 0),
                "missingItemCount": current_index.get("missingItemCount", 0),
                "matchedItemCount": current_index.get("matchedItemCount", 0),
            }

        self._save_manifest()

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        history = self.manifest.get("history", [])
        return history[:limit] if limit else history

    def get_average_seconds_per_item(self, limit: int = 5) -> Optional[float]:
        samples = []
        for run in self.get_history(limit=20):
            if not run.get("success"):
                continue
            indexed_items = run.get("indexed_items") or 0
            duration_seconds = run.get("duration_seconds") or 0
            if indexed_items > 0 and duration_seconds > 0:
                samples.append(duration_seconds / indexed_items)
            if len(samples) >= limit:
                break

        if not samples:
            return None

        return sum(samples) / len(samples)

    def get_index_status(self) -> Dict[str, Any]:
        return {
            "indexedCount": self.get_indexed_item_count(),
            "entryCount": len(self.items),
            "dimensions": self.dimensions,
            "currentIndex": self.manifest.get("currentIndex", {}),
            "lastSuccessfulRun": self.manifest.get("lastSuccessfulRun"),
        }
        
    def clear(self):
        """Clear all items"""
        self.items = []
        self.vectors = np.array([])
        self.dimensions = 0
        self.bm25 = None
        self.bm25_corpus_tokens = []
        self._bm25_stale = True
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.vectors_path):
            os.remove(self.vectors_path)
        self.manifest["currentIndex"] = {}
        self.manifest["itemSyncState"] = {}
        self.manifest["scopeState"] = {}
        self._save_manifest()
            
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    @staticmethod
    def _result_key(item: Dict[str, Any]) -> str:
        metadata = item.get("metadata", {}) or {}
        chunk_index = metadata.get("chunk_index")
        section_heading = str(metadata.get("section_heading") or "").strip()
        item_id = str(item.get("id", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        digest_source = f"{item_id}\n{chunk_index}\n{section_heading}\n{text[:400]}"
        digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]
        return f"{item_id}:{chunk_index}:{digest}"

    def _ensure_bm25(self):
        if not self._bm25_stale and self.bm25 is not None:
            return
        self._bm25_stale = False
        if not self.items:
            self.bm25 = None
            self.bm25_corpus_tokens = []
            return
        self.bm25_corpus_tokens = [
            self._tokenize(item.get("text", "")) for item in self.items
        ]
        self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

    def bm25_search(self, query_text: str, top_k: int = 5,
                    filter_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        self._ensure_bm25()
        if self.bm25 is None or len(self.items) == 0:
            return []
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        scored = []
        score_count = len(scores)
        for idx in range(min(len(self.items), score_count)):
            if filter_ids is not None and str(self.items[idx]["id"]) not in filter_ids:
                continue
            scored.append((idx, float(scores[idx])))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]
        return [{**self.items[idx], "score": score} for idx, score in scored if score > 0.0]

    def hybrid_search(self, query_embedding: List[float], query_text: str,
                      top_k: int = 5, min_score: float = 0.25,
                      filter_ids: Optional[set] = None, rrf_k: int = 60,
                      dense_weight: float = 1.2, sparse_weight: float = 0.8) -> List[Dict[str, Any]]:
        if len(self.items) == 0:
            return []
        candidate_count = max(top_k * 6, 30)
        dense_results = self.search(
            query_embedding, top_k=candidate_count,
            min_score=min_score, filter_ids=filter_ids,
        )
        sparse_results = self.bm25_search(
            query_text, top_k=candidate_count, filter_ids=filter_ids,
        )
        rrf_scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}
        for rank, result in enumerate(dense_results):
            key = self._result_key(result)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight / (rrf_k + rank + 1)
            if key not in item_map:
                item_map[key] = result
        for rank, result in enumerate(sparse_results):
            key = self._result_key(result)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_weight / (rrf_k + rank + 1)
            if key not in item_map:
                item_map[key] = result
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for key, rrf_score in ranked[:top_k]:
            result = dict(item_map[key])
            result["score"] = round(rrf_score, 6)
            results.append(result)
        return results

    def cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all vectors"""
        if len(self.vectors) == 0:
            return np.array([])
        
        # Normalize vectors
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(self.vectors))
            
        vectors_norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        vectors_norm[vectors_norm == 0] = 1  # Avoid division by zero
        
        normalized_vectors = self.vectors / vectors_norm
        normalized_query = query_vector / query_norm
        
        similarities = np.dot(normalized_vectors, normalized_query)
        return similarities
        
    def search(self, query_embedding: List[float], top_k: int = 5,
               min_score: float = 0.3, filter_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        """Search for similar items, optionally filtering by item IDs"""
        if len(self.items) == 0:
            return []

        query_vector = np.array(query_embedding)
        similarities = self.cosine_similarity(query_vector)

        # Build list of (index, score) pairs, optionally filtering
        scored = []
        sim_count = len(similarities)
        for idx in range(min(len(self.items), sim_count)):
            if filter_ids is not None and str(self.items[idx]["id"]) not in filter_ids:
                continue
            scored.append((idx, similarities[idx]))

        # Sort by score descending and take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        results = []
        for idx, score in scored:
            if score < min_score:
                continue
            item = self.items[idx]
            results.append({
                **item,
                "score": float(score)
            })

        return results
        
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            "count": self.get_indexed_item_count(),
            "entryCount": len(self.items),
            "dimensions": self.dimensions,
            "storageDir": self.storage_dir
        }
