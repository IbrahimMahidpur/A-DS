import logging
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class AgentMemory:
    def __init__(self, collection_name: str = "agent_memory", ttl_seconds: int = 86400):
        self.collection_name = collection_name
        self.ttl_seconds = ttl_seconds
        self._client = None
        self._collection = None
        self._init_chroma()

    def _init_chroma(self):
        """Initialize ChromaDB client.
        In production (Docker) we prefer a persistent client stored on disk.
        The environment variable ``CHROMA_PERSIST_DIR`` (exposed as ``CHROMA_DIR``
        in ``multimodal_ds.config``) determines the persistence directory.
        If the directory exists and is writable we use ``PersistentClient``;
        otherwise we fall back to an in‑memory ``EphemeralClient`` and log a warning.
        """
        try:
            import os
            import chromadb
            from chromadb.config import Settings
            from multimodal_ds.config import CHROMA_DIR

            # Determine whether to use persistent storage
            use_persistent = False
            try:
                # Path must exist and be a writable directory
                if CHROMA_DIR.exists() and os.access(str(CHROMA_DIR), os.W_OK):
                    use_persistent = True
            except Exception as e:
                logger.warning(f"[Memory] Unable to check CHROMA_DIR writability: {e}")

            if use_persistent:
                # Persistent client stores data on disk for production use
                self._client = chromadb.PersistentClient(
                    path=str(CHROMA_DIR),
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info(f"[Memory] ChromaDB initialized with PersistentClient at {CHROMA_DIR}")
            else:
                # Fallback to in‑memory client for dev / test environments
                self._client = chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.warning("[Memory] Using EphemeralClient for ChromaDB (no persistent storage)")

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.warning(f"[Memory] ChromaDB init failed: {e}")
            self._collection = None

    def store(self, content: str, metadata: dict = None, doc_id: str = None) -> str:
        import uuid
        entry_id = doc_id or str(uuid.uuid4())
        # Include timestamp for TTL handling
        meta = {"timestamp": datetime.now(timezone.utc).isoformat(), **(metadata or {})}
        meta = {k: str(v) for k, v in meta.items()}
        if self._collection:
            try:
                embedding = self._get_embedding(content)
                self._collection.upsert(
                    ids=[entry_id], documents=[content],
                    embeddings=[embedding] if embedding else None, metadatas=[meta]
                )
            except Exception as e:
                logger.warning(f"[Memory] Store failed: {e}")
        # After inserting, optionally purge old entries
        self._purge_expired()
        return entry_id

    def retrieve(self, query: str, n_results: int = 5, where: dict = None) -> list:
        if not self._collection:
            return []
        try:
            embedding = self._get_embedding(query)
            count = self._collection.count()
            if count == 0:
                return []
            kwargs = {"n_results": min(n_results, count)}
            if embedding:
                kwargs["query_embeddings"] = [embedding]
            else:
                kwargs["query_texts"] = [query]
            if where:
                kwargs["where"] = {"$and": [{k: v} for k, v in where.items()]} if len(where) > 1 else where
            results = self._collection.query(**kwargs)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            # Filter out expired entries based on timestamp TTL
            filtered = []
            cutoff = datetime.now(timezone.utc).timestamp() - self.ttl_seconds
            for d, m in zip(docs, metas):
                ts_str = m.get("timestamp")
                try:
                    ts = datetime.fromisoformat(ts_str).timestamp()
                except Exception:
                    ts = 0
                if ts >= cutoff:
                    filtered.append({"content": d, "metadata": m})
            return filtered
        except Exception as e:
            logger.warning(f"[Memory] Retrieve failed: {e}")
            return []

    def store_analysis_step(self, step_name: str, result: str, session_id: str = "default"):
        return self.store(
            content=f"[Step: {step_name}]\n{result}",
            metadata={"step": step_name, "session_id": session_id, "type": "analysis_step"}
        )

    def get_session_history(self, session_id: str) -> list:
        return self.retrieve(query="analysis step result", n_results=20, where={"session_id": session_id})

    def _get_embedding(self, text: str) -> Optional[list]:
        try:
            import httpx
            from multimodal_ds.config import OLLAMA_BASE_URL, EMBED_MODEL
            model_name = EMBED_MODEL.replace("ollama/", "")
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": model_name, "prompt": text[:2000]}, timeout=30,
            )
            if response.status_code == 200:
                return response.json().get("embedding")
        except Exception:
            pass
        return None

    def count(self) -> int:
        """Return the number of stored memory entries."""
        if not self._collection:
            return 0
        try:
            return self._collection.count()
        except Exception as e:
            logger.warning(f"[Memory] Count failed: {e}")
            return 0


    def get_relevant_lessons(self, query: str, task_type: str = None) -> list:
        """Retrieve up to 10 deduplicated lesson strings for a query.
        If task_type is provided, filters by that task; otherwise returns any reflection.
        """
        try:
            where = {"type": "reflection"}
            if task_type is not None:
                where = {"type": "reflection", "task": task_type}
            results = self.retrieve(query, n_results=10, where=where)
            lessons = []
            seen = set()
            for r in results:
                content = r.get("content", "")
                if "REFLECTION" not in content:
                    continue
                # Extract JSON after the marker
                if "REFLECTION [" in content:
                    try:
                        json_str = content.split("REFLECTION [")[1].split("]: ", 1)[1]
                    except Exception:
                        json_str = content
                else:
                    json_str = content
                try:
                    parsed = json.loads(json_str)
                except Exception:
                    continue
                for lesson in parsed.get("lessons", []):
                    if lesson not in seen:
                        seen.add(lesson)
                        lessons.append(lesson)
                        if len(lessons) >= 10:
                            break
                if len(lessons) >= 10:
                    break
            return lessons
        except Exception as e:
            logger.warning(f"[Memory] get_relevant_lessons failed: {e}")
            return []

    def _purge_expired(self):
        """Delete entries older than TTL from the Chroma collection."""
        if not self._collection:
            return
        try:
            # Retrieve all ids and metadatas
            all_entries = self._collection.get(include=["metadatas", "ids"])  # returns dict with keys 'ids' and 'metadatas'
            ids = all_entries.get("ids", [])
            metas = all_entries.get("metadatas", [])
            if not ids:
                return
            cutoff = datetime.now(timezone.utc).timestamp() - self.ttl_seconds
            to_delete = []
            for entry_id, meta in zip(ids, metas):
                ts_str = meta.get("timestamp")
                try:
                    ts = datetime.fromisoformat(ts_str).timestamp()
                except Exception:
                    continue
                if ts < cutoff:
                    to_delete.append(entry_id)
            if to_delete:
                self._collection.delete(ids=to_delete)
                logger.info(f"[Memory] Purged {len(to_delete)} expired entries")
        except Exception as e:
            logger.warning(f"[Memory] Purge expired failed: {e}")

