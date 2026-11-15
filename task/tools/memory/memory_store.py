import os
os.environ['OMP_NUM_THREADS'] = '1'

import asyncio
import json
from datetime import datetime, UTC, timedelta
from uuid import uuid4

import numpy as np
import faiss
from aidial_client import AsyncDial, ResourceNotFoundError
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """One JSON file per user in DIAL appdata, with a path-keyed process cache.

    Mutations are serialized per user within this process. Run a single worker;
    multiple workers would require shared caching and storage concurrency control.
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache: dict[str, MemoryCollection] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        faiss.omp_set_num_threads(1)

    def _client(self, api_key: str) -> AsyncDial:
        return AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview',
        )

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        home = await dial_client.my_appdata_home()
        if home is None:
            raise RuntimeError('DIAL appdata is unavailable for this request.')
        return f"files/{home.as_posix().strip('/')}/__long-memories/data.json"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        client = self._client(api_key)
        path = await self._get_memory_file_path(client)
        if path in self.cache:
            return self.cache[path].model_copy(deep=True)

        try:
            response = await client.files.download(path)
        except ResourceNotFoundError:
            collection = MemoryCollection()
        else:
            content = (await response.aget_content()).decode('utf-8')
            collection = MemoryCollection.model_validate(json.loads(content))

        self.cache[path] = collection.model_copy(deep=True)
        return collection

    async def _save_memories(self, api_key: str, memories: MemoryCollection) -> None:
        client = self._client(api_key)
        path = await self._get_memory_file_path(client)
        memories.updated_at = datetime.now(UTC)
        content = memories.model_dump_json().encode('utf-8')
        await client.files.upload(path, ('data.json', content, 'application/json'))
        # Publish only successfully persisted data; failed uploads must not alter cache.
        self.cache[path] = memories.model_copy(deep=True)

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        data = MemoryData(
            id=uuid4().int,
            content=content,
            importance=importance,
            category=category,
            topics=topics,
        )
        if not data.content.strip():
            raise ValueError('Memory content must not be blank.')
        path = await self._get_memory_file_path(self._client(api_key))
        async with self._locks.setdefault(path, asyncio.Lock()):
            memories = await self._load_memories(api_key)
            embedding = await asyncio.to_thread(self.model.encode, [data.content])
            memories.memories.append(Memory(data=data, embedding=embedding[0].tolist()))
            await self._save_memories(api_key, memories)
        return 'Memory successfully stored.'

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Search query must be a non-empty string.')
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError('top_k must be an integer between 1 and 20.')
        path = await self._get_memory_file_path(self._client(api_key))
        async with self._locks.setdefault(path, asyncio.Lock()):
            collection = await self._load_memories(api_key)
            if not collection.memories:
                return []
            if self._needs_deduplication(collection):
                collection = await self._deduplicate_and_save(api_key, collection)

            vectors = np.array([memory.embedding for memory in collection.memories], dtype=np.float32)
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            encoded = await asyncio.to_thread(self.model.encode, [query])
            query_vector = np.array(encoded, dtype=np.float32)
            faiss.normalize_L2(query_vector)
            _, indices = index.search(query_vector, min(top_k, len(collection.memories)))
            return [collection.memories[int(i)].data for i in indices[0]]

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        return len(collection.memories) > 10 and (
            collection.last_deduplicated_at is None
            or datetime.now(UTC) - collection.last_deduplicated_at >= timedelta(hours=self.DEDUP_INTERVAL_HOURS)
        )

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        collection.memories = self._deduplicate_fast(collection.memories)
        collection.last_deduplicated_at = datetime.now(UTC)
        await self._save_memories(api_key, collection)
        return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """Keep higher-importance records for cosine similarity strictly above 0.75.

        Exact FAISS range search avoids missing duplicates beyond a top-k cutoff.
        Its worst-case cost is quadratic, suitable for this task's ~1000 records.
        """
        if len(memories) < 2:
            return list(memories)
        vectors = np.array([memory.embedding for memory in memories], dtype=np.float32)
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        limits, _, neighbors = index.range_search(vectors, 0.75)
        removed: set[int] = set()
        for i in sorted(range(len(memories)), key=lambda i: memories[i].data.importance, reverse=True):
            if i in removed:
                continue
            removed.update(int(j) for j in neighbors[limits[i]:limits[i + 1]] if j != i)
        return [memory for i, memory in enumerate(memories) if i not in removed]

    async def delete_all_memories(self, api_key: str) -> str:
        client = self._client(api_key)
        path = await self._get_memory_file_path(client)
        async with self._locks.setdefault(path, asyncio.Lock()):
            try:
                await client.files.delete(path)
            except ResourceNotFoundError:
                self.cache.pop(path, None)
                return 'No stored memories found; memory cache cleared.'
            self.cache.pop(path, None)
        return 'All long-term memories successfully deleted, including the memory cache.'
