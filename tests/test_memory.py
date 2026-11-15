import asyncio
import json
import unittest
from datetime import datetime, UTC, timedelta
from pathlib import PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from aidial_client import ResourceNotFoundError

from task.tools.memory._models import Memory, MemoryCollection, MemoryData
from task.tools.memory.memory_delete_tool import DeleteMemoryTool
from task.tools.memory.memory_search_tool import SearchMemoryTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.memory.memory_store_tool import StoreMemoryTool
from task.tools.models import ToolCallParams


def memory(content: str, embedding: list[float], importance: float = 0.5) -> Memory:
    return Memory(data=MemoryData(id=1, content=content, importance=importance), embedding=embedding)


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.encoder = Mock()
        self.encoder.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        with patch('task.tools.memory.memory_store.SentenceTransformer', return_value=self.encoder):
            self.store = LongTermMemoryStore('http://localhost:8080')
        self.client = Mock()
        self.client.my_appdata_home = AsyncMock(return_value=PurePosixPath('bucket/appdata/agent'))
        self.client.files.download = AsyncMock(side_effect=ResourceNotFoundError('missing'))
        self.client.files.upload = AsyncMock()
        self.client.files.delete = AsyncMock()
        self.store._client = Mock(return_value=self.client)
        self.path = 'files/bucket/appdata/agent/__long-memories/data.json'

    async def test_concurrent_adds_are_persisted_and_reload_across_conversations(self):
        await asyncio.gather(*(
            self.store.add_memory('user', f'Fact {i}', 0.5, 'general', [])
            for i in range(10)
        ))
        upload = self.client.files.upload.call_args.args
        self.assertEqual(upload[0], self.path)
        persisted = MemoryCollection.model_validate_json(upload[1][1])
        self.assertEqual(len(persisted.memories), 10)
        self.assertEqual(len({m.data.id for m in persisted.memories}), 10)
        self.store.cache.clear()
        self.client.files.download.side_effect = None
        self.client.files.download.return_value = SimpleNamespace(
            aget_content=AsyncMock(return_value=upload[1][1]),
        )
        result = await self.store.search_memories('user', 'Facts', 20)
        self.assertEqual(len(result), 10)
        self.assertTrue(all(isinstance(m, MemoryData) for m in result))

    async def test_search_ranks_by_cosine_not_vector_magnitude(self):
        self.store.cache[self.path] = MemoryCollection(memories=[
            memory('unrelated', [0.0, 100.0]),
            memory('related', [2.0, 0.0]),
            memory('partially related', [1.0, 1.0]),
        ])
        result = await self.store.search_memories('user', 'related', 2)
        self.assertEqual([m.content for m in result], ['related', 'partially related'])

    async def test_failed_upload_does_not_publish_unsaved_memory(self):
        self.store.cache[self.path] = MemoryCollection(memories=[memory('old', [1.0, 0.0])])
        self.client.files.upload.side_effect = RuntimeError('storage unavailable')
        with self.assertRaisesRegex(RuntimeError, 'storage unavailable'):
            await self.store.add_memory('user', 'new', 0.5, 'general', [])
        self.assertEqual([m.data.content for m in self.store.cache[self.path].memories], ['old'])

    async def test_load_does_not_hide_transport_or_corrupt_file_errors(self):
        self.client.files.download.side_effect = RuntimeError('access denied')
        with self.assertRaisesRegex(RuntimeError, 'access denied'):
            await self.store._load_memories('user')
        self.client.files.download.side_effect = None
        self.client.files.download.return_value = SimpleNamespace(aget_content=AsyncMock(return_value=b'{'))
        with self.assertRaises(json.JSONDecodeError):
            await self.store._load_memories('user')
        self.assertNotIn(self.path, self.store.cache)

    async def test_delete_clears_only_current_users_cache_even_when_file_missing(self):
        self.store.cache[self.path] = MemoryCollection(memories=[memory('private', [1.0, 0.0])])
        self.store.cache['other-user'] = MemoryCollection()
        self.client.files.delete.side_effect = ResourceNotFoundError('missing')
        await self.store.delete_all_memories('user')
        self.client.files.delete.assert_awaited_once_with(self.path)
        self.assertNotIn(self.path, self.store.cache)
        self.assertIn('other-user', self.store.cache)

    async def test_failed_deletion_is_reported_and_preserves_cache(self):
        self.store.cache[self.path] = MemoryCollection()
        self.client.files.delete.side_effect = RuntimeError('access denied')
        with self.assertRaisesRegex(RuntimeError, 'access denied'):
            await self.store.delete_all_memories('user')
        self.assertIn(self.path, self.store.cache)

    async def test_missing_appdata_never_falls_back_to_shared_storage(self):
        self.client.my_appdata_home.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'appdata is unavailable'):
            await self.store.add_memory('user', 'fact', 0.5, 'general', [])
        self.client.files.upload.assert_not_awaited()

    async def test_search_triggers_due_deduplication_and_persists_timestamp(self):
        self.store.cache[self.path] = MemoryCollection(memories=[
            memory(f'Fact {i}', [1.0, 0.0], i / 20) for i in range(12)
        ])
        result = await self.store.search_memories('user', 'fact')
        self.assertEqual([m.content for m in result], ['Fact 11'])
        self.assertIsNotNone(self.store.cache[self.path].last_deduplicated_at)
        self.client.files.upload.assert_awaited_once()

    def test_deduplication_keeps_highest_importance_beyond_top_k_neighbors(self):
        entries = [memory(f'Fact {i}', [3.0, 0.0], i / 30) for i in range(25)]
        entries.append(memory('different', [0.0, 1.0]))
        result = self.store._deduplicate_fast(entries)
        self.assertEqual([m.data.content for m in result], ['Fact 24', 'different'])

    def test_deduplication_threshold_is_strict_and_ties_are_stable(self):
        entries = [
            memory('first', [1.0, 0.0]),
            memory('duplicate', [1.0, 0.0]),
            memory('boundary', [0.75, np.sqrt(1 - 0.75 ** 2)]),
        ]
        result = self.store._deduplicate_fast(entries)
        self.assertEqual([m.data.content for m in result], ['first', 'boundary'])

    def test_deduplication_schedule(self):
        collection = MemoryCollection(memories=[memory('fact', [1.0, 0.0])] * 10)
        self.assertFalse(self.store._needs_deduplication(collection))
        collection.memories.append(memory('another', [1.0, 0.0]))
        self.assertTrue(self.store._needs_deduplication(collection))
        collection.last_deduplicated_at = datetime.now(UTC)
        self.assertFalse(self.store._needs_deduplication(collection))
        collection.last_deduplicated_at -= timedelta(hours=25)
        self.assertTrue(self.store._needs_deduplication(collection))

    async def test_invalid_search_and_blank_memory_are_rejected(self):
        for top_k in (0, 21, True, 1.5):
            with self.assertRaises(ValueError):
                await self.store.search_memories('user', 'query', top_k)
        with self.assertRaises(ValueError):
            await self.store.search_memories('user', ' ')
        with self.assertRaises(ValueError):
            await self.store.add_memory('user', ' ', 0.5, 'general', [])
        self.client.files.upload.assert_not_awaited()


class MemoryToolTests(unittest.IsolatedAsyncioTestCase):
    def params(self, arguments: dict) -> ToolCallParams:
        return ToolCallParams(
            tool_call=Mock(function=Mock(arguments=json.dumps(arguments))),
            stage=Mock(), choice=Mock(), api_key='user', conversation_id='new-conversation',
        )

    async def test_store_defaults_and_stage(self):
        store = Mock(add_memory=AsyncMock(return_value='stored'))
        params = self.params({'content': 'Lives in Paris', 'category': 'personal_info'})
        result = await StoreMemoryTool(store)._execute(params)
        store.add_memory.assert_awaited_once_with(
            api_key='user', content='Lives in Paris', category='personal_info', importance=0.5, topics=[],
        )
        self.assertEqual(result, 'stored')
        params.stage.append_content.assert_called_once_with(result)

    async def test_search_empty_and_markdown_results(self):
        store = Mock(search_memories=AsyncMock(return_value=[]))
        tool = SearchMemoryTool(store)
        self.assertEqual(await tool._execute(self.params({'query': 'location'})), 'No memories found.')
        store.search_memories.return_value = [
            MemoryData(id=1, content='Lives in Paris', category='personal_info', topics=['location']),
        ]
        result = await tool._execute(self.params({'query': 'location'}))
        self.assertIn('Lives in Paris', result)
        self.assertIn('Category: personal_info', result)
        self.assertIn('Topics: location', result)

    async def test_delete_forwards_user_identity_and_stage(self):
        store = Mock(delete_all_memories=AsyncMock(return_value='deleted'))
        params = self.params({})
        self.assertEqual(await DeleteMemoryTool(store)._execute(params), 'deleted')
        store.delete_all_memories.assert_awaited_once_with('user')
        params.stage.append_content.assert_called_once_with('deleted')


if __name__ == '__main__':
    unittest.main()
