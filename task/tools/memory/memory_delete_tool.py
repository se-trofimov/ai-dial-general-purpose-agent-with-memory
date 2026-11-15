from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class DeleteMemoryTool(BaseTool):
    """Delete all persisted user memories and their cache."""

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return 'delete_all_memories'

    @property
    def description(self) -> str:
        return (
            'Permanently delete ALL long-term memories for the current user and clear their cache. '
            'Use only when the user explicitly requests forgetting all saved information. This '
            'cannot delete a single selected fact; for a partial deletion request, explain this '
            'limitation and ask permission before deleting everything. Never call alongside '
            'store_memory. Do not claim deletion succeeded if this tool reports an error.'
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {'type': 'object', 'properties': {}, 'additionalProperties': False}

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        result = await self.memory_store.delete_all_memories(tool_call_params.api_key)
        tool_call_params.stage.append_content(result)
        return result
