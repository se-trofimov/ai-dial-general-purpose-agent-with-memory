import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """Search user facts across conversations by semantic similarity."""

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return 'search_memory'

    @property
    def description(self) -> str:
        return (
            'Search long-term user memories by meaning. Use before personalized answers when user '
            'context is missing, even if remembering is not explicitly requested. For weather or '
            'clothing advice, search for where the user lives before asking their location. Also '
            'search before storing facts to avoid duplicates. Returns ranked candidates, not '
            'guaranteed matches; ignore irrelevant or outdated facts. Empty results mean no stored '
            'context is available, not permission to invent facts.'
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string', 'minLength': 1,
                    'description': 'A question or keywords to find relevant memories.',
                },
                'top_k': {
                    'type': 'integer', 'minimum': 1, 'maximum': 20, 'default': 5,
                    'description': 'Number of most relevant memories to return.',
                },
            },
            'required': ['query'],
            'additionalProperties': False,
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        results = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=arguments['query'],
            top_k=arguments.get('top_k', 5),
        )
        result = '\n\n'.join(
            f'- {memory.content}\n  - Category: {memory.category}'
            + (f'\n  - Topics: {", ".join(memory.topics)}' if memory.topics else '')
            for memory in results
        ) if results else 'No memories found.'
        tool_call_params.stage.append_content(result)
        return result
