import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class StoreMemoryTool(BaseTool):
    """Store a durable fact about the user."""

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return 'store_memory'

    @property
    def description(self) -> str:
        return (
            'Store an important, explicit fact about the user for future conversations: preferences, '
            'location, goals, plans or personal context. Use one concise fact per call. Search first '
            'to avoid storing a fact already remembered. Never store guesses, passwords, API keys, '
            'temporary task details, or facts the user asks not to remember. Report success only '
            'after this tool succeeds.'
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            'type': 'object',
            'properties': {
                'content': {
                    'type': 'string', 'minLength': 1,
                    'description': 'The memory content to store. A clear, concise fact about the user.',
                },
                'category': {
                    'type': 'string', 'default': 'general',
                    'description': 'Category: preferences, personal_info, goals, plans or context.',
                },
                'importance': {
                    'type': 'number', 'minimum': 0, 'maximum': 1, 'default': 0.5,
                    'description': 'Importance score between 0 and 1. Higher means more important to remember.',
                },
                'topics': {
                    'type': 'array', 'items': {'type': 'string'}, 'default': [],
                    'description': 'Related topics or tags for the memory.',
                },
            },
            'required': ['content', 'category'],
            'additionalProperties': False,
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        result = await self.memory_store.add_memory(
            api_key=tool_call_params.api_key,
            content=arguments['content'],
            category=arguments['category'],
            importance=arguments.get('importance', 0.5),
            topics=arguments.get('topics', []),
        )
        tool_call_params.stage.append_content(result)
        return result
