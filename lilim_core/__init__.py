"""
Lilim Core — Intelligence Layer for Lilith Linux

Modules:
- memory_manager: Rowboat-inspired persistent knowledge graph (Markdown vault)
- prompt_enhancer: Promptomatix-inspired automatic prompt optimization
- model_router: Plano + LiteLLM intelligent model routing
"""

from lilim_core.memory_manager import MemoryManager
from lilim_core.prompt_enhancer import PromptEnhancer
from lilim_core.model_router import ModelRouter

__all__ = ["MemoryManager", "PromptEnhancer", "ModelRouter"]
__version__ = "0.2.0"
