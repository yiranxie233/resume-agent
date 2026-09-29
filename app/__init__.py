"""Resume Agent application package.

The first implementation keeps provider integrations behind small, dependency-free
ports.  Optional LangChain/LangGraph adapters can be layered on top without making
the core modules import those packages at startup.
"""

__all__ = ["core"]
