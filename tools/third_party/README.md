# Lopen Third-Party Plugins

Place your custom plugin `.py` files here. Lopen will automatically discover
and load them on startup (or when `POST /plugins/reload` is called).

## Quick Start

See `PLUGINS.md` in the repository root for full documentation and examples.

## Requirements

Each plugin file must contain at least one class that:
- Inherits from `tools.base_tool.BaseTool`
- Defines `name`, `description` class attributes
- Implements the `run(self, query: str, **kwargs) -> str` method

## Example

```python
# tools/third_party/my_plugin.py
from tools.base_tool import BaseTool

class MyPlugin(BaseTool):
    name = "my_plugin"
    description = "Does something amazing for specific queries"
    tags = ["custom", "example"]
    version = "1.0.0"
    requires_permission = False

    def run(self, query: str, **kwargs) -> str:
        return f"MyPlugin handled: {query}"
```
