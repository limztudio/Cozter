"""Lazy real HTTP types, also usable by runtime annotation introspection.

Importing this facade does not import aiohttp. Attribute access uses ordinary
Python imports, so concurrent first use shares the library and resolver class.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp as aiohttp
    from aiohttp import ClientResponse as ClientResponse
    from aiohttp import ClientSession as ClientSession
    from aiohttp import ClientTimeout as ClientTimeout
    from aiohttp import TCPConnector as TCPConnector
    from ._public_resolver import PublicResolver as PublicResolver


def __getattr__(name: str) -> Any:
    if name == "PublicResolver":
        from ._public_resolver import PublicResolver
        value: Any = PublicResolver
    elif name in {"aiohttp", "ClientResponse", "ClientSession", "ClientTimeout", "TCPConnector"}:
        import aiohttp
        value = aiohttp if name == "aiohttp" else getattr(aiohttp, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
