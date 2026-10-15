"""Public-address resolver, imported only when HTTP tools need it."""

import socket
from typing import Any

from aiohttp.abc import AbstractResolver
from aiohttp.resolver import DefaultResolver

from . import base


class PublicResolver(AbstractResolver):
    """Resolve DNS normally, while allowing only globally routable answers."""

    def __init__(self) -> None:
        self._resolver: AbstractResolver = DefaultResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: Any = socket.AF_UNSPEC,
    ) -> list[Any]:
        answers = await self._resolver.resolve(host, port, family=family)
        if not answers:
            raise base.NonPublicAddressError(
                "URL host did not resolve to an address",
            )
        for answer in answers:
            address = answer.get("host") if isinstance(answer, dict) else None
            if not isinstance(address, str) or not base._is_public_ip(address):
                raise base.NonPublicAddressError(
                    "URL host resolves to a non-public address",
                )
        return answers

    async def close(self) -> None:
        await self._resolver.close()
