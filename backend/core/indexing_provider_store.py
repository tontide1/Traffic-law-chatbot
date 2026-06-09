import asyncio
import json
import os
from pathlib import Path


class IndexingProviderStore:
    # Keyed by (event_loop, resolved_file_path) to prevent loop binding and directory bottlenecks
    _locks: dict[tuple[asyncio.AbstractEventLoop, Path], asyncio.Lock] = {}

    def __init__(self, working_dir: str):
        self._path = Path(working_dir).resolve() / "indexed_providers.json"

    @property
    def _lock(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.Lock()

        key = (loop, self._path)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def _read(self) -> dict[str, str]:
        exists = await asyncio.to_thread(self._path.exists)
        if not exists:
            return {}
        try:
            content = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
            return json.loads(content)
        except Exception:
            return {}

    async def get_provider(self, file_path: str) -> str:
        async with self._lock:
            payload = await self._read()
            return payload.get(file_path, "legacy")

    async def get_all_providers(self) -> dict[str, str]:
        async with self._lock:
            return await self._read()

    async def set_provider(self, file_path: str, provider: str) -> None:
        async with self._lock:
            payload = await self._read()
            payload[file_path] = provider

            def _write_atomic():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self._path.with_suffix(f".tmp.{os.getpid()}")
                tmp_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(tmp_path, self._path)

            await asyncio.to_thread(_write_atomic)
