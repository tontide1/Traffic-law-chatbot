import asyncio
import json

from backend.core.indexing_provider_store import IndexingProviderStore


def test_store_returns_legacy_when_metadata_file_is_missing(tmp_path):
    store = IndexingProviderStore(str(tmp_path))

    assert asyncio.run(store.get_provider("law.pdf")) == "legacy"


def test_store_persists_and_reads_provider_by_file_path(tmp_path):
    store = IndexingProviderStore(str(tmp_path))

    asyncio.run(store.set_provider("law.pdf", "google_studio"))

    assert asyncio.run(store.get_provider("law.pdf")) == "google_studio"
    payload = json.loads((tmp_path / "indexed_providers.json").read_text(encoding="utf-8"))
    assert payload == {"law.pdf": "google_studio"}


def test_store_preserves_existing_entries_when_adding_new_one(tmp_path):
    store = IndexingProviderStore(str(tmp_path))

    asyncio.run(store.set_provider("a.pdf", "deepseek"))
    asyncio.run(store.set_provider("b.pdf", "google_studio"))

    assert asyncio.run(store.get_provider("a.pdf")) == "deepseek"
    assert asyncio.run(store.get_provider("b.pdf")) == "google_studio"


def test_store_get_all_providers_returns_full_payload(tmp_path):
    store = IndexingProviderStore(str(tmp_path))

    asyncio.run(store.set_provider("a.pdf", "deepseek"))
    asyncio.run(store.set_provider("b.pdf", "google_studio"))

    assert asyncio.run(store.get_all_providers()) == {
        "a.pdf": "deepseek",
        "b.pdf": "google_studio",
    }

