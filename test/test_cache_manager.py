from src.core.cache_manager import CacheManager


def test_cache_set_and_get(cache_manager: CacheManager) -> None:
    key = "test_key"
    data = {"param": "value"}
    value = {"result": 42}

    cache_manager.set(key, data, value)
    retrieved = cache_manager.get(key, data)
    assert retrieved == value, "Cached value should match stored value"


def test_cache_ttl_expired(cache_manager: CacheManager) -> None:
    key = "test_key"
    data = {"param": "value"}
    value = {"result": 42}

    cache_manager.set(key, data, value)
    retrieved = cache_manager.get(key, data, ttl_seconds=-1)
    assert retrieved is None, "Expired cache entry should return None"


def test_cache_invalidate(cache_manager: CacheManager) -> None:
    key = "test_key"
    data = {"param": "value"}
    value = {"result": 42}

    cache_manager.set(key, data, value)
    cache_manager.invalidate(key, data)
    retrieved = cache_manager.get(key, data)
    assert retrieved is None, "Invalidated cache entry should return None"


def test_cache_cleanup(cache_manager: CacheManager) -> None:
    small_cache = CacheManager(cache_manager.cache_dir, max_size_bytes=100)
    for i in range(10):
        small_cache.set(f"key_{i}", {}, {"data": "x" * 20})
    stats = small_cache.get_cache_stats()
    assert stats["total_size_bytes"] <= small_cache.max_size_bytes * 0.8, "Cache should be cleaned up"


def test_cache_stats(cache_manager: CacheManager) -> None:
    stats = cache_manager.get_cache_stats()
    assert stats["total_entries"] == 0, "Initial cache should have no entries"
    cache_manager.set("key1", {"data": "test"}, {"result": 1})
    stats = cache_manager.get_cache_stats()
    assert stats["total_entries"] == 1, "Cache should have one entry"
    assert stats["total_size_bytes"] > 0, "Cache size should be non-zero"
