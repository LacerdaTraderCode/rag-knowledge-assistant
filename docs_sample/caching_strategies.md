# Caching Strategies

Caching trades staleness for speed: instead of recomputing or refetching a value, the system reuses a stored copy for a while. The two questions every caching strategy has to answer are what invalidates the cache and how long an entry is allowed to live before it is considered stale.

Cache-aside is the most common pattern: the application checks the cache first, and on a miss it fetches the real data, stores it in the cache, and returns it. This keeps the cache and the source of truth loosely coupled, at the cost of every cache miss paying the full latency of the original fetch.

Write-through caching updates the cache at the same time as the underlying data store, so reads are never stale, but every write pays the cost of touching both. Time-based expiration (a TTL) is the simplest invalidation strategy and works well when some staleness is acceptable; event-based invalidation, where a write explicitly clears the affected cache entries, is more precise but requires the write path to know what it might be invalidating.
