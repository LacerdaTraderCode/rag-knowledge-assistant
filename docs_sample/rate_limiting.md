# Rate Limiting

A rate limiter caps how many requests a client may send in a given period. It protects a service from overload and stops one noisy client from starving everyone else.

The token bucket is a popular algorithm: a bucket refills with tokens at a steady rate, every request spends one, and a request that finds the bucket empty is rejected. Because the bucket can hold a reserve, short bursts are allowed while the long-term average stays capped. A fixed-window counter is simpler but lets a client send a double burst across the boundary between two windows.

A rejected request should receive HTTP 429 Too Many Requests, ideally with a Retry-After header telling the client how many seconds to wait. Well-behaved clients respond by backing off exponentially and adding random jitter, so that thousands of them do not all retry at the same instant and recreate the spike.
