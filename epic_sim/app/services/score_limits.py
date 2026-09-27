"""Rate limit on private-split rewards from /score (Stage 2 item deferred to Stage 5, closed in Stage 8).

The private split's labels live only in the operator's overlay; `/score` returns the reward alone for those items
(Stage 2.8), but an unlimited stream of queries could still reconstruct a label by probing. Private-split scores
are therefore counted per fixed window in Redis (shared by every worker) and refused with 429 beyond
`EPIC_SIM_PRIVATE_SCORE_LIMIT` per `EPIC_SIM_PRIVATE_SCORE_WINDOW_S` (0 disables the limit). Without Redis the
counter is per process. Public, heldout and train scoring is never limited (RL throughput).
"""

from __future__ import annotations

import logging
import threading
import time

from epic_sim.app.config import settings

log = logging.getLogger(__name__)


class PrivateScoreLimiter:
    def __init__(self, limit: int, window_s: int, redis_url: str | None):
        self.limit, self.window_s = int(limit), max(1, int(window_s))
        self._lock = threading.Lock()
        self._local: dict[int, int] = {}
        self._redis = None
        if redis_url and self.limit > 0:
            try:
                import redis
                r = redis.Redis.from_url(redis_url, socket_timeout=2)
                r.ping()
                self._redis = r
            except Exception as exc:  # noqa: BLE001 — fall back to a per-process counter
                log.warning("private score limiter: Redis unavailable (%s); counting per process", exc)

    def consume(self, n: int = 1) -> bool:
        """Count n private scores in the current window; False when that exceeds the limit."""
        if self.limit <= 0:
            return True
        window = int(time.time() // self.window_s)
        if self._redis is not None:
            try:
                key = f"score:private:{window}"
                used = int(self._redis.incrby(key, n))
                self._redis.expire(key, self.window_s * 2)
                return used <= self.limit
            except Exception as exc:  # noqa: BLE001
                log.warning("private score limiter: Redis error (%s); counting per process", exc)
        with self._lock:
            for w in [w for w in self._local if w < window]:
                del self._local[w]
            self._local[window] = self._local.get(window, 0) + n
            return self._local[window] <= self.limit


_LIMITER: PrivateScoreLimiter | None = None


def limiter() -> PrivateScoreLimiter:
    global _LIMITER
    if _LIMITER is None:
        _LIMITER = PrivateScoreLimiter(settings.private_score_limit, settings.private_score_window_s, settings.redis_url)
    return _LIMITER
