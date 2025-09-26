# app/services/http_client.py

import time
import random
import logging
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class RotatingHostHTTP:
    def __init__(self, hosts: List[str], timeout=(5, 10), max_retries=5, backoff_factor=0.5):
        self.hosts = hosts[:]  # list các base URL
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,      # exponential backoff
            status_forcelist=(429, 418, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"])
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": "CryptoPriceBot/1.0 (+contact admin)",
            "Accept": "application/json"
        })
        self.timeout = timeout  # (connect, read)
        # circuit breaker theo host
        self._host_down_until: Dict[str, float] = {}

    def _next_hosts(self) -> List[str]:
        # ưu tiên host không bị đánh dấu down; randomize nhẹ để tránh "thundering herd"
        now = time.time()
        candidates = [h for h in self.hosts if self._host_down_until.get(h, 0) < now]
        if not candidates:
            # nếu tất cả bị mark down, bỏ đánh dấu
            self._host_down_until.clear()
            candidates = self.hosts[:]
        random.shuffle(candidates)
        return candidates

    def mark_down(self, host: str, seconds: int = 120):
        self._host_down_until[host] = time.time() + seconds

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        last_err = None
        for base in self._next_hosts():
            url = base.rstrip("/") + "/" + path.lstrip("/")
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                # Nếu nhận 403/1020 (Cloudflare) → mark down host nhanh
                if r.status_code in (403, 409):
                    self.mark_down(base, seconds=300)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last_err = e
                # nếu lỗi mạng/timeout → mark down ngắn
                self.mark_down(base, seconds=90)
                logger.warning(f"[HTTP] {url} failed: {e}. Rotating host...")
                continue
        logger.error(f"[HTTP] All hosts failed for {path}: {last_err}")
        return None
