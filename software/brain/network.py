"""How a cloud brain reaches its AI service (SEC-5, SEC-5b). One place, so every brain makes the same promise:
no proxy unless the player opted in, certificates checked against our own roots, redirects refused."""
from __future__ import annotations

import ssl
import urllib.request
from urllib.parse import urlsplit

import certifi

from brain import CLIENT_ARGS


def trust_context() -> ssl.SSLContext:
    """Our own roots (certifi), never SSL_CERT_FILE or SSL_CERT_DIR, and never the Windows store: a network that inspects
    encrypted traffic with its own certificate is refused [RATCHET-NO-ENV-CA]."""
    return ssl.create_default_context(cafile=certifi.where())


def _network_proxy() -> str | None:
    """The proxy this PC's environment names for https, if it is a plain http(s) one."""
    proxies = urllib.request.getproxies()
    url = proxies.get("https") or proxies.get("http") or proxies.get("all")
    if not url:
        return None
    return url if urlsplit(url).scheme in ("http", "https") else None


class _ProxyUrl(str):
    """The proxy address, which may carry a user name and password from the environment. It works as the real address
    everywhere a connection needs it, but its repr (what a log line or an error shows) hides the credentials."""

    def __repr__(self) -> str:
        parts = urlsplit(str(self))
        host = parts.hostname or ""
        return f"<proxy {parts.scheme}://{'***@' if (parts.username or parts.password) else ''}{host}{':' + str(parts.port) if parts.port else ''}>"


def client_args(use_proxy: bool = False) -> dict:
    """The httpx arguments for a cloud brain. `use_proxy` is the player's PIN-guarded opt-in (cloud_proxy); off by default.
    The AI host is pinned by the brain's own base URL, so the proxy carries that host and nothing else."""
    args = dict(CLIENT_ARGS)
    if use_proxy:
        proxy = _network_proxy()
        if proxy:
            args["proxy"] = _ProxyUrl(proxy)
    args["verify"] = trust_context()  # our own roots in every case [RATCHET-NO-ENV-CA]
    return args
