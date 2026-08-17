#!/usr/bin/env python3
"""Dynamic resolution of the M4 worker's LAN address.

Vendored into nebulai from ``~/Mind/.hermes/scripts/m4host.py`` (the canonical
copy). The only local adaptations are the cache location (``_cache_path`` — the
original wrote next to the Mind checkout; here it writes to a per-user cache dir)
and this header. The ``resolve()`` contract, the discovery order, and every env
knob below are unchanged, so nebulai and Mind find the same box the same way.

The M4 is a company-managed device: its DHCP lease rotates its IP, and Private
Wi-Fi Address rotates its MAC, so we can pin neither a static reservation nor an
mDNS ``.local`` name (the corporate AP filters both). We therefore cannot address
the box by *where it is* — we find it by *what it serves*.

``resolve()`` returns the host (IP or name) every client should talk to. Order:

  1. explicit pin (M4_HOST / MIND_M4_HOST) when reachable — honored first;
  2. a fresh cached discovery — trusted outright for a few seconds, port-probed after;
  3. a /24 sweep for the box answering a known M4 service port, confirmed over HTTP
     (Ollama /api/tags carrying the pinned embed model, or the :8100 control plane),
     caching the winner.

On total failure it returns the pinned / last-known / literal-default host so
callers still form a URL and fail loudly at request time — never a silent guess
that a model is up. Everything here is best-effort and NEVER raises; a resolver
that throws would take down every command that imports it.

Env (all optional):
  M4_HOST / MIND_M4_HOST         preferred host; tried first, then discovery
  MIND_M4_STRICT=1               never scan; always use the pin (locked/again-static nets)
  MIND_M4_DISCOVERY=0            disable the /24 sweep entirely (cache + pin only)
  MIND_M4_SCAN_CIDR=a.b.c.0/24   override the subnet to sweep (narrow noise / fix autodetect)
  MIND_M4_DISCOVERY_PORTS=...    comma list, default "11435,11434,8100,8050"
  MIND_M4_CACHE_TTL=600          seconds a good discovery is probe-revalidated before re-sweep
  MIND_M4_TRUST_TTL=30           seconds a good discovery is trusted with no probe at all
  MIND_M4_NEG_TTL=45             seconds a failed sweep suppresses re-scanning
  MIND_M4_SCAN_TIMEOUT=0.35      per-host TCP connect timeout (seconds)
  MIND_M4_SCAN_WORKERS=128       concurrency for the sweep
  MIND_M4_ID_MODEL=mxbai-embed-large   substring an Ollama /api/tags entry must contain
  NEBULAI_M4_CACHE=/path.json    override the on-disk cache location (nebulai-local)

CLI:
  python3 -m nebulai.backend.m4host          # resolve and print the host (honors cache)
  python3 -m nebulai.backend.m4host --fresh  # force a re-scan, ignore cache
  python3 -m nebulai.backend.m4host --json   # {host, source, ms, cache}
"""
import concurrent.futures as _cf
import json
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "192.168.0.110"
DEFAULT_PORTS = (11435, 11434, 8100, 8050)

_memo = {"host": None, "ts": 0.0}   # in-process fast path across repeated resolve() calls


# --------------------------------------------------------------------------- env
def _env(key, default=None):
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def _flag(key, default=False):
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(key, default):
    try:
        return int(float(_env(key, default)))
    except (TypeError, ValueError):
        return default


def _float(key, default):
    try:
        return float(_env(key, default))
    except (TypeError, ValueError):
        return default


def _pin():
    """User-declared host, or None. Sentinels mean 'discover'."""
    for key in ("M4_HOST", "MIND_M4_HOST"):
        v = _env(key)
        if v and v.strip().lower() not in ("auto", "discover", "dynamic"):
            return v.strip()
    return None


def _ports():
    raw = _env("MIND_M4_DISCOVERY_PORTS")
    if not raw:
        return list(DEFAULT_PORTS)
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return out or list(DEFAULT_PORTS)


# ------------------------------------------------------------------------- cache
def _cache_path():
    """Where the last good discovery is cached.

    The upstream copy stored this next to the Mind checkout; vendored into
    nebulai there is no such anchor, so honor NEBULAI_M4_CACHE, else write under
    the user's XDG cache dir (falling back to a temp dir if that can't be made).
    Best-effort only: a bad path just means the sweep is not memoized to disk.
    """
    override = _env("NEBULAI_M4_CACHE")
    if override:
        return override
    base = _env("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    d = os.path.join(base, "nebulai")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = tempfile.gettempdir()
    return os.path.join(d, "m4host.json")


def _read_cache():
    try:
        with open(_cache_path(), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and d.get("host"):
            return d
    except (OSError, ValueError):
        pass
    return None


def _write_cache(host, ok):
    try:
        tmp = _cache_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"host": host, "ok": bool(ok), "ts": time.time()}, fh)
        os.replace(tmp, _cache_path())
    except OSError:
        pass


def _fresh(entry, ttl):
    try:
        return (time.time() - float(entry.get("ts", 0))) < ttl
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------ probing
def _port_open(host, port, timeout):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_json(url, timeout):
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _http_probe(url, timeout):
    """Like _http_json but returns (http_status, parsed_json_or_None). A 401/403 is a
    real HTTP status, not a connection failure, so a token-gated endpoint can still
    prove identity. Returns (None, None) only when nothing answered on the socket."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(body)
            except ValueError:
                return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, OSError):
        return None, None


def _is_m4(host, port, timeout):
    """Confirm the host answering `port` is really the M4, not some other LAN box that
    happens to run an Ollama or an OpenAI-shaped server. Identity signal per port."""
    if port in (11434, 11435):
        data = _http_json(f"http://{host}:{port}/api/tags", timeout)
        if not isinstance(data, dict):
            return False
        want = _env("MIND_M4_ID_MODEL", "mxbai-embed-large").lower()
        for m in data.get("models") or []:
            name = str(m.get("name") or m.get("model") or "").lower()
            if want in name or "embed" in name:
                return True
        return False
    if port == 8100:
        # The m4ai LAN control plane. /health is always no-auth, so it keeps proving
        # identity even after the operator sets M4AI_LAN_TOKEN (which gates /v1/*). And a
        # 401/403 from the token-gated /v1/status is ITSELF proof the control plane is
        # here — only it guards that route — so a locked box is still recognized.
        code, _ = _http_probe(f"http://{host}:{port}/health", timeout)
        if code == 200:
            return True
        code, data = _http_probe(f"http://{host}:{port}/v1/status", timeout)
        return code in (401, 403) or (code == 200 and bool(data))
    if port == 8050:
        data = _http_json(f"http://{host}:{port}/v1/models", timeout)
        return bool(isinstance(data, dict) and data.get("data"))
    return True  # unknown port: an open socket is the best signal we have


def _alive(host, ports, timeout, verify=True):
    """Quick 'is this host our worker' check for a known candidate (pin or cache)."""
    for p in ports:
        if _port_open(host, p, timeout):
            if not verify or _is_m4(host, p, max(timeout * 4, 1.5)):
                return True
    return False


# ---------------------------------------------------------------------- discovery
def _own_ipv4():
    """Source IP the OS would use to reach off-subnet — i.e. our address on the LAN the
    worker shares. UDP connect sends nothing; it just picks a route."""
    for target in ("1.1.1.1", "192.168.0.1", "10.255.255.255"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((target, 1))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        except OSError:
            continue
        finally:
            s.close()
    return None


def _cidr_hosts():
    """Host IPs to sweep. Explicit MIND_M4_SCAN_CIDR (/24..'/32'), else own /24."""
    cidr = _env("MIND_M4_SCAN_CIDR")
    if cidr and "/" in cidr:
        try:
            net, bits = cidr.split("/")
            bits = int(bits)
            octs = [int(x) for x in net.split(".")]
            if len(octs) == 4 and 16 <= bits <= 32:
                base = (octs[0] << 24) | (octs[1] << 16) | (octs[2] << 8) | octs[3]
                base &= (0xFFFFFFFF << (32 - bits)) & 0xFFFFFFFF
                count = min(1 << (32 - bits), 1024)  # cap runaway sweeps
                out = []
                for i in range(count):
                    a = base + i
                    if bits < 31 and (i == 0 or i == count - 1):
                        continue  # skip network / broadcast for real subnets
                    out.append(f"{(a >> 24) & 255}.{(a >> 16) & 255}.{(a >> 8) & 255}.{a & 255}")
                return out
        except (ValueError, IndexError):
            pass
    ip = _own_ipv4()
    if not ip:
        return []
    a, b, c, _d = ip.split(".")
    return [f"{a}.{b}.{c}.{h}" for h in range(1, 255)]


def _anchor():
    """Last octet to bias scan order toward — last-known, then our own address."""
    c = _read_cache()
    for src in ((c or {}).get("host"), _own_ipv4()):
        if src and src.count(".") == 3 and src.rsplit(".", 1)[1].isdigit():
            return int(src.rsplit(".", 1)[1])
    return 128


def _sweep(ports, timeout, workers):
    hosts = _cidr_hosts()
    if not hosts:
        return None
    own = _own_ipv4()
    anchor = _anchor()
    hosts = [h for h in hosts if h != own]
    hosts.sort(key=lambda ip: abs(int(ip.rsplit(".", 1)[1]) - anchor))
    for port in ports:
        opens = []
        with _cf.ThreadPoolExecutor(max_workers=max(8, workers)) as ex:
            futs = {ex.submit(_port_open, h, port, timeout): h for h in hosts}
            for fut in _cf.as_completed(futs):
                try:
                    if fut.result():
                        opens.append(futs[fut])
                except Exception:
                    pass
        opens.sort(key=lambda ip: abs(int(ip.rsplit(".", 1)[1]) - anchor))
        for h in opens:
            if _is_m4(h, port, max(timeout * 4, 1.5)):
                return h
    return None


# ------------------------------------------------------------------------- public
def resolve(force=False):
    """Best current host for the M4 worker. Cheap on the hot path, never raises."""
    now = time.time()
    memo_ttl = _int("MIND_M4_TRUST_TTL", 30)
    if not force and _memo["host"] and (now - _memo["ts"]) < memo_ttl:
        return _memo["host"]

    ports = _ports()
    timeout = _float("MIND_M4_SCAN_TIMEOUT", 0.35)
    pin = _pin()

    def _win(host, ok=True, cache=True):
        _memo["host"], _memo["ts"] = host, now
        if cache:
            _write_cache(host, ok)
        return host

    # 1. strict pin: authoritative, no discovery ever.
    if pin and _flag("MIND_M4_STRICT"):
        return _win(pin, cache=False)

    # 2. pin, if reachable (trust an explicit pin on a bare open port).
    if pin and not force and _alive(pin, ports, timeout, verify=False):
        return _win(pin)

    cache = _read_cache()

    # 3. very fresh good cache: trust without touching the network (burst fast path).
    if not force and cache and cache.get("ok") and _fresh(cache, _int("MIND_M4_TRUST_TTL", 30)):
        return _win(cache["host"], cache=False)

    # 4. still-good cache: probe-revalidate before trusting (guards IP reassignment).
    if not force and cache and cache.get("ok") and _fresh(cache, _int("MIND_M4_CACHE_TTL", 600)):
        if _alive(cache["host"], ports, timeout, verify=True):
            return _win(cache["host"])

    # 5. recent failed sweep: don't hammer the LAN — return last-known and move on.
    if not force and cache and not cache.get("ok") and _fresh(cache, _int("MIND_M4_NEG_TTL", 45)):
        return _memo_or(pin, cache)

    # 6. discovery sweep.
    if _flag("MIND_M4_DISCOVERY", default=True):
        found = _sweep(ports, timeout, _int("MIND_M4_SCAN_WORKERS", 128))
        if found:
            return _win(found)

    # 7. give up loudly-later: pin / last-known / literal default.
    fallback = pin or (cache or {}).get("host") or DEFAULT_HOST
    _write_cache(fallback, False)
    _memo["host"], _memo["ts"] = fallback, now
    return fallback


def _memo_or(pin, cache):
    host = pin or (cache or {}).get("host") or DEFAULT_HOST
    _memo["host"], _memo["ts"] = host, time.time()
    return host


def invalidate():
    """Forget the current answer so the next resolve() re-probes / re-scans. Call this
    after a connection failure to a resolved host."""
    _memo["host"], _memo["ts"] = None, 0.0
    try:
        os.remove(_cache_path())
    except OSError:
        pass


def base(port, path="", scheme="http"):
    """Convenience URL builder: base(8050, '/v1/models') -> 'http://<host>:8050/v1/models'."""
    url = f"{scheme}://{resolve()}:{port}"
    if path:
        url += "/" + path.lstrip("/")
    return url


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    as_json = "--json" in sys.argv
    t0 = time.time()
    if fresh:
        invalidate()
    host = resolve(force=fresh)
    if as_json:
        c = _read_cache() or {}
        print(json.dumps({
            "host": host,
            "ms": round((time.time() - t0) * 1000),
            "pin": _pin(),
            "cache_ok": c.get("ok"),
            "cache_age_s": round(time.time() - float(c.get("ts", 0))) if c.get("ts") else None,
        }, indent=2))
    else:
        print(host)
