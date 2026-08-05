"""A `codex` stand-in, so the attach transport is testable without Codex.

It implements only the surface `attach.py` actually uses — `--version`,
`app-server generate-json-schema`, and enough of `app-server` to start a thread,
run a turn and ask for an approval. The method *names* come from the golden
fixture rather than from this file, so the fake cannot drift from the recorded
protocol without the gate noticing.

Env knobs, all off by default:

* `FAKE_CODEX_DROP` — comma-separated notification methods to omit from the
  generated schema, i.e. pretend a newer build removed them.
* `FAKE_CODEX_ADD` — methods to add, i.e. pretend a newer build gained them.
* `FAKE_CODEX_APPROVAL` — ask for one command approval during the turn.
* `FAKE_CODEX_DECISION_FILE` — write the decision we were given to this path,
  so a test can assert what the client answered.
* `FAKE_CODEX_METHOD_LOG` — append every method name we receive to this path,
  one per line, so a test can assert what did *not* go over the wire.
* `FAKE_CODEX_THREADS` — path to a JSON file `{"threads": [<thread>, …]}` that
  `thread/list` and `thread/read` serve. `thread/list` strips `turns` the way
  the real server does, so a test that forgets to call `thread/read` gets an
  empty history rather than a quietly complete one.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GOLDEN = Path(__file__).with_name("codex-appserver-protocol.json")
VERSION = "codex-cli 0.144.6"


def _schema(methods: list[str]) -> dict:
    return {
        "oneOf": [
            {"type": "object",
             "properties": {"method": {"type": "string", "enum": [m]}}}
            for m in methods
        ]
    }


def generate(out: str) -> int:
    g = json.loads(GOLDEN.read_text())
    notes = list(g["server_notifications"])
    reqs = list(g["server_requests"])
    for m in filter(None, os.environ.get("FAKE_CODEX_DROP", "").split(",")):
        if m in notes:
            notes.remove(m)
        if m in reqs:
            reqs.remove(m)
    notes += [m for m in filter(None, os.environ.get("FAKE_CODEX_ADD", "").split(","))]
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ServerNotification.json").write_text(json.dumps(_schema(notes)))
    (d / "ServerRequest.json").write_text(json.dumps(_schema(reqs)))
    return 0


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def note(method: str, **params) -> None:
    send({"jsonrpc": "2.0", "method": method, "params": params})


def _threads() -> list[dict]:
    path = os.environ.get("FAKE_CODEX_THREADS")
    if not path or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text()).get("threads") or []


def app_server() -> int:
    thread_id = "th_fake"
    pending_approval: int | None = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        rid, method = msg.get("id"), msg.get("method")
        log = os.environ.get("FAKE_CODEX_METHOD_LOG")
        if log and method:
            with open(log, "a") as fh:
                fh.write(f"{method}\n")

        if method is None and rid is not None and rid == pending_approval:
            path = os.environ.get("FAKE_CODEX_DECISION_FILE")
            if path:
                Path(path).write_text(json.dumps(msg))
            _finish_turn(thread_id)
            continue

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": rid,
                  "result": {"userAgent": f"{VERSION} (fake)",
                             "codexHome": "/tmp/fake-codex",
                             "platformFamily": "unix", "platformOs": "darwin"}})
        elif method == "thread/start":
            send({"jsonrpc": "2.0", "id": rid,
                  "result": {"thread": {"id": thread_id, "cliVersion": VERSION},
                             "cwd": msg["params"].get("cwd"),
                             "model": "gpt-5", "modelProvider": "openai",
                             "approvalPolicy": msg["params"].get("approvalPolicy"),
                             "approvalsReviewer": None,
                             "sandbox": msg["params"].get("sandbox")}})
            note("thread/started", threadId=thread_id)
        elif method == "turn/start":
            send({"jsonrpc": "2.0", "id": rid, "result": {"turn": {"id": "turn_1"}}})
            note("turn/started", threadId=thread_id, turnId="turn_1")
            note("item/started", threadId=thread_id,
                 item={"id": "it_1", "type": "commandExecution",
                       "command": "pytest -q", "cwd": "/repo"})
            note("item/completed", threadId=thread_id,
                 item={"id": "it_1", "type": "commandExecution",
                       "command": "pytest -q", "exitCode": 0, "durationMs": 1200,
                       "aggregatedOutput": "3 passed"})
            if os.environ.get("FAKE_CODEX_APPROVAL"):
                pending_approval = 9001
                send({"jsonrpc": "2.0", "id": pending_approval,
                      "method": "item/commandExecution/requestApproval",
                      "params": {"threadId": thread_id, "itemId": "it_2",
                                 "command": "rm -rf build"}})
            else:
                _finish_turn(thread_id)
        elif method == "thread/list":
            page = int((msg.get("params") or {}).get("limit") or 25)
            cursor = int((msg.get("params") or {}).get("cursor") or 0)
            rows = _threads()[cursor:cursor + page]
            nxt = cursor + len(rows)
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "data": [{k: v for k, v in t.items() if k != "turns"} for t in rows],
                "nextCursor": str(nxt) if nxt < len(_threads()) else None,
            }})
        elif method == "thread/read":
            want = (msg.get("params") or {}).get("threadId")
            hit = next((t for t in _threads() if t.get("id") == want), None)
            if hit is None:
                send({"jsonrpc": "2.0", "id": rid,
                      "error": {"code": -32602, "message": f"no thread {want}"}})
            else:
                send({"jsonrpc": "2.0", "id": rid, "result": {"thread": hit}})
        elif method in ("initialized", "thread/unsubscribe", "turn/interrupt"):
            continue
        elif rid is not None:
            send({"jsonrpc": "2.0", "id": rid,
                  "error": {"code": -32601, "message": f"no such method {method}"}})
    return 0


def _finish_turn(thread_id: str) -> None:
    note("thread/tokenUsage/updated", threadId=thread_id,
         usage={"inputTokens": 1200, "cachedInputTokens": 900,
                "outputTokens": 300, "reasoningOutputTokens": 120})
    note("turn/completed", threadId=thread_id, turnId="turn_1",
         usage={"inputTokens": 1200, "cachedInputTokens": 900,
                "outputTokens": 300, "reasoningOutputTokens": 120})


def main(argv: list[str]) -> int:
    if "--version" in argv:
        print(VERSION)
        return 0
    if argv[:1] == ["app-server"]:
        rest = argv[1:]
        if rest[:1] == ["generate-json-schema"]:
            out = rest[rest.index("--out") + 1] if "--out" in rest else "."
            return generate(out)
        if rest[:1] == ["proxy"]:
            return app_server()
        return app_server()
    sys.stderr.write(f"fake codex: unsupported {argv}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
