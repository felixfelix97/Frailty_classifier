"""L3 Logger: append-only audit log with a SHA-256 hash chain (Art. 12, REQ-UC1-08).

Each entry's hash is computed over the previous entry's hash plus this
entry's record and timestamp, so tampering with any past entry (or
reordering/deleting one) breaks every hash after it - detectable via
verify_chain(). The class exposes no delete or update method; the only
way to add data is append(). The backing file is opened strictly in
append mode ("a"), never overwritten, so the durable record survives
even if the in-memory list were somehow mutated.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS_HASH = "0" * 64


def _canonical(record):
    """Deterministic JSON string, so the same record always hashes the same way."""
    return json.dumps(record, sort_keys=True, default=str)


def _compute_hash(prev_hash, record, timestamp):
    payload = prev_hash + _canonical(record) + timestamp
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLogger:
    """Append-only, hash-linked audit log. No delete/update methods by design."""

    def __init__(self, path="audit_log.jsonl"):
        self._path = Path(path)
        self._entries = []
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._entries.append(json.loads(line))

    def append(self, record):
        """The only way to add data. Returns the new entry."""
        prev_hash = self._entries[-1]["hash"] if self._entries else GENESIS_HASH
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = {
            "index": len(self._entries),
            "timestamp": timestamp,
            "record": record,
            "prev_hash": prev_hash,
            "hash": _compute_hash(prev_hash, record, timestamp),
        }
        self._entries.append(entry)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    @property
    def entries(self):
        """Read-only view - a copy, so callers can't mutate the internal log."""
        return tuple(dict(e) for e in self._entries)

    def __len__(self):
        return len(self._entries)

    def verify_chain(self):
        """Re-walks the whole chain from genesis, recomputing every hash and
        checking every link. Returns (is_valid, problems) - problems is a list
        of human-readable strings naming exactly which entry/entries failed
        and how, empty if the chain is fully intact."""
        problems = []
        expected_prev = GENESIS_HASH
        for i, entry in enumerate(self._entries):
            if entry["index"] != i:
                problems.append(f"entry {i}: index field is {entry['index']}, expected {i}")
            if entry["prev_hash"] != expected_prev:
                problems.append(
                    f"entry {i}: prev_hash={entry['prev_hash'][:12]}... does not match "
                    f"the actual previous entry's hash={expected_prev[:12]}... (chain broken)"
                )
            recomputed = _compute_hash(entry["prev_hash"], entry["record"], entry["timestamp"])
            if recomputed != entry["hash"]:
                problems.append(
                    f"entry {i}: stored hash={entry['hash'][:12]}... does not match "
                    f"recomputed hash={recomputed[:12]}... (record was tampered with)"
                )
            expected_prev = entry["hash"]
        return (len(problems) == 0, problems)
