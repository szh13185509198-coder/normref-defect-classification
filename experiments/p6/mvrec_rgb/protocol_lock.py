"""Canonical-LF SHA256 helpers and the P6.B pre-formal protocol lock artifact.

The three locked protocol documents must not be modified to carry their own
hash (self-referential hashes are forbidden), so their canonical-LF SHA256
values are recorded in a separate machine-readable lock artifact:

    docs/P6B_PRE_FORMAL_PROTOCOL_LOCK.json
"""

import hashlib
import json
import os

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

PROTOCOL_A = os.path.join(
    _ROOT, "docs",
    "P6B_MVREC_RGB_SEMANTIC_PRESERVATION_FAIR_COMPARISON_PROTOCOL.md",
)
PROTOCOL_B = os.path.join(
    _ROOT, "docs",
    "P6B_K8_NESTED_SHOT_SCALING_PROTOCOL.md",
)
DECISION_RULE = os.path.join(
    _ROOT, "docs",
    "P6B_V1_GO_NO_GO_DECISION_RULE.md",
)
LOCK_ARTIFACT = os.path.join(
    _ROOT, "docs",
    "P6B_PRE_FORMAL_PROTOCOL_LOCK.json",
)

PROTOCOL_LOCK_COMMIT = "9714220f0406f47813e541acec4d2bb73b1b17c4"
MVREC_UPSTREAM_COMMIT = "112c187b6de8a7400fad89ca2a201545842a5d84"


def canonical_lf_sha256(path):
    """SHA256 of a file after unifying every line ending to LF."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for raw in f:
            line = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            h.update(line)
    return h.hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compute_protocol_hashes():
    return {
        "protocol_A_canonical_lf_sha256": canonical_lf_sha256(PROTOCOL_A),
        "protocol_B_canonical_lf_sha256": canonical_lf_sha256(PROTOCOL_B),
        "decision_rule_canonical_lf_sha256": canonical_lf_sha256(DECISION_RULE),
    }


def ensure_lock_artifact(protocol_lock_commit=PROTOCOL_LOCK_COMMIT,
                         mvrec_upstream_commit=MVREC_UPSTREAM_COMMIT):
    """Create or verify the protocol lock artifact.

    Returns (artifact, created_or_verified_ok, problems).
    The artifact itself is excluded from hashing (no self-reference).
    """
    hashes = compute_protocol_hashes()
    artifact = {
        "schema_version": "p6b_pre_formal_protocol_lock_v1",
        "protocol_lock_commit": protocol_lock_commit,
        "protocol_A_canonical_lf_sha256": hashes["protocol_A_canonical_lf_sha256"],
        "protocol_B_canonical_lf_sha256": hashes["protocol_B_canonical_lf_sha256"],
        "decision_rule_canonical_lf_sha256": hashes["decision_rule_canonical_lf_sha256"],
        "mvrec_upstream_commit": mvrec_upstream_commit,
    }
    problems = []
    if os.path.exists(LOCK_ARTIFACT):
        with open(LOCK_ARTIFACT, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for key in (
            "protocol_lock_commit",
            "protocol_A_canonical_lf_sha256",
            "protocol_B_canonical_lf_sha256",
            "decision_rule_canonical_lf_sha256",
            "mvrec_upstream_commit",
        ):
            if existing.get(key) != artifact[key]:
                problems.append(
                    f"lock artifact mismatch on {key}: "
                    f"{existing.get(key)!r} != {artifact[key]!r}"
                )
        return artifact, len(problems) == 0, problems

    os.makedirs(os.path.dirname(LOCK_ARTIFACT), exist_ok=True)
    with open(LOCK_ARTIFACT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")
    return artifact, True, []
