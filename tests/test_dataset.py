"""Unit tests for dataset.py that never touch the network.

The pipeline stages are stubbed at the project.py boundary (``search_studies`` /
``scan_iter``), so these cover the checkpoint/resume machinery itself:

    python tests/test_dataset.py
    pytest tests/test_dataset.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset as D  # noqa: E402
from project import Project  # noqa: E402


def _project(acc, record_count=10):
    return Project.from_dict({"accession": acc, "record_count": record_count,
                              "title": f"study {acc}", "published": "2021-01-01"})


class _Harness:
    """Stub search_studies/scan_iter and count how often each is really used."""

    def __init__(self, accessions, crash_at=None, oversized=()):
        self.accessions = list(accessions)
        self.crash_at = crash_at
        self.oversized = set(oversized)
        self.searches = 0
        self.scanned: list[str] = []

    def search_studies(self, **kw):
        self.searches += 1
        return list(self.accessions)

    def scan_iter(self, accessions, **kw):
        for acc in accessions:
            if acc == self.crash_at:
                self.crash_at = None  # crash once, then let the rerun through
                raise KeyboardInterrupt(f"interrupted at {acc}")
            self.scanned.append(acc)
            if acc in self.oversized:
                yield acc, None, ValueError("record_count exceeds max_records")
            else:
                yield acc, _project(acc), None

    def __enter__(self):
        self._orig = (D.search_studies, D.scan_iter)
        D.search_studies, D.scan_iter = self.search_studies, self.scan_iter
        return self

    def __exit__(self, *exc):
        D.search_studies, D.scan_iter = self._orig
        return False


def test_successful_run_writes_output_and_removes_the_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.json")
        with _Harness([f"SRP{i}" for i in range(5)]) as h:
            D.save_recent_studies(path, max_studies=5, checkpoint_every=2)
        saved = json.load(open(path, encoding="utf-8"))
        assert [s["accession"] for s in saved] == [f"SRP{i}" for i in range(5)]
        assert h.searches == 1
        # no checkpoint and no half-written temp file left behind
        assert os.listdir(d) == ["out.json"]


def test_a_crashed_run_resumes_without_researching_or_rescanning():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.json")
        accs = [f"SRP{i}" for i in range(10)]
        h = _Harness(accs, crash_at="SRP6")
        with h:
            try:
                D.save_recent_studies(path, max_studies=10, checkpoint_every=2)
            except KeyboardInterrupt:
                pass
            # the crash left a checkpoint, not an output file
            assert os.path.exists(D._checkpoint_path(path))
            assert not os.path.exists(path)
            first_pass = list(h.scanned)

            D.save_recent_studies(path, max_studies=10, checkpoint_every=2)

        assert first_pass == accs[:6]          # crashed just before SRP6
        assert h.scanned == accs[:6] + accs[6:]  # resumed there, nothing rescanned
        assert h.searches == 1                 # the enumeration was not repeated
        saved = json.load(open(path, encoding="utf-8"))
        assert [s["accession"] for s in saved] == accs
        assert not os.path.exists(D._checkpoint_path(path))


def test_resume_false_starts_over():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.json")
        accs = [f"SRP{i}" for i in range(6)]
        h = _Harness(accs, crash_at="SRP4")
        with h:
            try:
                D.save_recent_studies(path, max_studies=6, checkpoint_every=2)
            except KeyboardInterrupt:
                pass
            D.save_recent_studies(path, max_studies=6, checkpoint_every=2, resume=False)
        assert h.searches == 2  # re-enumerated instead of reusing the checkpoint
        assert json.load(open(path, encoding="utf-8"))[0]["accession"] == "SRP0"


def test_a_checkpoint_from_different_search_params_is_ignored():
    # the study list is a *random* sample, so resuming a 10-study harvest into a
    # 20-study one would silently splice two different samples together
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.json")
        h = _Harness([f"SRP{i}" for i in range(10)], crash_at="SRP4")
        with h:
            try:
                D.save_recent_studies(path, max_studies=10, checkpoint_every=2)
            except KeyboardInterrupt:
                pass
            assert D._load_checkpoint(D._checkpoint_path(path),
                                      {"max_studies": 999}) is None
            D.save_recent_studies(path, max_studies=20, checkpoint_every=2)
        assert h.searches == 2


def test_oversized_studies_are_checkpointed_as_skipped_not_retried():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.json")
        accs = [f"SRP{i}" for i in range(8)]
        h = _Harness(accs, crash_at="SRP5", oversized={"SRP1", "SRP3"})
        with h:
            try:
                D.save_recent_studies(path, max_studies=8, checkpoint_every=1)
            except KeyboardInterrupt:
                pass
            D.save_recent_studies(path, max_studies=8, checkpoint_every=1)
        # dropped studies are remembered as done, so the rerun doesn't refetch them
        assert h.scanned == accs
        saved = [s["accession"] for s in json.load(open(path, encoding="utf-8"))]
        assert saved == ["SRP0", "SRP2", "SRP4", "SRP5", "SRP6", "SRP7"]


def test_write_json_is_atomic_and_leaves_no_temp_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.json")
        D._write_json(path, {"version": 1, "done": {}})
        D._write_json(path, {"version": 1, "done": {"SRP1": None}})
        assert os.listdir(d) == ["ckpt.json"]
        assert json.load(open(path, encoding="utf-8"))["done"] == {"SRP1": None}


def test_corrupt_checkpoint_is_ignored_rather_than_crashing():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.json")
        with open(D._checkpoint_path(path), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        with _Harness(["SRP1"]) as h:
            D.save_recent_studies(path, max_studies=1)
        assert h.searches == 1
        assert json.load(open(path, encoding="utf-8"))[0]["accession"] == "SRP1"


# --------------------------------------------------------------------------- #
# standalone runner (mirrors test_offline.py)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import traceback

    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
