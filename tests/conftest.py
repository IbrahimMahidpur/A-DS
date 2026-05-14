"""
tests/conftest.py — shared fixtures and module-level stubs for the top-level
test suite.

Presidio is imported and instantiated at module level in graph.py.  We must
insert stub modules into sys.modules BEFORE graph.py is ever imported so that
AnalyzerEngine() and AnonymizerEngine() don't hit the real library.

This conftest is picked up automatically by pytest for every test in tests/.
"""
from __future__ import annotations

import sys
import types

# ── Presidio stubs (must run before any import of multimodal_ds.graph) ─────────

class _DummyAnalyzerEngine:
    def analyze(self, text, language="en"):
        return []


class _DummyAnonymizerEngine:
    def anonymize(self, text, analyzer_results):
        class _R:
            def __init__(self, t):
                self.text = t
        return _R(text)


def _make_stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


for _mod_name, _cls_name, _cls in [
    ("presidio_analyzer",  "AnalyzerEngine",  _DummyAnalyzerEngine),
    ("presidio_anonymizer","AnonymizerEngine", _DummyAnonymizerEngine),
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _make_stub_module(_mod_name, **{_cls_name: _cls})
