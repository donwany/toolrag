"""
LangGraph / LangChain serde compatibility.

Import this module before any `langgraph` import. LangGraph's JsonPlusSerializer
uses `Reviver()` with `allowed_objects=None`, which triggers a
LangChainPendingDeprecationWarning on every CLI run.
"""

from __future__ import annotations

import warnings
from typing import Any

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from langchain_core.load.load import Reviver

_original_reviver_init = Reviver.__init__


def _reviver_init_with_explicit_allowlist(
    self: Reviver,
    allowed_objects: Any = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    if allowed_objects is None:
        allowed_objects = "messages"
    _original_reviver_init(self, allowed_objects, *args, **kwargs)


Reviver.__init__ = _reviver_init_with_explicit_allowlist  # type: ignore[method-assign]

warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=r"The default value of `allowed_objects`.*",
)
