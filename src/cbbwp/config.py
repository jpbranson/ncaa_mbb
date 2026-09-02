"""One place where deployment settings come from, for every entry point.

The Mac and a container differ in paths and nothing else, so the difference is
expressed as environment variables rather than as two copies of the code. Every
setting has a working default, so `python3 scripts/serve_live.py` does the right
thing on a laptop with no environment set at all.

The model version is deliberately a setting rather than a constant. Swapping
which model serves is then a config change and a restart -- not an edit -- which
is what keeps "we can change the model later" true in practice. `serve.py` still
refuses to load an artifact whose state-rules version disagrees with the code,
so a careless swap fails loudly at startup instead of silently serving skew.

    CBBWP_ROOT              project root (default: the repo this file is in)
    CBBWP_REGISTRY          model registry dir      (default: <root>/registry)
    CBBWP_MODEL_VERSION     which model to serve    (default: v2)
    CBBWP_CONTEXT           ratings snapshot path   (default: <registry>/context_latest.json)
    CBBWP_LIVE_DIR          JSONL output dir        (default: <root>/data/live)
    CBBWP_FIXTURE_DIR       replay from disk instead of the network (default: unset)
    CBBWP_API_HOST          API bind address        (default: 127.0.0.1)
    CBBWP_API_PORT          API port                (default: 8808)
    CBBWP_API_HISTORY       states kept per game    (default: 240)
    CBBWP_RATINGS_MAX_AGE   days before the snapshot is called stale (default: 3)
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field

_HERE = pathlib.Path(__file__).resolve()
_DEFAULT_ROOT = _HERE.parents[2]


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


@dataclass(frozen=True)
class Settings:
    root: pathlib.Path
    registry: pathlib.Path
    model_version: str
    context_path: pathlib.Path
    live_dir: pathlib.Path
    fixture_dir: pathlib.Path | None
    api_host: str
    api_port: int
    api_history: int
    ratings_max_age_days: float
    _source: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        root = pathlib.Path(_env("CBBWP_ROOT", str(_DEFAULT_ROOT))).resolve()
        registry = pathlib.Path(_env("CBBWP_REGISTRY", str(root / "registry")))
        fixture = os.environ.get("CBBWP_FIXTURE_DIR") or None
        overridden = {k: os.environ[k] for k in os.environ if k.startswith("CBBWP_")}
        return cls(
            root=root,
            registry=registry,
            model_version=_env("CBBWP_MODEL_VERSION", "v2"),
            context_path=pathlib.Path(
                _env("CBBWP_CONTEXT", str(registry / "context_latest.json"))),
            live_dir=pathlib.Path(_env("CBBWP_LIVE_DIR", str(root / "data" / "live"))),
            fixture_dir=pathlib.Path(fixture) if fixture else None,
            api_host=_env("CBBWP_API_HOST", "127.0.0.1"),
            api_port=int(_env("CBBWP_API_PORT", "8808")),
            api_history=int(_env("CBBWP_API_HISTORY", "240")),
            ratings_max_age_days=float(_env("CBBWP_RATINGS_MAX_AGE", "3")),
            _source=overridden,
        )

    def describe(self) -> str:
        """What every entry point prints at startup.

        Printing the resolved settings, and which of them came from the
        environment, is the cheapest possible defence against the class of
        outage where a service is quietly serving from the wrong directory.
        """
        lines = [
            f"  root            {self.root}",
            f"  registry        {self.registry}",
            f"  model version   {self.model_version}",
            f"  ratings         {self.context_path}",
            f"  live output     {self.live_dir}",
            f"  api             http://{self.api_host}:{self.api_port}",
        ]
        if self.fixture_dir:
            lines.append(f"  FIXTURES        {self.fixture_dir}  (no network)")
        if self._source:
            lines.append(f"  from environment: {', '.join(sorted(self._source))}")
        return "\n".join(lines)
