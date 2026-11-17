"""Versioned, atomic, verified artefact storage.

Layout under ``<DATA_ROOT>/outputs``::

    <stage>/
        v1/  manifest.json  recipes.parquet  ...
        v2/  ...
        LATEST            # text file containing e.g. "v2"

* Atomicity: files are written to ``<stage>/.tmp_<uuid>/``, re-hashed, then the whole folder is
  moved to ``<stage>/vN`` with ``os.replace``. Any exception removes the temp folder, so a version
  either exists completely (with its manifest) or not at all.
* Consistency: DataFrames listed in ``schemas`` are validated BEFORE anything is written.
* Isolation: each stage writes only under its own folder; stage names must be in config.
* Durability: existing versions are never overwritten; every version has a manifest with sha256,
  row/column counts, dtypes, git commit, UTC timestamp, input references and params.

Supported objects (by type): ``pandas.DataFrame`` -> parquet, ``scipy.sparse`` -> npz,
``dict``/``list`` -> json, ``str`` -> txt, matplotlib ``Figure`` -> png (loads back as the png
``Path``), anything else -> joblib.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from recsys import __version__
from recsys.config import REPO_ROOT
from recsys.contracts import Schema, validate

if TYPE_CHECKING:
    from recsys.config import Config

MANIFEST_NAME = "manifest.json"
LATEST_NAME = "LATEST"
MANIFEST_FORMAT = 1

_STAGE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_OBJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_VERSION_RE = re.compile(r"^v(\d+)$")
_CHUNK = 1 << 20


class ArtifactError(RuntimeError):
    """Invalid request, missing version, or an attempt to overwrite a version."""


class ArtifactIntegrityError(ArtifactError):
    """A stored file does not match the sha256 recorded in its manifest."""


@dataclass(frozen=True)
class Artifact:
    """One stored (or just-saved) stage version."""

    stage: str
    version: str
    path: Path
    manifest: Mapping[str, Any]
    objects: Mapping[str, Any]

    def __getitem__(self, name: str) -> Any:
        return self.objects[name]

    def keys(self) -> Iterable[str]:
        return self.objects.keys()

    def ref(self) -> dict[str, Any]:
        """Reference recorded in downstream manifests' ``inputs``."""
        return {
            "kind": "artifact",
            "stage": self.stage,
            "version": self.version,
            "manifest_sha256": sha256_file(self.path / MANIFEST_NAME),
        }


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Input reference for a non-artefact file (e.g. a raw CSV): path, size and sha256."""
    p = Path(path)
    if not p.is_file():
        raise ArtifactError(f"Input file not found: {p}")
    return {"kind": "file", "path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


class ArtifactStore:
    """Read/write stage versions under ``root`` (normally ``<DATA_ROOT>/outputs``)."""

    def __init__(
        self, root: str | os.PathLike[str], allowed_stages: Iterable[str] | None = None
    ) -> None:
        self.root = Path(root)
        self.allowed_stages = frozenset(allowed_stages) if allowed_stages is not None else None

    @classmethod
    def from_config(cls, cfg: Config) -> ArtifactStore:
        return cls(cfg.outputs_dir, allowed_stages=cfg.stages)

    # ----------------------------------------------------------------- queries

    def stage_dir(self, stage: str) -> Path:
        self._check_stage(stage)
        return self.root / stage

    def versions(self, stage: str) -> list[str]:
        """Existing versions of ``stage``, oldest first."""
        d = self.stage_dir(stage)
        if not d.is_dir():
            return []
        found = [(int(m.group(1)), p.name) for p in d.iterdir() if (m := _VERSION_RE.match(p.name))]
        return [name for _, name in sorted(found)]

    def latest(self, stage: str) -> str | None:
        """Version named in ``<stage>/LATEST``, or None if the stage has never been saved."""
        f = self.stage_dir(stage) / LATEST_NAME
        if not f.is_file():
            return None
        version = f.read_text(encoding="utf-8").strip()
        if not _VERSION_RE.match(version) or not (f.parent / version).is_dir():
            raise ArtifactError(f"{f} points to missing or invalid version {version!r}")
        return version

    # ----------------------------------------------------------------- save

    def save(
        self,
        stage: str,
        objects: Mapping[str, Any],
        inputs: Sequence[Artifact | Mapping[str, Any]] = (),
        params: Mapping[str, Any] | None = None,
        schemas: Mapping[str, Schema] | None = None,
    ) -> Artifact:
        """Atomically write ``objects`` as the next version of ``stage``.

        Args:
            stage: Stage name (must be listed in config ``stages``).
            objects: name -> object. Names become file stems (letters, digits, ``_.-``).
            inputs: Upstream :class:`Artifact` objects or refs (e.g. :func:`describe_file`).
            params: JSON-serialisable parameters used to produce the outputs.
            schemas: name -> Schema for DataFrames to validate before writing.

        Returns:
            The saved :class:`Artifact` (objects are the in-memory objects passed in).

        Raises:
            SchemaValidationError: a DataFrame violates its schema (nothing written).
            ArtifactError: bad stage/object names, or the target version already exists.
        """
        stage_dir = self.stage_dir(stage)
        if not objects:
            raise ArtifactError("save() needs at least one object")
        for name in objects:
            if not isinstance(name, str) or not _OBJECT_RE.match(name):
                raise ArtifactError(f"Invalid object name {name!r} (use letters, digits, _ . -)")

        # Consistency: validate everything before touching the filesystem.
        schemas = dict(schemas or {})
        for name, schema in schemas.items():
            if name not in objects:
                raise ArtifactError(f"Schema given for unknown object {name!r}")
            validate(objects[name], schema)
        input_refs = [_input_ref(i) for i in inputs]
        params_json = _to_jsonable(dict(params or {}))

        stage_dir.mkdir(parents=True, exist_ok=True)
        version = self._next_version(stage)
        target = stage_dir / version
        tmp = stage_dir / f".tmp_{uuid.uuid4().hex}"
        tmp.mkdir()
        try:
            files: dict[str, dict[str, Any]] = {}
            for name, obj in objects.items():
                entry = _write_object(tmp, name, obj)
                if name in schemas:
                    entry["schema"] = schemas[name].name
                files[name] = entry

            manifest = {
                "format": MANIFEST_FORMAT,
                "stage": stage,
                "version": version,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "recsys_version": __version__,
                "git": _git_info(),
                "params": params_json,
                "inputs": input_refs,
                "files": files,
            }
            (tmp / MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            _verify_dir(tmp, manifest)  # re-read from disk: catches partial writes

            if target.exists():
                raise ArtifactError(f"Refusing to overwrite existing version {target}")
            os.replace(tmp, target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        self._write_latest(stage_dir, version)
        return Artifact(stage, version, target, manifest, dict(objects))

    # ----------------------------------------------------------------- load

    def load(
        self, stage: str, version: str = "LATEST", names: Iterable[str] | None = None
    ) -> Artifact:
        """Load a stage version, verifying every file's sha256 against the manifest."""
        resolved = self.latest(stage) if version == "LATEST" else version
        if resolved is None:
            raise ArtifactError(f"Stage {stage!r} has no saved versions under {self.root}")
        if not _VERSION_RE.match(resolved):
            raise ArtifactError(f"Invalid version {resolved!r}")
        path = self.stage_dir(stage) / resolved
        if not path.is_dir():
            raise ArtifactError(f"Version not found: {path}")
        return read_artifact_dir(path, names=names)

    def export(self, stage: str, dest: str | os.PathLike[str], version: str = "LATEST") -> Path:
        """Copy a verified version folder to ``dest`` (e.g. outputs/sample -> repo data/sample).

        Refuses if ``dest`` already holds a manifest (no overwrite). Placeholder files such as
        ``.gitkeep`` are left alone.
        """
        art = self.load(stage, version)
        dest_p = Path(dest)
        if (dest_p / MANIFEST_NAME).exists():
            raise ArtifactError(f"{dest_p} already contains an artefact; remove it first")
        dest_p.mkdir(parents=True, exist_ok=True)
        for entry in art.manifest["files"].values():
            shutil.copy2(art.path / entry["file"], dest_p / entry["file"])
        shutil.copy2(art.path / MANIFEST_NAME, dest_p / MANIFEST_NAME)
        _verify_dir(dest_p, art.manifest)
        return dest_p

    # ----------------------------------------------------------------- internals

    def _check_stage(self, stage: str) -> None:
        if not _STAGE_RE.match(stage):
            raise ArtifactError(f"Invalid stage name {stage!r}")
        if self.allowed_stages is not None and stage not in self.allowed_stages:
            raise ArtifactError(
                f"Unknown stage {stage!r}; add it to config.yaml `stages` "
                f"(known: {sorted(self.allowed_stages)})"
            )

    def _next_version(self, stage: str) -> str:
        existing = self.versions(stage)
        n = int(_VERSION_RE.match(existing[-1]).group(1)) if existing else 0  # type: ignore[union-attr]
        return f"v{n + 1}"

    @staticmethod
    def _write_latest(stage_dir: Path, version: str) -> None:
        tmp = stage_dir / f".{LATEST_NAME}.tmp_{uuid.uuid4().hex}"
        try:
            tmp.write_text(version + "\n", encoding="utf-8")
            os.replace(tmp, stage_dir / LATEST_NAME)
        finally:
            tmp.unlink(missing_ok=True)


def read_artifact_dir(path: str | os.PathLike[str], names: Iterable[str] | None = None) -> Artifact:
    """Read and verify a version folder (also works on an exported copy such as data/sample)."""
    p = Path(path)
    mpath = p / MANIFEST_NAME
    if not mpath.is_file():
        raise ArtifactError(f"No {MANIFEST_NAME} in {p}")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    files = manifest["files"]
    wanted = list(files) if names is None else list(names)
    unknown = [n for n in wanted if n not in files]
    if unknown:
        raise ArtifactError(f"{p}: unknown objects {unknown}; available {sorted(files)}")
    objects = {}
    for name in wanted:
        entry = files[name]
        fpath = p / entry["file"]
        _verify_file(fpath, entry["sha256"])
        objects[name] = _read_object(fpath, entry["kind"])
    return Artifact(manifest["stage"], manifest["version"], p, manifest, objects)


# --------------------------------------------------------------------------- object IO


def _write_object(folder: Path, name: str, obj: Any) -> dict[str, Any]:
    entry: dict[str, Any]
    if isinstance(obj, pd.DataFrame):
        fpath = folder / f"{name}.parquet"
        obj.to_parquet(fpath, engine="pyarrow")
        entry = {
            "kind": "dataframe",
            "n_rows": int(obj.shape[0]),
            "n_cols": int(obj.shape[1]),
            "columns": {str(c): str(t) for c, t in obj.dtypes.items()},
        }
    elif sp.issparse(obj):
        fpath = folder / f"{name}.npz"
        mat = obj if obj.format in ("csr", "csc", "coo", "bsr", "dia") else obj.tocsr()
        sp.save_npz(fpath, mat)
        entry = {
            "kind": "sparse",
            "shape": list(obj.shape),
            "nnz": int(obj.nnz),
            "format": mat.format,
        }
    elif isinstance(obj, (dict, list)):
        fpath = folder / f"{name}.json"
        fpath.write_text(
            json.dumps(_to_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        entry = {"kind": "json"}
    elif isinstance(obj, str):
        fpath = folder / f"{name}.txt"
        fpath.write_text(obj, encoding="utf-8")
        entry = {"kind": "text"}
    elif _is_figure(obj):
        fpath = folder / f"{name}.png"
        obj.savefig(fpath, dpi=150, bbox_inches="tight")
        entry = {"kind": "figure"}
    else:
        fpath = folder / f"{name}.joblib"
        joblib.dump(obj, fpath)
        entry = {"kind": "joblib", "type": f"{type(obj).__module__}.{type(obj).__qualname__}"}
    entry["file"] = fpath.name
    entry["bytes"] = fpath.stat().st_size
    entry["sha256"] = sha256_file(fpath)
    return entry


def _read_object(fpath: Path, kind: str) -> Any:
    if kind == "dataframe":
        table = pq.read_table(fpath)
        df = table.to_pandas()
        # pyarrow returns list columns as numpy arrays; restore Python lists so the
        # "list" schema dtype holds after a round trip.
        for f in table.schema:
            if pa.types.is_list(f.type) or pa.types.is_large_list(f.type):
                df[f.name] = df[f.name].map(lambda v: None if v is None else list(v))
        return df
    if kind == "sparse":
        return sp.load_npz(fpath)
    if kind == "json":
        return json.loads(fpath.read_text(encoding="utf-8"))
    if kind == "text":
        return fpath.read_text(encoding="utf-8")
    if kind == "figure":
        return fpath
    if kind == "joblib":
        return joblib.load(fpath)
    raise ArtifactError(f"Unknown object kind {kind!r} for {fpath}")


def _verify_file(fpath: Path, expected: str) -> None:
    if not fpath.is_file():
        raise ArtifactIntegrityError(f"Missing file listed in manifest: {fpath}")
    actual = sha256_file(fpath)
    if actual != expected:
        raise ArtifactIntegrityError(f"sha256 mismatch for {fpath}: {actual} != {expected}")


def _verify_dir(folder: Path, manifest: Mapping[str, Any]) -> None:
    for entry in manifest["files"].values():
        _verify_file(folder / entry["file"], entry["sha256"])


def _is_figure(obj: Any) -> bool:
    return type(obj).__module__.startswith("matplotlib") and hasattr(obj, "savefig")


def _input_ref(item: Artifact | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(item, Artifact):
        return item.ref()
    if isinstance(item, Mapping) and item.get("kind") in ("artifact", "file"):
        return _to_jsonable(dict(item))
    raise TypeError(
        "inputs must be Artifact objects or refs from Artifact.ref() / describe_file(); "
        f"got {type(item).__name__}"
    )


def _to_jsonable(obj: Any) -> Any:
    """Convert params/json objects; raise TypeError for anything not representable."""
    try:
        return json.loads(json.dumps(obj, default=_json_default))
    except TypeError as exc:
        raise TypeError(f"Object is not JSON-serialisable for the manifest: {exc}") from exc


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if hasattr(o, "item") and callable(o.item):  # numpy scalars
        return o.item()
    if hasattr(o, "tolist") and callable(o.tolist):  # numpy arrays
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            res = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return res.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain") if commit else None
    return {"commit": commit, "dirty": None if status is None else bool(status)}
