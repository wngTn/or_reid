"""Checkpoint discovery helpers.

The public entry point is `resolve_checkpoint_hint`. A hint can be a checkpoint
file, a directory containing checkpoints, or an alias from `ckpts/manifest.yaml`.
"""

from pathlib import Path
import re

import yaml

CKPT_URI_PREFIX = "ckpts://"
DEFAULT_MANIFEST_PATH = Path("ckpts") / "manifest.yaml"
VALID_SPLITS = [0, 1, 2, 3]


def _normalize_hint(value):
    value = str(value).strip()
    if value.startswith(CKPT_URI_PREFIX):
        value = value[len(CKPT_URI_PREFIX) :]
    return value.strip("/")


def _normalize_key(value):
    return re.sub(r"[^a-z0-9]+", "/", str(value).lower()).strip("/")


def _candidate_paths(hint):
    path = Path(hint).expanduser()
    paths = [path]
    if not path.is_absolute():
        paths.append(Path.cwd() / path)
    return paths


def _infer_modality(cfgs):
    data_in_use = cfgs.get("data_cfg", {}).get("data_in_use", [])
    if len(data_in_use) > 3 and data_in_use[3]:
        return "rgb"
    if any(bool(data_in_use[idx]) for idx in [1, 2, 4] if idx < len(data_in_use)):
        return "depth"
    return None


def _load_manifest(manifest_path=DEFAULT_MANIFEST_PATH):
    for path in _candidate_paths(manifest_path):
        if path.is_file():
            with path.open("r") as stream:
                manifest = yaml.safe_load(stream) or {}
            return manifest, path.parent
    return {}, Path.cwd()


def _manifest_entries(manifest):
    entries = manifest.get("checkpoints", [])
    if isinstance(entries, dict):
        return entries.values()
    return entries


def _resolve_manifest_path(path_value, manifest_root):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = manifest_root / path
    return path


def _entry_aliases(entry):
    aliases = [entry.get("name", "")]
    aliases.extend(entry.get("aliases", []))
    return {_normalize_key(alias) for alias in aliases if alias}


def _match_manifest_entry(hint, cfgs, manifest):
    normalized_hint = _normalize_key(hint)
    if not normalized_hint:
        return None

    for entry in _manifest_entries(manifest):
        if normalized_hint in _entry_aliases(entry):
            return entry

    modality = _infer_modality(cfgs)
    dataset = cfgs.get("data_cfg", {}).get("train_dataset_name")
    for entry in _manifest_entries(manifest):
        if normalized_hint != _normalize_key(entry.get("name", "")):
            continue
        if modality and entry.get("modality") and _normalize_key(entry["modality"]) != _normalize_key(modality):
            continue
        if dataset and entry.get("train_dataset"):
            if _normalize_key(entry["train_dataset"]) != _normalize_key(dataset):
                continue
        return entry
    return None


def _resolve_manifest_entry(entry, split, manifest_root):
    if "file" in entry:
        path = _resolve_manifest_path(entry["file"], manifest_root)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint alias '{entry.get('name')}' points to missing file: {path}")
        return path.as_posix()

    splits = entry.get("splits", {})
    split_key = str(split)
    path_value = splits.get(split, splits.get(split_key))
    if path_value is None:
        available = sorted(str(key) for key in splits.keys())
        raise FileNotFoundError(
            f"Checkpoint alias '{entry.get('name')}' has no file for split {split}. "
            f"Available splits: {available}"
        )

    path = _resolve_manifest_path(path_value, manifest_root)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint alias '{entry.get('name')}' points to missing file: {path}")
    return path.as_posix()


def _list_checkpoint_files(directory):
    return sorted(p for p in directory.rglob("*.pt") if p.is_file())


def _filter_by_modality(paths, cfgs):
    modality = _infer_modality(cfgs)
    if modality is None:
        return paths

    modality_aliases = {
        "depth": {"depth", "pointcloud", "pc"},
        "rgb": {"rgb"},
    }
    aliases = modality_aliases.get(modality, {modality})

    matches = []
    for path in paths:
        normalized_parts = {_normalize_key(part).replace("/", "") for part in path.parts}
        normalized_parts.add(_normalize_key(path.stem).replace("/", ""))
        if any(alias in normalized_parts or alias in _normalize_key(path.as_posix()) for alias in aliases):
            matches.append(path)
    return matches or paths


def _filter_by_dataset(paths, cfgs):
    dataset = cfgs.get("data_cfg", {}).get("train_dataset_name")
    if not dataset:
        return paths
    dataset_key = _normalize_key(dataset).replace("/", "")
    matches = [
        path for path in paths if dataset_key in _normalize_key(path.as_posix()).replace("/", "")
    ]
    return matches or paths


def _split_number(path):
    match = re.search(r"split[-_]?([0-9]+)(?:$|[^0-9])", path.stem.lower())
    return int(match.group(1)) if match else None


def _resolve_directory(directory, cfgs, split):
    all_ckpts = _list_checkpoint_files(directory)
    if not all_ckpts:
        raise FileNotFoundError(f"No .pt checkpoints found under: {directory}")

    candidates = _filter_by_modality(all_ckpts, cfgs)
    candidates = _filter_by_dataset(candidates, cfgs)

    if len(candidates) == 1:
        return candidates[0].as_posix()

    split_to_paths = {}
    for path in candidates:
        number = _split_number(path)
        if number is not None:
            split_to_paths.setdefault(number, []).append(path)

    one_based_numbers = {split + 1 for split in VALID_SPLITS}
    zero_based_numbers = set(VALID_SPLITS)
    available_numbers = set(split_to_paths.keys())
    if one_based_numbers.issubset(available_numbers):
        target_split_number = split + 1
    elif zero_based_numbers.issubset(available_numbers):
        target_split_number = split
    else:
        target_split_number = split if split in split_to_paths else split + 1

    matches = split_to_paths.get(target_split_number, [])
    if len(matches) == 1:
        return matches[0].as_posix()
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous checkpoints for split {split}: {[p.as_posix() for p in matches]}")

    names = [path.as_posix() for path in candidates]
    raise FileNotFoundError(
        f"Could not find a checkpoint for split {split} under {directory}. "
        f"Candidate checkpoints: {names}"
    )


def resolve_checkpoint_hint(hint, cfgs, split, manifest_path=DEFAULT_MANIFEST_PATH):
    """Resolve a checkpoint hint to a concrete checkpoint file.

    Args:
        hint: File path, directory path, or manifest alias. Aliases may also use
            the `ckpts://` prefix.
        cfgs: Loaded experiment config, used to infer modality for directories.
        split: Zero-based split index.
        manifest_path: Optional path to the checkpoint manifest.

    Returns:
        A string path to a checkpoint file.
    """
    normalized_hint = _normalize_hint(hint)

    for path in _candidate_paths(normalized_hint):
        if path.is_file():
            return path.as_posix()
        if path.is_dir():
            return _resolve_directory(path, cfgs, split)

    manifest, manifest_root = _load_manifest(manifest_path)
    entry = _match_manifest_entry(normalized_hint, cfgs, manifest)
    if entry:
        return _resolve_manifest_entry(entry, split, manifest_root)

    raise FileNotFoundError(
        f"Checkpoint hint '{hint}' is not a file, directory, or alias in {manifest_path}."
    )
