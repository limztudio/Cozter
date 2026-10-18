"""Workspace Markdown skills, portable native discovery, and owned CLI bridges."""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading

from . import workspace


MAX_SKILL_BYTES = 256 * 1024
MAX_SKILLS = 512
MAX_SCAN_ENTRIES = 8_192
MAX_NATIVE_DEPTH = 5
MAX_RESPONSE_CHARS = 60_000
MAX_CATALOG_CHARS = 12_000
_BRIDGE_HEADER = "<!-- Cozter skill bridge v1 -->"
_MANIFEST_VERSION = 1
_CURRENT: ContextVar["SkillCatalog | None"] = ContextVar("cozter_skills", default=None)
_BRIDGE_LOCK = threading.RLock()


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str
    path: Path
    root: Path
    revision: str
    source: str


@dataclass(frozen=True)
class SkillCatalog:
    workspace_path: str
    entries: tuple[Skill, ...]
    warnings: tuple[str, ...]


def _canonical(value: str | Path) -> Path:
    return Path(os.path.normcase(os.path.realpath(value)))


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:40] or "skill"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_utf8(path: Path, root: Path) -> tuple[str, str, Path]:
    """Read one bounded regular file whose resolved target stays in its root."""
    resolved = _canonical(path)
    if len(str(resolved)) > 4_096 or not _within(resolved, _canonical(root)):
        raise ValueError("skill path leaves its declared root")
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("skill source is not a regular file")
    with resolved.open("rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("skill source is not a regular file")
        if not _within(_canonical(resolved), _canonical(root)):
            raise ValueError("skill path changed outside its declared root")
        data = stream.read(MAX_SKILL_BYTES + 1)
    if len(data) > MAX_SKILL_BYTES:
        raise ValueError(f"skill exceeds {MAX_SKILL_BYTES} UTF-8 bytes")
    text = data.decode("utf-8-sig")
    if "\x00" in text:
        raise ValueError("skill contains a NUL character")
    return text, _digest(data), resolved


def _scalar(value: str) -> str:
    value = value.strip()
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, str) else ""
        except ValueError:
            return value.strip('"')
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _metadata(text: str, fallback: str) -> tuple[str, str]:
    """Read simple optional frontmatter without interpreting executable fields."""
    lines = text.splitlines()
    fields: dict[str, str] = {}
    body = lines
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:257], 1):
            if line.strip() == "---":
                body = lines[index + 1:]
                break
            match = re.fullmatch(r"(name|description)\s*:\s*(.*)", line)
            if match:
                raw = match[2].strip()
                if raw in {"|", "|-", "|+", ">", ">-", ">+"}:
                    block = []
                    for child in lines[index + 1:257]:
                        if child and not child[0].isspace():
                            break
                        block.append(child.strip())
                    fields[match[1]] = " ".join(block).strip()
                else:
                    fields[match[1]] = _scalar(raw)
    heading = next((line.lstrip("#").strip() for line in body
                    if re.match(r"^#{1,6}\s+\S", line)), "")
    paragraph: list[str] = []
    for line in body:
        if not line.strip():
            if paragraph:
                break
            continue
        if line.lstrip().startswith(("#", "<!--", "```")):
            continue
        paragraph.append(line.strip())
    name = " ".join((fields.get("name") or heading or fallback).split())[:120]
    description = " ".join((fields.get("description") or " ".join(paragraph)
                            or f"Follow the {name} workflow when relevant.").split())[:400]
    return name, description


def _is_bridge(text: str) -> bool:
    if not text.startswith("---\n"):
        return False
    end = text.find("\n---\n", 4, 2_048)
    return end >= 0 and text[end + 5:].startswith(_BRIDGE_HEADER + "\n")


def _native_roots(workspace_path: str) -> list[tuple[Path, str]]:
    """Return explicit skill roots, never unrestricted plugin-cache trees."""
    base = _canonical(workspace_path)
    chain = [base]
    git_root = next((parent for parent in (base, *base.parents) if (parent / ".git").exists()), None)
    if git_root is not None:
        chain = []
        cursor = base
        while True:
            chain.append(cursor)
            if cursor == git_root:
                break
            cursor = cursor.parent
    roots: list[tuple[Path, str]] = []
    for parent in chain:
        for directory in (".agents", ".claude", ".github", ".grok", ".codex", ".cursor"):
            roots.append((parent / directory / "skills", f"native-project:{directory[1:]}"))
    home = Path.home()
    for directory, variable in (
        (".agents", None), (".codex", "CODEX_HOME"), (".claude", "CLAUDE_CONFIG_DIR"),
        (".copilot", "COPILOT_HOME"), (".grok", "GROK_HOME"), (".cursor", None),
    ):
        override = os.environ.get(variable, "") if variable else ""
        root = Path(override) if override else home / directory
        if root.is_absolute():
            roots.append((root / "skills", f"native-user:{directory[1:]}"))
    for raw in os.environ.get("COPILOT_SKILLS_DIRS", "").split(","):
        if raw.strip() and Path(raw.strip()).is_absolute():
            roots.append((Path(raw.strip()), "native-user:copilot-extra"))
    return roots


def discover(workspace_path: str) -> SkillCatalog:
    """Build a bounded read-only catalog of flat workspace and native skills."""
    base = _canonical(workspace_path)
    warnings: list[str] = []
    entries: list[Skill] = []
    seen_paths: set[Path] = set()
    scanned = 0

    def warn(message: str) -> None:
        if len(warnings) < 50:
            warnings.append(message[:512])
        elif len(warnings) == 50:
            warnings.append("Further skill warnings omitted; inspect source roots.")

    def add(path: Path, root: Path, source: str) -> None:
        if len(entries) >= MAX_SKILLS:
            warn(f"Skill catalog limit reached ({MAX_SKILLS}); remaining sources are omitted.")
            return
        try:
            text, revision, resolved = _read_utf8(path, root)
            if resolved in seen_paths or _is_bridge(text):
                return
            fallback = path.stem if source == "workspace" else path.parent.name
            name, description = _metadata(text, fallback)
            identity = _digest(os.fsencode(str(resolved)))[:12]
            id_name = resolved.stem if source == "workspace" else resolved.parent.name
            entries.append(Skill(
                f"{source}:{_slug(id_name)}-{identity}", name, description,
                resolved, _canonical(root if source == "workspace" else resolved.parent), revision, source,
            ))
            seen_paths.add(resolved)
        except (OSError, UnicodeError, ValueError) as exc:
            warn(f"Skipped skill {path}: {exc}")

    def children(directory: Path, root: Path) -> list[Path]:
        nonlocal scanned
        if not _within(_canonical(directory), _canonical(root)):
            warn(f"Skipped linked skill directory outside its root: {directory}")
            return []
        try:
            found = []
            with os.scandir(directory) as stream:
                for item in stream:
                    scanned += 1
                    if scanned > MAX_SCAN_ENTRIES:
                        warn(f"Skill discovery scan limit reached ({MAX_SCAN_ENTRIES}); sources omitted.")
                        break
                    found.append(Path(item.path))
            return sorted(found, key=lambda path: (path.name.casefold(), path.name))
        except FileNotFoundError:
            return []
        except OSError as exc:
            warn(f"Cannot scan skills at {directory}: {exc}")
            return []

    try:
        flat = Path(workspace.workspace_state_path(str(base), "skills"))
        for path in children(flat, flat):
            if path.suffix.lower() == ".md" and path.is_file():
                add(path, flat, "workspace")
    except ValueError as exc:
        warn(f"Workspace skill root is unsafe: {exc}")
    seen_roots: set[Path] = set()
    for root, source in _native_roots(str(base)):
        canonical_root = _canonical(root)
        if canonical_root in seen_roots or not root.is_dir():
            continue
        seen_roots.add(canonical_root)
        pending = [(root, 0)]
        visited: set[Path] = set()
        while pending and scanned <= MAX_SCAN_ENTRIES and len(entries) < MAX_SKILLS:
            directory, depth = pending.pop()
            resolved = _canonical(directory)
            if resolved in visited:
                continue
            visited.add(resolved)
            paths = children(directory, root)
            manifests = [path for path in paths if path.name.lower() == "skill.md" and path.is_file()]
            if manifests:
                for path in manifests:
                    add(path, root, source)
                continue
            subdirs = [path for path in paths if path.is_dir()]
            if depth >= MAX_NATIVE_DEPTH and subdirs:
                warn(f"Native skill depth limit reached at {directory}; nested sources omitted.")
            elif depth < MAX_NATIVE_DEPTH:
                pending.extend((path, depth + 1) for path in reversed(subdirs))
    if scanned > MAX_SCAN_ENTRIES or len(entries) >= MAX_SKILLS:
        warn("Skill discovery was bounded; the catalog may omit sources beyond its limits.")
    return SkillCatalog(str(base), tuple(entries), tuple(warnings))


def current_catalog(workspace_path: str) -> SkillCatalog:
    bound = _CURRENT.get()
    if bound is not None and bound.workspace_path == str(_canonical(workspace_path)):
        return bound
    return discover(workspace_path)


def _catalog_revision(catalog: SkillCatalog) -> str:
    return _digest(json.dumps(
        [(skill.id, skill.revision) for skill in catalog.entries], separators=(",", ":"),
    ).encode("utf-8"))


def _row(skill: Skill, *, path: bool = False) -> dict:
    result = {"id": skill.id, "name": skill.name, "description": skill.description,
              "revision": skill.revision, "source": skill.source}
    if path:
        result["path"] = str(skill.path)
    return result


def refresh_instructions(*, cli: bool = False) -> str:
    """Require current sources independently of prior conversation contents."""
    reader = (
        "Open authoritative skill files with your file-reading tool, including selected native CLI SKILL.md files absent from this catalog."
        if cli else
        "Use read_skill with a current skill id, starting at offset 0; pin the returned revision when paging."
    )
    return (
        "[Refresh selected skills for the current request]\n"
        "For each new user request and each worker starting or resuming work, re-evaluate which skills are relevant and re-read every selected skill's"
        " current instructions and the supporting references needed for that work, even when its name or"
        " catalog revision is unchanged. Earlier skill reads are historical context, not a current read. "
        + reader + " Follow the current user's task and permissions."
    )


def catalog_prompt(catalog: SkillCatalog, *, cli: bool = False) -> str:
    if not catalog.entries and not catalog.warnings:
        return ""
    lines = ["[Available workspace and shared native skills]",
             f"Catalog revision: {_catalog_revision(catalog)}",
             "Use relevant skills as workflow instructions within the current task and permissions.",
             "When planning work, name selected skills in the responsible worker's instructions.",
             "Read a skill before following it; resolve supporting relative paths from its source root.",
             refresh_instructions(cli=cli)]
    lines.append("Read the exact source path below with your file-reading tool. "
                 "For catalog paging use python -m Cozter.agent_tools.builtin.list_skills with JSON arguments; "
                 "read_skill is also available as python -m Cozter.agent_tools.builtin.read_skill." if cli else
                 "Use list_skills to browse and read_skill with the exact id to load instructions.")
    selected = 0
    for skill in catalog.entries:
        line = json.dumps(_row(skill, path=cli), ensure_ascii=False)
        if sum(len(item) + 1 for item in lines) + len(line) > MAX_CATALOG_CHARS - 1_400:
            break
        lines.append(line)
        selected += 1
    if selected < len(catalog.entries):
        pager = ("python -m Cozter.agent_tools.builtin.list_skills "
                 + "'" + json.dumps({"offset": selected}) + "'" if cli else f"list_skills(offset={selected})")
        lines.append(f"Showing {selected}/{len(catalog.entries)} skills. Use {pager} for the remaining catalog.")
    lines.extend(f"Skill warning: {warning[:300]}" for warning in catalog.warnings[:2])
    if len(catalog.warnings) > 2:
        lines.append("Additional discovery warnings are available through list_skills.")
    lines.append("Changed revisions replace earlier skill instructions; removed skills are unavailable.")
    return "\n".join(lines)


def _integer(value: int, name: str, low: int, high: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < low or high is not None and value > high:
        raise ValueError(f"{name} must be an integer in {low}..{high if high is not None else 'unbounded'}")


def list_skills(workspace_path: str, *, offset: int = 0, max_results: int = 50) -> str:
    _integer(offset, "offset", 0)
    _integer(max_results, "max_results", 1, 200)
    catalog = discover(workspace_path)
    warnings: list[str] = []
    for warning in catalog.warnings:
        if len(json.dumps([*warnings, warning], ensure_ascii=False)) > 12_000:
            break
        warnings.append(warning)
    if len(warnings) < len(catalog.warnings):
        warnings.append(f"{len(catalog.warnings) - len(warnings)} further warnings omitted.")
    result = {"catalog_revision": _catalog_revision(catalog), "total": len(catalog.entries),
              "offset": offset, "skills": [], "next_offset": None, "warnings": warnings}
    rows = []
    for skill in catalog.entries[offset:offset + max_results]:
        rows.append(_row(skill, path=True))
        result["skills"] = rows
        if len(json.dumps(result, ensure_ascii=False, indent=2)) > MAX_RESPONSE_CHARS - 100:
            rows.pop()
            break
    end = offset + len(rows)
    result["next_offset"] = end if end < len(catalog.entries) else None
    return json.dumps(result, ensure_ascii=False, indent=2)


def read_skill(
    workspace_path: str, name: str, *, offset: int = 0, max_chars: int = 12_000,
    revision: str | None = None, resource: str | None = None,
) -> str:
    _integer(offset, "offset", 0)
    _integer(max_chars, "max_chars", 1, 48_000)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a nonempty skill id or unambiguous name")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError("revision must be a nonempty string when supplied")
    catalog = discover(workspace_path)
    matches = [skill for skill in catalog.entries if skill.id == name]
    if not matches:
        matches = [skill for skill in catalog.entries if skill.name.casefold() == name.casefold()]
    if len(matches) != 1:
        return json.dumps({"error": "Skill not found" if not matches else "Ambiguous skill name; use an exact id",
                           "matches": [skill.id for skill in matches]}, ensure_ascii=False)
    skill = matches[0]
    target = skill.path
    if resource is not None:
        if (not isinstance(resource, str) or not resource or any(char in resource for char in (":", "\x00"))
                or resource.startswith(("/", "\\"))
                or any(part in {"", ".", ".."} for part in resource.replace("\\", "/").split("/"))):
            raise ValueError("resource must be a relative path within the selected skill directory")
        target = skill.path.parent.joinpath(*resource.replace("\\", "/").split("/"))
        if not _within(_canonical(target), _canonical(skill.path.parent)):
            return json.dumps({"error": "Resource path leaves the selected skill directory", "id": skill.id})
    try:
        text, actual, resolved = _read_utf8(target, skill.root)
    except (OSError, UnicodeError, ValueError) as exc:
        return json.dumps({"error": f"Skill unavailable: {exc}", "id": skill.id}, ensure_ascii=False)
    if (resource is None and (resolved != skill.path or actual != skill.revision)
            or revision is not None and revision != actual):
        return json.dumps({"error": "Skill revision changed; list skills again before continuing",
                           "id": skill.id, "revision": actual}, ensure_ascii=False)
    end = min(len(text), offset + max_chars)
    result = {**_row(skill, path=True), "root": str(skill.root), "offset": offset,
              "skill_revision": skill.revision, "revision": actual, "resource": resource,
              "resource_path": str(resolved),
              "total_chars": len(text), "content": text[offset:end],
              "next_offset": end if end < len(text) else None}
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    while len(encoded) > MAX_RESPONSE_CHARS and end > offset:
        end = offset + max(0, (end - offset) * 3 // 4)
        result.update(content=text[offset:end], next_offset=end if end < len(text) else None)
        encoded = json.dumps(result, ensure_ascii=False, indent=2)
    return encoded


def _target(base: Path, relative: str) -> Path:
    path = base / relative
    resolved = _canonical(path)
    if not _within(resolved, base) or os.path.normcase(os.path.abspath(path)) != str(resolved):
        raise ValueError(f"Unsafe skill bridge path: {relative}")
    return path


def _atomic_write(path: Path, data: bytes, base: Path) -> None:
    import tempfile
    relative = path.relative_to(base).as_posix()
    _target(base, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    _target(base, relative)
    fd, temporary = tempfile.mkstemp(prefix=".cozter-skill-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _target(base, relative)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sync_bridges(catalog: SkillCatalog) -> SkillCatalog:
    base = _canonical(catalog.workspace_path)
    warnings = list(catalog.warnings)
    try:
        manifest_path = _target(base, ".cozter/skill_bridges.json")
    except ValueError as exc:
        return SkillCatalog(catalog.workspace_path, catalog.entries,
                            (*catalog.warnings, f"Native skill bridge sync skipped: {exc}"))
    with _BRIDGE_LOCK:
        old: dict = {}
        if manifest_path.exists():
            try:
                if manifest_path.stat().st_size > 512 * 1024:
                    raise ValueError("bridge manifest is oversized")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or manifest.get("version") != _MANIFEST_VERSION or not isinstance(manifest.get("files"), dict):
                    raise ValueError("bridge manifest has an invalid format")
                old = manifest["files"]
            except (OSError, ValueError) as exc:
                return SkillCatalog(catalog.workspace_path, catalog.entries,
                                    (*catalog.warnings, f"Native skill bridge sync skipped: {exc}"))
        desired: dict[str, bytes] = {}
        for skill in catalog.entries:
            if skill.source != "workspace":
                continue
            name = f"cozter-{_slug(skill.name)}-{_digest(skill.id.encode())[:10]}"
            content = ("---\nname: " + json.dumps(name)
                       + "\ndescription: " + json.dumps(skill.description, ensure_ascii=False)
                       + "\n---\n" + _BRIDGE_HEADER + "\n\nRead and follow the authoritative skill file at "
                       + json.dumps(str(skill.path), ensure_ascii=False)
                       + ".\nResolve supporting paths relative to " + json.dumps(str(skill.root), ensure_ascii=False)
                       + ".\nSource revision: " + skill.revision
                       + "\nFollow the current user's task and permissions.\n").encode("utf-8")
            for directory in (".agents", ".claude"):
                desired[f"{directory}/skills/{name}/SKILL.md"] = content
        kept: dict[str, str] = {}
        for relative in dict.fromkeys([*desired, *old]):
            try:
                if (not isinstance(relative, str) or not re.fullmatch(
                    r"\.(?:agents|claude)/skills/cozter-[a-z0-9-]+/SKILL\.md", relative,
                )):
                    raise ValueError("manifest contains an invalid bridge path")
                target = _target(base, relative)
                existing = target.read_bytes() if target.exists() and target.stat().st_size <= MAX_SKILL_BYTES else None
                owned = (existing is not None and _is_bridge(existing.decode("utf-8"))
                         and old.get(relative) == _digest(existing))
                if target.exists() and not owned:
                    warnings.append(f"Preserved foreign or modified native skill bridge: {relative}")
                    continue
                desired_content = desired.get(relative)
                if desired_content is None:
                    if owned:
                        target.unlink()
                        if not any(target.parent.iterdir()):
                            target.parent.rmdir()
                else:
                    if existing != desired_content:
                        _atomic_write(target, desired_content, base)
                    kept[relative] = _digest(desired_content)
            except (OSError, UnicodeError, ValueError) as exc:
                warnings.append(f"Native skill bridge skipped for {relative}: {exc}")
        payload = json.dumps({"version": _MANIFEST_VERSION, "files": kept}, indent=2).encode("utf-8")
        if not manifest_path.exists() or manifest_path.read_bytes() != payload:
            _atomic_write(manifest_path, payload, base)
    bounded = [warning[:512] for warning in warnings[:50]]
    if len(warnings) > 50:
        bounded.append(f"{len(warnings) - 50} additional bridge/discovery warnings omitted.")
    return SkillCatalog(catalog.workspace_path, catalog.entries, tuple(bounded))


async def _prepare_catalog(workspace_path: str) -> SkillCatalog:
    """Finish catalog/bridge preparation before releasing its caller's scope."""
    def prepare() -> SkillCatalog:
        workspace.ensure_workspace_state_dir(workspace_path, "skills")
        return _sync_bridges(discover(workspace_path))

    task = asyncio.create_task(asyncio.to_thread(prepare), name="prepare-workspace-skills")
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
        raise


async def refresh(workspace_path: str) -> SkillCatalog:
    """Refresh live sources, replacing only the caller's matching bound catalog."""
    catalog = await _prepare_catalog(workspace_path)
    bound = _CURRENT.get()
    if bound is not None and bound.workspace_path == catalog.workspace_path:
        _CURRENT.set(catalog)
    return catalog


@asynccontextmanager
async def scope(workspace_path: str):
    catalog = await _prepare_catalog(workspace_path)
    token = _CURRENT.set(catalog)
    try:
        yield catalog
    finally:
        _CURRENT.reset(token)
