"""Metadata written as text, for the language model to read next to (or instead of) the code.

`python -m commitclf metatext` streams the source CSV once and writes <processed_dir>/metatext.csv,
one row per file row of rows.csv (same order) with four texts:

    file_short / file_long      the change line + this file's line (per-file encoder input)
    change_short / change_long  the change line + summaries over all its files (one per change,
                                repeated on every file row; used as change-level metadata)

Short = compact key=value style, long = plain sentences. Both carry the same facts:

  change  number of Java files / all files, commits, lines added / deleted, deletion share, test
          files, file statuses, how long the PR (or issue) was open, the file names in the change
  file    file name (no path), rename source name, status, lines added / deleted, path flags as
          words (main code, test, config ...), type kind (class / interface / enum ...), file length
          before -> after, hunks, changed methods (hunk headers), added / removed method
          declarations, edit actions counted on added and removed lines (if, null check, try,
          catch, throw, log call, @Test ...), imports added / removed (the repo's own packages left out)

Left out on purpose (leak the label or name the repo): repo, org, path, author, messages, PR title,
labels, issue / JIRA fields, SHAs, URLs, unit_type, diff_method.

Embedding inputs use the texts with `text: <column>` (see embeddings.py):
    meta_short: {model: codebert, text: file_short}                       # the text alone
    patch_meta_short: {model: codebert, text: file_short, columns: [patch]}  # text + patch, one input
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from commitclf.data.prepare import KEY_COLS, load_rows, load_units, processed_path, read_source, row_key

TEXT_COLUMNS = ("file_short", "file_long", "change_short", "change_long")
SOURCE_COLS = ["file_name", "old_file_path", "status", "additions", "deletions", "is_test_file", "content_omitted",
               "n_files_in_pr", "created_at", "closed_at", "merged_at", "patch", "content_before", "content_after"]
MAX_FILES, MAX_METHODS, MAX_IMPORTS = 10, 6, 4

# ---------------------------------------------------------------------------------------------
# Facts from the code

# (short name, long name, pattern) counted on the added and on the removed lines of the diff
ACTIONS = [
    ("if", "if statement", r"\bif\s*\("),
    ("else", "else branch", r"\belse\b"),
    ("null", "null check", r"[!=]=\s*null\b|\bnull\s*[!=]=|\bObjects\.(?:requireNonNull|isNull|nonNull)\b"),
    ("optional", "Optional use", r"\bOptional\b"),
    ("try", "try block", r"\btry\s*[({]"),
    ("catch", "catch clause", r"\bcatch\s*\("),
    ("finally", "finally block", r"\bfinally\b"),
    ("throw", "throw statement", r"\bthrow\s"),
    ("throws", "throws clause", r"\bthrows\s"),
    ("return", "return statement", r"\breturn\b"),
    ("loop", "loop", r"\b(?:for|while)\s*\(|\bdo\s*\{"),
    ("switch", "switch or case", r"\bswitch\s*\(|\bcase\s"),
    ("log", "log call", r"\b(?:log|logger|LOG|LOGGER|Log|Logger)\.(?:trace|debug|info|warn|warning|error|fatal)\s*\("),
    ("assert", "assertion", r"\bassert\w*\s*\(|\bassert\s"),
    ("sync", "synchronization", r"\bsynchronized\b|\.(?:lock|unlock)\s*\("),
    ("new", "object creation", r"\bnew\s+[A-Z]\w*"),
    ("@Override", "@Override", r"@Override\b"),
    ("@Test", "@Test", r"@(?:Test|ParameterizedTest)\b"),
    ("@Deprecated", "@Deprecated", r"@Deprecated\b"),
]
_ACTIONS = [(s, l, re.compile(p)) for s, l, p in ACTIONS]
_TODO = re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b")
_MODIFIERS = r"(?:(?:public|protected|private|static|final|abstract|synchronized|native|default|strictfp)\s+)"
_METHOD = re.compile(r"^(?:@\w+(?:\([^)]*\))?\s+)*" + _MODIFIERS + r"*(?:<[^>]+>\s+)?([\w.$<>\[\]?,]+(?:\s*<[^>]*>)?)\s+(\w+)\s*\([^;{}]*\)?\s*(?:throws\s+[\w.$, ]+)?\s*\{?\s*$")
_CTOR = re.compile(r"^(?:@\w+(?:\([^)]*\))?\s+)*" + _MODIFIERS + r"+([A-Z]\w*)\s*\([^;{}]*\)?\s*(?:throws\s+[\w.$, ]+)?\s*\{?\s*$")
_NOT_TYPE = {"return", "new", "throw", "else", "case", "if", "for", "while", "switch", "catch", "try", "do", "yield"}
_NOT_NAME = {"if", "for", "while", "switch", "catch", "synchronized", "return", "new", "super", "this", "try"}
_HUNK_NAME = re.compile(r"(\w+)\s*\(")
_HUNK_TYPE = re.compile(r"\b(?:class|interface|enum|record)\s+(\w+)")
_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+?)(?:\.\*)?\s*;")
_PACKAGE = re.compile(r"^[+\- ]?\s*package\s+([\w.]+)\s*;", re.MULTILINE)   # file content or diff lines
_KIND = re.compile(r"^[ \t]*(?:@[\w.]+(?:\([^)]*\))?\s*)*((?:(?:public|protected|private|abstract|final|static|sealed|non-sealed|strictfp)\s+)*)"
                   r"(class|interface|enum|record|@interface)\s+\w+", re.MULTILINE)
KIND_WORDS = {"class": "class", "interface": "interface", "enum": "enum", "record": "record", "@interface": "annotation"}


def method_name(line: str) -> str | None:
    """Name of a method / constructor declared on this line (approximate, one-line signatures)."""
    m = _METHOD.match(line)
    if m and m.group(1) not in _NOT_TYPE and m.group(2) not in _NOT_NAME:
        return m.group(2)
    m = _CTOR.match(line)
    return m.group(1) if m else None


def type_kind(code: str) -> str | None:
    m = _KIND.search(code or "")
    if not m:
        return None
    kind = KIND_WORDS[m.group(2)]
    return f"abstract {kind}" if "abstract" in m.group(1) and kind == "class" else kind


def own_prefix(code: str) -> str | None:
    """The first three parts of the file's package (e.g. org.apache.kafka): imports from it name the repo."""
    m = _PACKAGE.search(code or "")
    return ".".join(m.group(1).split(".")[:3]) if m else None


def n_lines(code) -> int:
    return code.count("\n") + 1 if isinstance(code, str) and code else 0


def diff_facts(patch: str, prefix: str | None) -> dict:
    """Hunks, changed methods, declared methods, edit actions and imports of one file's diff."""
    hunks, changed, plus_m, minus_m = 0, [], [], []
    acts = {"+": Counter(), "-": Counter()}
    imports = {"+": [], "-": []}
    for line in str(patch or "").splitlines():
        if line.startswith("@@"):
            hunks += 1
            ctx = line.split("@@", 2)[-1].strip()
            m = _HUNK_NAME.findall(ctx) or _HUNK_TYPE.findall(ctx)
            if m and m[-1] not in changed and m[-1] not in _NOT_NAME:
                changed.append(m[-1])
            continue
        sign = line[:1]
        if sign not in acts or line.startswith(("+++", "---")):
            continue
        code = line[1:].strip()
        if not code:
            continue
        c = acts[sign]
        if code.startswith(("//", "/*", "*")):
            c["comment"] += 1
            c["todo"] += bool(_TODO.search(code))
            continue
        m = _IMPORT.match(code)
        if m:
            name = m.group(1)
            if not (prefix and name.startswith(prefix)):
                imports[sign].append(name)
            continue
        name = method_name(code)
        if name:
            (plus_m if sign == "+" else minus_m).append(name)
        for short, _, rx in _ACTIONS:
            n = len(rx.findall(code))
            if n:
                c[short] += n
    added = [m for m in dict.fromkeys(plus_m) if m not in minus_m]
    removed = [m for m in dict.fromkeys(minus_m) if m not in plus_m]
    return {"hunks": hunks, "changed": changed, "added_methods": added, "removed_methods": removed,
            "plus": acts["+"], "minus": acts["-"],
            "imports_added": [i for i in dict.fromkeys(imports["+"]) if i not in imports["-"]],
            "imports_removed": [i for i in dict.fromkeys(imports["-"]) if i not in imports["+"]]}


def file_info(r: pd.Series) -> dict:
    before, after = r["content_before"], r["content_after"]
    before = before if isinstance(before, str) else ""
    after = after if isinstance(after, str) else ""
    info = diff_facts(r["patch"], own_prefix(after) or own_prefix(before) or own_prefix(str(r["patch"] or "")))
    old = r["old_file_path"] if isinstance(r["old_file_path"], str) else ""
    old_name = old.replace("\\", "/").rsplit("/", 1)[-1]
    name = str(r["file_name"]) if isinstance(r["file_name"], str) else ""
    info.update(name=name, old_name=old_name if old_name and old_name != name else "", status=str(r["status"]),
                add=int(r["additions"]), dele=int(r["deletions"]), test=bool(r["is_test_file"]),
                omitted=bool(r["content_omitted"]), kind=type_kind(after) or type_kind(before),
                lines_before=n_lines(before), lines_after=n_lines(after), files_in_pr=r["n_files_in_pr"],
                created=r["created_at"], closed=r["closed_at"], merged=r["merged_at"])
    return info


# ---------------------------------------------------------------------------------------------
# Text

FLAG_WORDS = [("file__path_src_main", "main code"), ("file__path_src_test", "test folder"),
              ("file__path_examples_docs", "examples or docs"), ("file__path_name_exception", "exception class"),
              ("file__path_name_util", "utility class"), ("file__path_name_config", "configuration class"),
              ("file__path_name_impl", "implementation class"), ("file__path_package_info", "package-info")]
ACTION_LONG = {s: l for s, l, _ in ACTIONS} | {"comment": "comment line", "todo": "TODO/FIXME"}
ACTION_ORDER = [s for s, _, _ in ACTIONS] + ["comment", "todo"]


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}es" if word.endswith(("ch", "sh", "s", "x")) else f"{n} {word}s"


def cap(items: list[str], k: int, sep: str = " ") -> str:
    more = f"{sep}+{len(items) - k} more" if len(items) > k else ""
    return sep.join(items[:k]) + more


def duration(start, end) -> tuple[str, str] | None:
    """(short, long) of end - start, or None when unknown."""
    try:
        s, e = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    except (ValueError, TypeError):
        return None
    if pd.isna(s) or pd.isna(e) or e < s:
        return None
    hours = (e - s).total_seconds() / 3600
    if hours < 24:
        h = max(1, round(hours))
        return f"{h}h", plural(h, "hour")
    d = round(hours / 24)
    return f"{d}d", plural(d, "day")


def actions_short(plus: Counter, minus: Counter) -> str:
    parts = []
    for a in ACTION_ORDER:
        p, m = plus.get(a, 0), minus.get(a, 0)
        if p or m:
            parts.append(a + (f"+{p}" if p else "") + (f"-{m}" if m else ""))
    return " ".join(parts)


def actions_long(counter: Counter, where: str) -> str:
    items = [plural(counter[a], ACTION_LONG[a]) for a in ACTION_ORDER if counter.get(a)]
    if not items:
        return ""
    return f"The {where} lines contain " + (", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0]) + "."


def change_text(files: list[dict], n_commits: int) -> tuple[str, str]:
    n = len(files)
    add, dele = sum(f["add"] for f in files), sum(f["dele"] for f in files)
    share = round(100 * dele / max(add + dele, 1))
    tests = sum(f["test"] for f in files)
    st = Counter(f["status"] for f in files)
    total = files[0]["files_in_pr"]
    total = int(total) if pd.notna(total) and total else n
    f0 = files[0]
    dur = duration(f0["created"], f0["merged"] if isinstance(f0["merged"], str) and f0["merged"] else f0["closed"])
    names = [f["name"] for f in files]
    short = (f"change: {n} java of {total} files, {plural(int(n_commits), 'commit')}, +{add} -{dele} ({share}% del), "
             f"{plural(tests, 'test')}, "
             f"{st['modified']} mod {st['added']} new {st['removed']} rm {st['renamed']} ren"
             + (f", open {dur[0]}" if dur else "") + f" | files: {cap(names, MAX_FILES)}")
    long = (f"The change touches {plural(n, 'Java file')} out of {plural(total, 'file')} in total, in "
            f"{plural(int(n_commits), 'commit')}, with {plural(add, 'line')} added and {dele} deleted ({share}% deletions). "
            f"{tests} of the Java files {'is a test file' if tests == 1 else 'are test files'}. "
            f"{st['modified']} modified, {st['added']} added, {st['removed']} removed and {st['renamed']} renamed. "
            + (f"It was open for {dur[1]}. " if dur else "")
            + f"Files in the change: {cap(names, MAX_FILES, ', ')}.")
    return short, long


def file_text(f: dict, flags: list[str]) -> tuple[str, str]:
    lines = f"{f['lines_before']}->{f['lines_after']} lines" if not f["omitted"] else "content omitted (large file)"
    s = [f"file: {f['name']}" + (f" (from {f['old_name']})" if f["old_name"] else ""),
         f"{f['status']} +{f['add']} -{f['dele']}" + (" test" if f["test"] else "")]
    if flags:
        s.append(", ".join(flags))
    s.append(", ".join(x for x in (f["kind"], lines, plural(f["hunks"], "hunk")) if x))
    if f["changed"]:
        s.append("in: " + cap(f["changed"], MAX_METHODS))
    if f["added_methods"]:
        s.append("+methods: " + cap(f["added_methods"], MAX_METHODS))
    if f["removed_methods"]:
        s.append("-methods: " + cap(f["removed_methods"], MAX_METHODS))
    act = actions_short(f["plus"], f["minus"])
    if act:
        s.append("actions: " + act)
    if f["imports_added"]:
        s.append("+imports: " + cap(f["imports_added"], MAX_IMPORTS))
    if f["imports_removed"]:
        s.append("-imports: " + cap(f["imports_removed"], MAX_IMPORTS))
    short = " | ".join(s)

    verb = {"modified": "was modified", "added": "was added", "removed": "was removed", "renamed": "was renamed"}
    l = [f"The file {f['name']} {verb.get(f['status'], 'was ' + f['status'])}"
         + (f" (renamed from {f['old_name']})" if f["old_name"] else "")
         + f": {plural(f['add'], 'line')} added and {f['dele']} deleted."]
    what = (["a test file"] if f["test"] else []) + flags
    if what:
        l.append("It is " + ", ".join(what) + ".")
    if f["omitted"]:
        l.append("Its content is omitted (large file).")
    else:
        l.append(f"It is {'an' if (f['kind'] or 'x')[0] in 'aeiou' else 'a'} {f['kind'] or 'Java file'} with "
                 f"{f['lines_before']} lines before and {f['lines_after']} after the change, edited in {plural(f['hunks'], 'hunk')}.")
    if f["changed"]:
        l.append("Changed methods: " + cap(f["changed"], MAX_METHODS, ", ") + ".")
    if f["added_methods"]:
        l.append("Added methods: " + cap(f["added_methods"], MAX_METHODS, ", ") + ".")
    if f["removed_methods"]:
        l.append("Removed methods: " + cap(f["removed_methods"], MAX_METHODS, ", ") + ".")
    l += [x for x in (actions_long(f["plus"], "added"), actions_long(f["minus"], "removed")) if x]
    if f["imports_added"]:
        l.append("Added imports: " + cap(f["imports_added"], MAX_IMPORTS, ", ") + ".")
    if f["imports_removed"]:
        l.append("Removed imports: " + cap(f["imports_removed"], MAX_IMPORTS, ", ") + ".")
    return short, " ".join(l)


def change_summary(files: list[dict]) -> tuple[str, str]:
    """Edit actions, changed methods and imports over all files of a change (for the change text)."""
    plus, minus = Counter(), Counter()
    for f in files:
        plus.update(f["plus"])
        minus.update(f["minus"])
    def union(key):
        return list(dict.fromkeys(x for f in files for x in f[key]))
    changed, added, removed = union("changed"), union("added_methods"), union("removed_methods")
    imp_a, imp_r = union("imports_added"), union("imports_removed")
    kinds = Counter(f["kind"] for f in files if f["kind"])
    s = []
    if kinds:
        s.append("kinds: " + " ".join(f"{k}={v}" for k, v in kinds.most_common()))
    if changed:
        s.append("in: " + cap(changed, MAX_METHODS))
    if added:
        s.append("+methods: " + cap(added, MAX_METHODS))
    if removed:
        s.append("-methods: " + cap(removed, MAX_METHODS))
    act = actions_short(plus, minus)
    if act:
        s.append("actions: " + act)
    if imp_a:
        s.append("+imports: " + cap(imp_a, MAX_IMPORTS))
    if imp_r:
        s.append("-imports: " + cap(imp_r, MAX_IMPORTS))
    l = []
    if kinds:
        l.append("The files are " + ", ".join(plural(v, k) for k, v in kinds.most_common()) + ".")
    if changed:
        l.append("Changed methods: " + cap(changed, MAX_METHODS, ", ") + ".")
    if added:
        l.append("Added methods: " + cap(added, MAX_METHODS, ", ") + ".")
    if removed:
        l.append("Removed methods: " + cap(removed, MAX_METHODS, ", ") + ".")
    l += [x for x in (actions_long(plus, "added"), actions_long(minus, "removed")) if x]
    if imp_a:
        l.append("Added imports: " + cap(imp_a, MAX_IMPORTS, ", ") + ".")
    if imp_r:
        l.append("Removed imports: " + cap(imp_r, MAX_IMPORTS, ", ") + ".")
    return " | ".join(s), " ".join(l)


# ---------------------------------------------------------------------------------------------

def metatext_path(cfg: dict):
    return processed_path(cfg, "metatext.csv")


def run(cfg: dict, show: int = 3) -> pd.DataFrame:
    rows = load_rows(cfg)
    index = pd.Series(np.arange(len(rows)), index=rows["row_key"])
    info: list[dict | None] = [None] * len(rows)
    for chunk in tqdm(read_source(cfg, sorted(set(KEY_COLS) | set(SOURCE_COLS))), desc="chunks"):
        chunk = chunk.reset_index(drop=True)
        for c in ("additions", "deletions", "is_test_file", "content_omitted"):
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").fillna(0)
        chunk["n_files_in_pr"] = pd.to_numeric(chunk["n_files_in_pr"], errors="coerce")
        idx = row_key(chunk).map(index)
        for i, r in chunk[idx.notna().to_numpy()].iterrows():
            info[int(idx[i])] = file_info(r)
    missing = sum(x is None for x in info)
    if missing:
        raise ValueError(f"{missing} of {len(rows)} rows of rows.csv were not found in the source")

    commits = load_units(cfg).set_index("unit_key")["n_commits"]
    flags = rows[[c for c, _ in FLAG_WORDS]].to_numpy() > 0
    out = pd.DataFrame({"row_key": rows["row_key"]})
    for c in TEXT_COLUMNS:
        out[c] = ""
    cols = {c: out.columns.get_loc(c) for c in TEXT_COLUMNS}
    for unit, pos in tqdm(pd.Series(np.arange(len(rows))).groupby(rows["unit_key"].to_numpy()), desc="changes"):
        pos = pos.to_numpy()
        files = [info[p] for p in pos]
        c_short, c_long = change_text(files, int(commits.get(unit, 1)))
        s_short, s_long = change_summary(files)
        out.iloc[pos, cols["change_short"]] = c_short + (" | " + s_short if s_short else "")
        out.iloc[pos, cols["change_long"]] = c_long + (" " + s_long if s_long else "")
        for p, f in zip(pos, files):
            words = [w for (_, w), on in zip(FLAG_WORDS, flags[p]) if on]
            f_short, f_long = file_text(f, words)
            out.iat[p, cols["file_short"]] = c_short + "\n" + f_short
            out.iat[p, cols["file_long"]] = c_long + "\n" + f_long
    out.to_csv(metatext_path(cfg), index=False)
    print(f"Wrote {metatext_path(cfg)} ({len(out)} rows)")
    lengths = {c: out[c].str.len() for c in TEXT_COLUMNS}
    print("characters (median / p90 / max): " + ", ".join(
        f"{c} {int(v.median())}/{int(v.quantile(0.9))}/{int(v.max())}" for c, v in lengths.items()))
    for i in np.random.default_rng(cfg["seed"]).choice(len(out), min(show, len(out)), replace=False):
        print(f"\n--- {out.at[i, 'row_key'].split('|')[1]} (row {i})")
        for c in TEXT_COLUMNS:
            print(f"[{c}]\n{out.at[i, c]}")
    return out


def load_metatext(cfg: dict) -> pd.DataFrame:
    path = metatext_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `metatext` first")
    return pd.read_csv(path, keep_default_na=False, dtype=str).set_index("row_key")
