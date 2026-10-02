"""String-level helpers for patches and Java source (no heavy dependencies)."""

from __future__ import annotations

import re


def extract_code_from_patch(patch: str) -> str:
    """Turns a unified-diff patch into the 'before' version of the code."""
    before_lines = []

    for line in patch.splitlines():
        if line.startswith("@@"):
            # Keep only the trailing code part after the second '@@'
            code_part = line.split("@@", 2)[-1].strip()
            if code_part:
                before_lines.append(code_part)
        elif line.startswith("-"):
            # Removed line: present in the 'before' version
            before_lines.append(line[1:])
        elif line.startswith("+"):
            # Added line: only in the 'after' version
            continue
        else:
            before_lines.append(line)

    return "\n".join(before_lines)


def clean_java_code(java_code: str) -> str:
    """Removes the leading license comment, imports and package statements."""
    if not isinstance(java_code, str):
        return ""  # empty cells / NaN

    # Only the first block comment (the license header) is removed.
    cleaned = re.sub(r"/\*.*?\*/", "", java_code, count=1, flags=re.DOTALL)
    cleaned = re.sub(r"^\s*import\s+.*;", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*package\s+.*;", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
    return cleaned.strip()


def add_sep_to_subset(full_code_with_sep: str, subset_code_without_sep: str) -> str:
    """Adds ``<sep>`` tags to a code subset, based on a fully tagged version.

    Builds a map of which (stripped) lines of the full version carry a tag,
    then tags matching lines of the subset.  Robust to indentation differences
    and to non-contiguous subsets.
    """
    sep_map = {}
    for line in full_code_with_sep.splitlines():
        if "<sep>" in line:
            clean_line = line.replace("<sep>", "").strip()
            if clean_line:
                sep_map[clean_line] = True

    output_lines = []
    for line in subset_code_without_sep.splitlines():
        stripped = line.strip()
        output_lines.append(("<sep>" + stripped) if sep_map.get(stripped) else stripped)

    return "\n".join(output_lines)


_CONTROL_FLOW_RE = re.compile(r"^\s*(if|for|while|switch|try|catch|finally)\s*\(.*\)\s*\{?$")


def add_sep_to_java_statements(java_function_body: str) -> str:
    """Inserts ``<sep>`` after each complete statement (string heuristics only)."""
    code = re.sub(r"/\*.*?\*/", "", java_function_body, flags=re.DOTALL)

    processed = []
    in_control_flow_header = False
    paren_depth = 0

    for line in code.split("\n"):
        if not line.strip():
            processed.append(line)
            continue

        line_without_comments = re.sub(r"//.*$", "", line)
        without_comments = line_without_comments.strip()

        # Multi-line if/for/while headers: wait until the parentheses close.
        if _CONTROL_FLOW_RE.match(without_comments) or in_control_flow_header:
            in_control_flow_header = True
            paren_depth += line_without_comments.count("(") - line_without_comments.count(")")
            if paren_depth <= 0:
                in_control_flow_header = False
                paren_depth = 0
            processed.append(line)
            continue

        if without_comments.endswith(";") or without_comments.endswith("}"):
            processed.append(line + " <sep>")
        else:
            processed.append(line)

    return "\n".join(processed)
