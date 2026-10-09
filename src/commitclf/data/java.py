"""String processing for Java code and unified-diff patches."""
import re


def _patch_side(patch: str, keep: str, drop: str) -> str:
    """One side of a unified diff: context lines plus the `keep` ('-' or '+') lines.

    The text after the second '@@' on a hunk header (usually the enclosing
    method signature) is kept as a line too.
    """
    lines = []
    for line in str(patch).splitlines():
        if line.startswith("@@"):
            code = line.split("@@", 2)[-1].strip()
            if code:
                lines.append(code)
        elif line.startswith(keep):
            lines.append(line[1:])
        elif line.startswith(drop) or line.startswith("\\"):   # "\ No newline at end of file"
            continue
        else:
            lines.append(line[1:] if line.startswith(" ") else line)
    return "\n".join(lines)


def patch_before(patch: str) -> str:
    """Pre-change side of a diff: context + removed lines."""
    return _patch_side(patch, keep="-", drop="+")


def patch_after(patch: str) -> str:
    """Post-change side of a diff: context + added lines."""
    return _patch_side(patch, keep="+", drop="-")


_LICENSE_WORDS = re.compile(r"licen[cs]e|copyright|spdx|\(c\)|all rights reserved", flags=re.IGNORECASE)
_LEADING_BLOCK = re.compile(r"\s*/\*.*?\*/", flags=re.DOTALL)
_LEADING_LINES = re.compile(r"(?:\s*//[^\n]*(?:\n|$))+")
_IMPORT = re.compile(r"^\s*import\s+.*;", flags=re.MULTILINE)
_PACKAGE = re.compile(r"^\s*package\s+.*;", flags=re.MULTILINE)
_BLANK_LINES = re.compile(r"\n\s*\n")


def strip_license_header(code: str) -> str:
    """Remove comments at the very top of the text that look like a license header.

    Only leading comments are considered (block comments or runs of // lines), and only
    those mentioning license / copyright / SPDX / "all rights reserved". A leading Javadoc or
    any other comment is kept, and so is everything that follows it.
    """
    while True:
        m = _LEADING_BLOCK.match(code) or _LEADING_LINES.match(code)
        if not m or not m.group(0).strip() or not _LICENSE_WORDS.search(m.group(0)):
            return code
        code = code[m.end():]


def clean_java(code) -> str:
    """Strip the license header, imports and the package line.

    Other comments are left alone on purpose. Non-strings become "".
    """
    if not isinstance(code, str):
        return ""
    code = strip_license_header(code)
    code = _IMPORT.sub("", code)
    code = _PACKAGE.sub("", code)
    code = _BLANK_LINES.sub("\n", code)
    return code.strip()
