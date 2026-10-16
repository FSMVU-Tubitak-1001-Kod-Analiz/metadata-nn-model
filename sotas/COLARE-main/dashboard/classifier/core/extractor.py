from __future__ import annotations
import platform
from functools import lru_cache


SLASH_BASED = ["c", "cpp", "cc", "h", "hh", "java", "go", "js", "rs", "php"]
SHARP_BASED = ["py", "sh", "php", "rb"]
CODE_TYPES = SLASH_BASED + SHARP_BASED


def get_delimeter():
    if platform.system() in ("Linux", "Darwin"):
        return "\n"
    if platform.system in ("Windows"):
        return "\r\n"


class GenericInfo:
    def __init__(self, type: str = None):
        self.changes = {}
        self.type = None


class CodeInfo(GenericInfo):
    def __init__(
        self,
        type: str = None,
        code_added: int = 0,
        code_deleted: int = 0,
        comment_added: int = 0,
        comment_deleted: int = 0,
        space_added: int = 0,
        space_deleted: int = 0,
    ):
        super().__init__(type)
        self.changes["code_added"] = code_added
        self.changes["code_deleted"] = code_deleted
        self.changes["comment_added"] = comment_added
        self.changes["comment_deleted"] = comment_deleted
        self.changes["space_added"] = space_added
        self.changes["space_deleted"] = space_deleted

    def merge_by(self, other: CodeInfo):
        assert isinstance(other, CodeInfo), f"CodeInfo can't merge with {type(other)}"
        for key in self.changes:
            self.changes[key] += other.changes[key]

    def to_dict(self):
        self.changes["code_diff"] = (
            self.changes["code_added"] - self.changes["code_deleted"]
        )
        self.changes["comment_diff"] = (
            self.changes["comment_added"] - self.changes["comment_deleted"]
        )
        self.changes["space_diff"] = (
            self.changes["space_added"] - self.changes["space_deleted"]
        )
        return self.changes


class DocumentInfo(GenericInfo):
    def __init__(self, text_added: int = 0, text_deleted: int = 0):
        super().__init__()
        self.changes["text_added"] = text_added
        self.changes["text_deleted"] = text_deleted

    def merge_by(self, other: DocumentInfo):
        assert isinstance(
            other, DocumentInfo
        ), f"DocumentInfo can't merge with {type(other)}"
        for key in self.changes:
            self.changes[key] += other.changes[key]

    def to_dict(self):
        self.changes["text_diff"] = (
            self.changes["text_added"] - self.changes["text_deleted"]
        )
        return self.changes


class GenericExtractor:
    def extract(self, code: str) -> GenericInfo:
        raise NotImplementedError


class CodeExtractor(GenericExtractor):
    def is_comment_line(self, code_line: str, language: str) -> bool:
        """This is a simple implementation here."""
        if language in SLASH_BASED:
            return (
                code_line.startswith("//")
                or code_line.startswith("*")
                or code_line.startswith("/*")
                or code_line.startswith("/*")
            )
        if language in SHARP_BASED:
            return code_line.startswith("#")
        raise TypeError(f"unknown code type {language}")

    def is_empty_line(self, code_line: str) -> bool:
        for c in code_line:
            if c.isalpha():
                return False
            if c.isdigit():
                return False
        return True

    def extract(self, code: str, language: str) -> GenericInfo:
        code_info_extracted = CodeInfo()
        codes = [code_line.strip() for code_line in code.split(get_delimeter())]
        for code_line in codes:
            if not code_line.startswith("+") and not code_line.startswith("-"):
                continue
            if code_line.startswith("+++") or code_line.startswith("---"):
                continue
            postfix = "_added" if code_line[0] == "+" else "_deleted"
            code_line = code_line[1:].strip()
            if self.is_comment_line(code_line, language):
                code_info_extracted.changes["comment" + postfix] += 1
            elif self.is_empty_line(code_line):
                code_info_extracted.changes["space" + postfix] += 1
            else:
                code_info_extracted.changes["code" + postfix] += 1
        return code_info_extracted


class DocumentExtractor(GenericExtractor):
    def extract(self, code: str) -> GenericInfo:
        codes = [code_line.strip() for code_line in code.split(get_delimeter())]
        text_added, text_deleted = 0, 0
        for code_line in codes:
            if not code_line.startswith("+") and not code_line.startswith("-"):
                continue
            if code_line[0] == "+" and not code_line.startswith("+++"):
                text_added += 1
            if code_line[0] == "-" and not code_line.startswith("---"):
                text_deleted += 1
        return DocumentInfo(text_added, text_deleted)


class Extractor(GenericExtractor):
    def is_test_file(self, start_line: str) -> bool:
        if not self._is_code_file(start_line):
            return False
        start_line = start_line.split()[-1].strip()
        key_words = ["test", "tests", "spec"]
        for key_word in key_words:
            if start_line.find(key_word) != -1:
                return True
        return False

    def _is_code_file(self, start_line: str) -> bool:
        start_line = start_line.split()[-1].strip()
        basename = start_line[start_line.rfind("/") + 1 :]
        if basename.rfind(".") == -1:
            return False
        file_type = basename[basename.rfind(".") + 1 :].lower()
        return file_type in CODE_TYPES

    def get_code_file_type(self, start_line: str) -> str:
        if not self._is_code_file(start_line):
            return None
        start_line = start_line.split()[-1].strip()
        return start_line[start_line.rfind(".") + 1 :].lower()

    def is_code_file(self, start_line: str) -> bool:
        return (
            self._is_code_file(start_line)
            and not self.is_test_file(start_line)
            and not self.is_example_file(start_line)
        )

    def is_document_file(self, start_line: str) -> bool:
        return not self._is_code_file(start_line)

    def is_example_file(self, start_line: str) -> bool:
        if not self._is_code_file(start_line):
            return False
        start_line = start_line.split()[-1].strip()
        key_words = ["example", "examples"]
        for key_word in key_words:
            if start_line.find(key_word) != -1:
                return True
        return False

    def extract(self, code: str) -> GenericInfo:
        start_line = code.split(get_delimeter())[0]
        assert start_line.startswith("diff --git")
        if self.is_document_file(start_line):
            res = DocumentExtractor().extract(code)
            res.type = "document"
            return res
        if self.is_code_file(start_line):
            res = CodeExtractor().extract(code, self.get_code_file_type(start_line))
            res.type = "code"
            return res
        if self.is_test_file(start_line):
            res = CodeExtractor().extract(code, self.get_code_file_type(start_line))
            res.type = "test"
            return res
        if self.is_example_file(start_line):
            res = CodeExtractor().extract(code, self.get_code_file_type(start_line))
            return res
        raise Exception(f"unknown file type {start_line}")


def convert_to_numerical_features(features_dict: dict):
    numerical_features = []
    for _, file_changes in features_dict.items():
        for _, change_value in file_changes.items():
            numerical_features.append(change_value)
    assert len(numerical_features) == 21
    return numerical_features


@lru_cache(maxsize=256)
def process_whole_diff(diff: str):
    file_count = diff.count("diff --git")
    block_index = []
    for _ in range(file_count):
        if not block_index:
            start_index = 0
        else:
            start_index = block_index[-1] + 1
        block_index.append(diff.find("diff --git", start_index))

    file_blocks = []
    for i in range(len(block_index)):
        if i + 1 == len(block_index):
            file_blocks.append(diff[block_index[i] :])
        else:
            file_blocks.append(diff[block_index[i] : block_index[i + 1]])

    file_infos = []
    for block in file_blocks:
        file_infos.append(Extractor().extract(block))

    code_infos = [info for info in file_infos if info.type == "code"]
    test_infos = [info for info in file_infos if info.type == "test"]
    document_infos = [info for info in file_infos if info.type == "document"]

    code_info = CodeInfo()
    for other in code_infos:
        code_info.merge_by(other)

    test_info = CodeInfo()
    for other in test_infos:
        test_info.merge_by(other)

    document_info = DocumentInfo()
    for other in document_infos:
        document_info.merge_by(other)

    return convert_to_numerical_features(
        {
            "code": code_info.to_dict(),
            "test": test_info.to_dict(),
            "document": document_info.to_dict(),
        }
    )
