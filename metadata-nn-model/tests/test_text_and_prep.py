import pandas as pd

from difflabel.data.java_text import (
    add_sep_to_java_statements, add_sep_to_subset, clean_java_code, extract_code_from_patch,
)
from difflabel.data.preparation import (
    filter_for_equal_label_count, filter_top_repos_equal_representation, standardize_labels,
)


def test_extract_code_from_patch_keeps_before_side():
    patch = "@@ -1,3 +1,3 @@ void f() {\n-old\n+new\n same"
    assert extract_code_from_patch(patch) == "void f() {\nold\n same"


def test_clean_java_code_removes_license_imports_package():
    code = "/* license */\npackage a.b;\nimport x.Y;\n\nclass A {}\n"
    assert clean_java_code(code) == "class A {}"
    assert clean_java_code(float("nan")) == ""


def test_clean_java_code_only_first_block_comment():
    assert clean_java_code("/* a */ int x; /* b */") == "int x; /* b */"


def test_add_sep_helpers():
    assert add_sep_to_java_statements("int a = 1;\nif (x) {\n  b();\n}") == "int a = 1; <sep>\nif (x) {\n  b(); <sep>\n} <sep>"
    assert add_sep_to_subset("a(); <sep>\nb();", "  a();\n  b();") == "<sep>a();\nb();"


def test_standardize_labels_is_case_insensitive_and_drops_nonmatches():
    df = pd.DataFrame({"labels": ["Bug", "kind/enhancement", "docs", "performance-issue"]})
    out = standardize_labels(df, ["bug", "enhancement", "performance"])
    assert out["labels"].tolist() == ["bug", "enhancement", "performance"]


def test_equal_label_count_balances():
    df = pd.DataFrame({"labels": ["bug"] * 5 + ["enhancement"] * 3})
    out = filter_for_equal_label_count(df, ["enhancement", "bug"], seed=1)
    assert out["labels"].value_counts().to_dict() == {"enhancement": 3, "bug": 3}


def test_top_repos_keeps_repo_column_and_drops_small_repos():
    rows = ([("big", "bug")] * 6 + [("big", "enhancement")] * 6
            + [("small", "bug")] * 6 + [("small", "enhancement")] * 1)
    df = pd.DataFrame(rows, columns=["full_repo_name", "labels"])
    out = filter_top_repos_equal_representation(df, ["enhancement", "bug"], n=8)
    assert set(out["full_repo_name"]) == {"big"}          # `small` can't supply 4 enhancement rows
    assert len(out) == 8 and out["labels"].value_counts().to_dict() == {"enhancement": 4, "bug": 4}
