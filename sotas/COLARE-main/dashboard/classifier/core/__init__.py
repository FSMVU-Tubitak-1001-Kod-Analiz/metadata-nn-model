import re
import requests
import github
from github import Github
from functools import lru_cache

from .common import CommitInfo, PaginatorCache
from .extractor import process_whole_diff
from .model import model_classify
from .classify_zip import classify_zip_repo

from django.conf import settings

GITHUB_REPO_REGEX = (
    r"^(https?:\/\/)?(www\.)?github\.com\/([a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+)(\/.*)?$"
)
GITHUB_COMMIT_REGEX = r"^(https?:\/\/)?(www\.)?github\.com\/([a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+)\/commit\/([a-f0-9]{40})(\?.*)?$"

GITHUB_CLIENT = Github(settings.DEFAULT_GH_TOKEN, per_page=100)


@lru_cache(maxsize=256)
def fetch_raw_diff(repo_url: str, sha: str):
    url = f"https://api.github.com/repos/{repo_url}/commits/{sha}"
    return requests.get(
        url,
        headers={"Accept": "application/vnd.github.diff"},
        auth=(settings.DEFAULT_GH_USER, settings.DEFAULT_GH_TOKEN),
    ).text


def fetch_commit_paginator(url: str) -> PaginatorCache:
    if match := re.match(GITHUB_COMMIT_REGEX, url):
        repo_url = match.group(3)
        sha = match.group(4)
        repo = GITHUB_CLIENT.get_repo(match.group(3))
        commit = repo.get_commit(sha)
        return PaginatorCache(commit, repo_url, 1)
    if match := re.match(GITHUB_REPO_REGEX, url):
        repo_url = match.group(3)
        repo = GITHUB_CLIENT.get_repo(repo_url)
        commit_paginator = repo.get_commits()
        return PaginatorCache(commit_paginator, repo_url, commit_paginator.totalCount)
    raise RuntimeError


def simplify_message(message: str):
    if (pos := message.find("\n")) != -1:
        return f"{message[:pos]} ..."
    if len(message) > 100:
        return f"{message[:100]} ..."
    return message


def fetch_commits_by_page(
    commit_paginator,
    repo_url: str,
    page: int,
    num_every_page: int,
    cache: dict[str, str] = None,
):
    cache = cache or {}
    if isinstance(commit_paginator, github.PaginatedList.PaginatedList):
        commits = list(
            commit_paginator[(page - 1) * num_every_page : page * num_every_page]
        )
        commit_sha = [commit.sha for commit in commits]
        commit_message = [commit.commit.message for commit in commits]
        commit_url = [
            f"https://github.com/{repo_url}/commit/{sha}" for sha in commit_sha
        ]
        need_classify = [True for sha in commit_sha if sha not in cache]
        labeld_commits = []
        for sha, message, url, flag in zip(
            commit_sha, commit_message, commit_url, need_classify
        ):
            if not flag:
                label = cache[sha]
            else:
                label = "NULL"
            labeld_commits.append(
                CommitInfo(
                    sha=sha,
                    author=None,
                    commit_message=message,
                    commit_diff=None,
                    filenames=None,
                    date=None,
                    label=label,
                    url=url,
                )
            )

        classify_index = [
            index for index, commit in enumerate(labeld_commits) if commit.need_clasify
        ]
        classify_message = [
            commit.commit_message for commit in labeld_commits if commit.need_clasify
        ]
        classify_diff = [
            fetch_raw_diff(repo_url, commit.sha)
            for commit in labeld_commits
            if commit.need_clasify
        ]
        classify_feature = [process_whole_diff(diff) for diff in classify_diff]

        classified_labels = model_classify(classify_message, classify_feature)
        for index, label in zip(classify_index, classified_labels):
            labeld_commits[index].label = label
            cache[labeld_commits[index].sha] = label
        return labeld_commits
    elif isinstance(commit_paginator, github.Commit.Commit):
        message_page = [commit_paginator.commit.message]
        sha_page = [commit_paginator.sha]
        diff_page = [fetch_raw_diff(repo_url, commit_paginator.sha)]
        url_page = [f"https://github.com/{repo_url}/commit/{commit_paginator.sha}"]
        feature_page = [process_whole_diff(diff) for diff in diff_page]
        label_page = model_classify(message_page, feature_page)
        labeld_commits = [
            CommitInfo(
                sha=sha,
                author=None,
                commit_message=message,
                commit_diff=None,
                filenames=None,
                date=None,
                label=label,
                url=url,
            )
            for label, message, sha, url in zip(
                label_page, message_page, sha_page, url_page
            )
        ]
        return labeld_commits
    else:
        raise ValueError
