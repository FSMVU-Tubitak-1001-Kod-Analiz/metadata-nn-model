import pygit2
from pygit2 import Repository
from dataclasses import dataclass
from datetime import datetime
from collections import Counter, defaultdict
from .extractor import process_whole_diff
from .model import model_classify
from .common import CommitInfo


@dataclass
class Contributor:
    name: str = "NULL"
    corrective_count: int = 0
    perfective_count: int = 0
    adaptive_count: int = 0


class RepoClient:
    def __init__(self, location) -> None:
        self._repo = Repository(location)
        self._branch_commits = {}

    def list_all_branches(self) -> list[str]:
        return self._repo.listall_branches()

    def list_all_contributors(self) -> list[str]:
        contributors = set()
        for commit in self._repo.walk(self._repo.head.target, pygit2.GIT_SORT_TIME):
            contributors.add(commit.author.name)
        return list(contributors)

    def is_branch_classified(self, branch) -> bool:
        assert branch in self.list_all_branches()
        commits = self.list_branch_commits(branch)
        return all(not commit.need_clasify for commit in commits)

    def list_branch_commits(self, branch: str = None) -> list[CommitInfo]:
        if branch is None:
            branch = self.list_all_branches()[0]
        assert branch in self.list_all_branches()

        if branch in self._branch_commits:
            return self._branch_commits[branch]

        git_branch = self._repo.lookup_branch(branch)

        retreived_commits = []

        for commit in self._repo.walk(git_branch.target, pygit2.GIT_SORT_TIME):
            if commit.parents:
                diff = self._repo.diff(commit.parents[0], commit)
            else:
                diff = commit.tree.diff_to_tree()

            retreived_commits.append(
                CommitInfo(
                    sha=commit.id.hex,
                    author=commit.author.name,
                    commit_message=commit.message,
                    commit_diff=diff.patch,
                    date=datetime.fromtimestamp(commit.commit_time),
                    filenames=[patch.old_file.path for patch in diff.deltas],
                )
            )
        self._branch_commits[branch] = retreived_commits
        return retreived_commits

    def page_branch_commits(
        self, branch: str = None, start: int = 0, end: int = None
    ) -> list[CommitInfo]:
        commits = self.list_branch_commits(branch)
        end = end or len(commits) + 1
        assert start >= 0 and start <= end
        return commits[start:end]

    def _classify_commits(
        self,
        commits: list[CommitInfo],
        batch_size: int = 24,
        cache: dict[str, str] = None,
    ):
        commit_message = [commit.commit_message for commit in commits]
        commit_feature = [process_whole_diff(commit.commit_diff) for commit in commits]

        split = len(commits) // batch_size + int(len(commits) % batch_size != 0)
        labels = []
        for i in range(split):
            batch_label = model_classify(
                messages=commit_message[i * batch_size : (i + 1) * batch_size],
                numerical_features=commit_feature[
                    i * batch_size : (i + 1) * batch_size
                ],
            )
            labels.extend(batch_label)

        cache = cache or {}
        for commit, label in zip(commits, labels):
            commit.label = label
            cache[commit.sha] = label

    def classify_branch(self, branch: str, cache: dict[str, str] = None):
        assert branch in self.list_all_branches()
        commits = self.list_branch_commits(branch)

        # use cache to speed up classify
        cache = cache or {}
        for commit in commits:
            if not commit.need_clasify:
                continue
            if commit.sha in cache:
                commit.label = cache[commit.sha]

        commits_need_classify = [commit for commit in commits if commit.need_clasify]
        self._classify_commits(commits_need_classify)

    def get_files_sorted_by_label(self, branch: str, label: str) -> list[tuple]:
        assert label in ("Corrective", "Adaptive", "Perfective")
        commits = self.list_branch_commits(branch)
        if any(commit.need_clasify for commit in commits):
            self.classify_branch(branch)
        file_counter = Counter()
        for commit in commits:
            if commit.label == label:
                file_counter += Counter(commit.filenames)
        return file_counter.most_common()

    def get_contributor_sorted_by_label(self, branch: str, label: str) -> list[tuple]:
        assert label in ("Corrective", "Adaptive", "Perfective")
        commits = self.list_branch_commits(branch)
        if any(commit.need_clasify for commit in commits):
            self.classify_branch(branch)
        contributor_counter = Counter()
        for commit in commits:
            if commit.label == label:
                contributor_counter[commit.author] += 1
        return contributor_counter.most_common()

    def get_contributor_with_label_count(self, branch: str) -> list[Contributor]:
        if not self.is_branch_classified(branch):
            self.classify_branch(branch)
        commits = self.list_branch_commits(branch)
        contributor_counter = defaultdict(Contributor)
        for commit in commits:
            if commit.label == "Corrective":
                contributor_counter[commit.author].corrective_count += 1
            if commit.label == "Perfective":
                contributor_counter[commit.author].perfective_count += 1
            if commit.label == "Adaptive":
                contributor_counter[commit.author].adaptive_count += 1

        for name, counter in contributor_counter.items():
            counter.name = name
        contributors = list(contributor_counter.values())
        contributors.sort(
            key=lambda c: c.corrective_count + c.perfective_count + c.adaptive_count,
            reverse=True,
        )
        return contributors
