from dataclasses import dataclass
from github import PaginatedList


@dataclass
class PaginatorCache:
    paginator: PaginatedList
    repo_url: str
    item_counts: int


@dataclass
class CommitInfo:
    sha: str
    author: str
    commit_message: str
    commit_diff: str
    filenames: list[str]
    date: str
    label: str = "NULL"
    url: str = "#"

    @property
    def need_clasify(self):
        return self.label == "NULL"

    @property
    def short_sha(self):
        return self.sha[:7]

    @property
    def date_ymd(self):
        return self.date.strftime("%Y-%m-%d")

    @property
    def commit_message_to_display(self):
        if (pos := self.commit_message.find("\n")) != -1:
            return f"{self.commit_message[:pos]} ..."
        if len(self.commit_message) > 100:
            return f"{self.commit_message[:100]} ..."
        return self.commit_message
