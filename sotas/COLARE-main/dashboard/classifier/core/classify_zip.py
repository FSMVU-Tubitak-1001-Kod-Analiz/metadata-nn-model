import os
import zipfile
import random
import string
from pathlib import Path
from django.conf import settings
from .exception import IsNotGitZip
from .git_client import RepoClient


def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for _ in range(length))


def classify_zip_repo(zip_file: str, cache: dict[str, str]) -> RepoClient:
    extract_location = os.path.join(
        settings.RECEIVE_ZIP_PATH, generate_random_string(48)
    )
    extract_location = Path(extract_location)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_location)

    if not extract_location.joinpath(".git").exists():
        repo_location = None
        subdirectories = []
        for root, dirs, files in os.walk(extract_location):
            for dir in dirs:
                subdirectories.append(Path(os.path.join(root, dir)))
        for subdir in subdirectories:
            if subdir.joinpath(".git").exists():
                repo_location = subdir
                break
        if repo_location is None:
            return IsNotGitZip
    else:
        repo_location = extract_location

    repo = RepoClient(repo_location)
    branch = repo.list_all_branches()[0]
    repo.classify_branch(branch, cache)
    return repo
