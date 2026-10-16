import os
import hashlib
from pathlib import Path
from github.GithubException import UnknownObjectException
from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from .core import (
    fetch_commit_paginator,
    fetch_commits_by_page,
    classify_zip_repo,
)
from .core.common import PaginatorCache
from .core.exception import IsNotGitZip
from .core.git_client import RepoClient
from collections import defaultdict
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


GLOBAL_SHA_LABEL_CACHE: dict[str, str] = {}
FILE_MD5_RESULT_MAP: dict[str, RepoClient] = {}
QUERY_MD5_PAGINATOR_MAP: dict[str, PaginatorCache] = {}


def get_zip_hash(zipfile: Path, limit: int = 24):
    with open(zipfile, "rb") as file:
        hash_function = hashlib.md5()
        while True:
            data = file.read(4096)
            if not data:
                break
            hash_function.update(data)
    return hash_function.hexdigest()[:limit]


def get_str_hash(query_url: str, limit: int = 24):
    hash_function = hashlib.md5()
    hash_function.update(query_url.encode())
    hash_value = hash_function.hexdigest()
    return hash_value[:limit]


def fetch_commit_page(request, md5):
    if md5 not in QUERY_MD5_PAGINATOR_MAP:
        return render(request, "error.html", {"help_message": "404 NOFOUND:)"})

    paginator_cache = QUERY_MD5_PAGINATOR_MAP.get(md5)

    default_page = 1
    num_every_page = 10
    page = request.GET.get("page", default_page)

    paginator = Paginator(
        [i for i in range(paginator_cache.item_counts)], num_every_page
    )

    try:
        items_page = paginator.page(page)
        page = int(page)
    except PageNotAnInteger:
        items_page = paginator.page(default_page)
        page = default_page
    except EmptyPage:
        items_page = paginator.page(paginator.num_pages)
        page = paginator.num_pages

    try:
        labeled_commits = fetch_commits_by_page(
            paginator_cache.paginator,
            paginator_cache.repo_url,
            page,
            num_every_page,
            GLOBAL_SHA_LABEL_CACHE,
        )
        return render(
            request,
            "commit_list.html",
            {
                "items_page": items_page,
                "labeled_commits": labeled_commits,
                "range_before": range(max(items_page.number - 4, 1), items_page.number),
                "range_after": range(
                    items_page.number + 1,
                    min(items_page.paginator.num_pages + 1, items_page.number + 5),
                ),
            },
        )
    except Exception:
        return render(
            request, "error.html", {"help_message": "Sorry! something went wrong."}
        )


def zip_commit_page_api(request, md5):
    if md5 not in FILE_MD5_RESULT_MAP:
        render(request, "error.html", {"help_message": "404 NO FOUND"})
    repo = FILE_MD5_RESULT_MAP.get(md5)

    default_page = 1
    num_every_page = 10
    page = request.GET.get("page", default_page)

    branch = repo.list_all_branches()[0]
    commits = repo.list_branch_commits(branch)
    paginator = Paginator(commits, num_every_page)

    try:
        items_page = paginator.page(page)
        page = int(page)
    except PageNotAnInteger:
        items_page = paginator.page(default_page)
        page = default_page
    except EmptyPage:
        items_page = paginator.page(paginator.num_pages)
        page = paginator.num_pages

    return render(
        request,
        "commit_list.html",
        {
            "items_page": items_page,
            "labeled_commits": items_page,
            "range_before": range(max(items_page.number - 4, 1), items_page.number),
            "range_after": range(
                items_page.number + 1,
                min(items_page.paginator.num_pages + 1, items_page.number + 5),
            ),
            "chart": True,
        },
    )


def analyse_commit_api(request, md5):
    repo = FILE_MD5_RESULT_MAP.get(md5)
    branch = repo.list_all_branches()[0]
    commits = repo.list_branch_commits(branch)
    # draw overview figure
    corrective_num = sum(1 for commit in commits if commit.label == "Corrective")
    perfective_num = sum(1 for commit in commits if commit.label == "Perfective")
    adaptive_num = sum(1 for commit in commits if commit.label == "Adaptive")

    x = ["Corrective", "Adaptive", "Perfective"]
    y = [corrective_num, adaptive_num, perfective_num]
    fig, ax = plt.subplots()
    ax.bar(
        x,
        y,
        color=[
            "#FA7F6F",
            "#8ECFC9",
            "#82B0D2",
        ],
        ec="black",
        width=0.6,
    )

    save_dir = os.path.join(settings.STATIC_ROOT, md5)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "overview.png"))
    overview_describe = (
        f"We can see from the chart that the project had a total of {len(commits)} code commits, "
        f"of which {corrective_num} were in the Corrective category, {perfective_num} were in the Perfective category, and {adaptive_num} were in the Adaptive category."
    )
    plt.clf()

    # draw trend figure
    corrective_num = defaultdict(int)
    perfective_num = defaultdict(int)
    adaptive_num = defaultdict(int)
    total_num = defaultdict(int)

    date2num = {}
    num2date = {}

    for commit in commits:
        year, month, day = commit.date_ymd.split("-")
        if f"{year}-{month}" not in date2num:
            date2num[f"{year}-{month}"] = int(year) * 12 + int(month)
            num2date[int(year) * 12 + int(month)] = f"{year}-{month}"
        if commit.label == "Corrective":
            corrective_num[f"{year}-{month}"] += 1
        if commit.label == "Perfective":
            perfective_num[f"{year}-{month}"] += 1
        if commit.label == "Adaptive":
            adaptive_num[f"{year}-{month}"] += 1
        total_num[f"{year}-{month}"] += 1

    x = list(sorted(num2date.keys()))
    y_corrective = [corrective_num[num2date[month]] for month in x]
    y_perfective = [perfective_num[num2date[month]] for month in x]
    y_adaptive = [adaptive_num[num2date[month]] for month in x]
    y_total = [total_num[num2date[month]] for month in x]
    start_date, end_date = num2date[x[0]], num2date[x[-1]]
    x = list(range(len(x)))
    fig, ax = plt.subplots()
    ax.plot(x, y_corrective, color="#FA7F6F", label="Corrective")
    ax.plot(x, y_adaptive, color="#8ECFC9", label="Adaptive")
    ax.plot(x, y_perfective, color="#82B0D2", label="Perfective")
    ax.plot(x, y_total, color="#BEB8DC", label="Total")
    ax.legend()
    ax.set_ylabel("commit number")
    ax.set_xlabel("date")
    ax.set_xticks([x[0], x[-1]])
    ax.set_xticklabels([start_date, end_date])
    plt.savefig(os.path.join(save_dir, "trend.png"))
    plt.clf()
    trend_describe = f"The figure shows the changes in the number of different types of commits in this project from {start_date} to {end_date} for each month."
    return render(
        request,
        "analyse.html",
        {
            "contributors": repo.get_contributor_with_label_count(branch),
            "files": repo.get_files_sorted_by_label(branch, "Corrective"),
            "overview": f"{md5}/overview.png",
            "trend": f"{md5}/trend.png",
            "overview_describe": overview_describe,
            "trend_describe": trend_describe,
        },
    )


def upload(request):
    if request.method == "POST" and request.FILES["zip_file"]:
        zip_file = request.FILES["zip_file"]
        fs = FileSystemStorage()
        filename = fs.save(zip_file.name, zip_file)
        filepath = Path(fs.location).joinpath(filename)
        file_hash = get_zip_hash(filepath)

        if file_hash not in FILE_MD5_RESULT_MAP:
            try:
                repo = classify_zip_repo(filepath, GLOBAL_SHA_LABEL_CACHE)
                FILE_MD5_RESULT_MAP[file_hash] = repo
            except IsNotGitZip:
                os.remove(filepath)
                return render(
                    request,
                    "error.html",
                    {
                        "help_message": "The zip file you uploaded should include the .git directory."
                    },
                )
            except Exception:
                os.remove(filepath)
                return render(
                    request,
                    "error.html",
                    {"help_message": "Sorry! something went wrong."},
                )

        os.remove(filepath)
        return redirect(f"/zip/{file_hash}")

    return render(request, "upload.html")


def submit_query(request):
    if request.method == "POST":
        query_url = request.POST.get("query_url")
        hash_id = get_str_hash(query_url)
        if hash_id in QUERY_MD5_PAGINATOR_MAP:
            return redirect(f"/results/{hash_id}")

        try:
            paginator_cache = fetch_commit_paginator(query_url)
            QUERY_MD5_PAGINATOR_MAP[hash_id] = paginator_cache
        except RuntimeError:
            help_message = (
                "You need to enter a correct GitHub repository or commit URL."
            )
            render(request, "error.html", {"help_message": help_message})
        except UnknownObjectException:
            help_message = "Sorry, the repository pointed to by the URL you entered does not exist (possibly due to lack of access permissions from your system's token or because the repo/commit has been deleted)."
            render(request, "error.html", {"help_message": help_message})
        except Exception:
            help_message = "Sorry! something went wrong."
            render(request, "error.html", {"help_message": help_message})
        return redirect(f"/results/{hash_id}")
    else:
        return redirect("/")


def index(request):
    return render(request, "index.html")
