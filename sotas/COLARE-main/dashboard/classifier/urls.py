from django.urls import path, re_path
from . import views

urlpatterns = [
    # re_path(r'^commit/(?P<commit_hash>[0-9a-fA-F]{64})/$', views.commit_detail, name='commit_detail'),
    re_path(
        r"^results/(?P<md5>[0-9a-fA-F]{24})/$",
        views.fetch_commit_page,
        name="fetch_commit_page",
    ),
    re_path(
        r"^zip/(?P<md5>[0-9a-fA-F]{24})/$",
        views.zip_commit_page_api,
        name="zip_commit_page_api",
    ),
    re_path(
        r"^zip/(?P<md5>[0-9a-fA-F]{24})/analyse$",
        views.analyse_commit_api,
        name="analyse_commit_api",
    ),
    path("search/", views.submit_query, name="submit_query"),
    path("upload/", views.upload),
    path("", views.index),
]
