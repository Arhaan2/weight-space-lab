#!/usr/bin/env python3
"""Publish the reviewed project with the maintainer's GitHub identity.

Requires Git and the authenticated GitHub CLI on the machine running it.
Does not stage or commit unreviewed files automatically.
"""

import subprocess
from pathlib import Path


def run(*args, capture=False):
    return subprocess.run(args, cwd=Path(__file__).resolve().parents[1], check=True,
                          text=True, capture_output=capture)


def main():
    expected_email = "59648536+Arhaan2@users.noreply.github.com"
    login = run("gh", "api", "user", "--jq", ".login", capture=True).stdout.strip()
    if login != "Arhaan2":
        raise SystemExit("The authenticated account must be Arhaan2.")
    if run("git", "status", "--porcelain", capture=True).stdout.strip():
        raise SystemExit("Review and commit the working tree before publishing.")
    history = run("git", "log", "--format=%ae|%ce|%B%x00", capture=True).stdout
    for block in history.split("\x00"):
        if not block.strip():
            continue
        author, committer, message = block.strip().split("|", 2)
        if author != expected_email or committer != expected_email:
            raise SystemExit("Project history contains an unexpected commit identity.")
        if "co-authored-by:" in message.lower():
            raise SystemExit("Review the coauthor attribution in project history.")
    if run("git", "branch", "--show-current", capture=True).stdout.strip() != "main":
        raise SystemExit("Publish the initial project from main.")
    remotes = run("git", "remote", capture=True).stdout.split()
    if "origin" in remotes:
        raise SystemExit("An origin already exists; inspect it and push explicitly.")
    run("gh", "repo", "create", "Arhaan2/weight-space-lab", "--public",
        "--description", "Controlled experiments in direct model-weight editing and independent evaluation",
        "--source=.", "--remote=origin", "--push")


if __name__ == "__main__":
    main()
