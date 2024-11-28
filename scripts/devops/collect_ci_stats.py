#!/usr/bin/env python3
import datetime
import json
import os

import requests

GITHUB_HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'Bearer {os.environ.get("GITHUB_TOKEN")}',
    'X-GitHub-Api-Version': '2022-11-28',
}

def parse_iso8601(s):
    return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S%z')

def parse_step(step: dict):
    return {
        'name': step['name'],
        'conclusion': step['conclusion'],
        'duration_s': (parse_iso8601(step['completed_at']) - parse_iso8601(step['started_at'])).total_seconds() + 1 if step['conclusion'] else None,
    }

def parse_job(job: dict):
    return {
        'name': job['name'],
        'conclusion': job['conclusion'],
        'duration_s': (parse_iso8601(job['completed_at']) - parse_iso8601(job['started_at'])).total_seconds() + 1 if job['conclusion'] else None,
        'steps': [parse_step(step) for step in job['steps']],
    }

def fetch_jobs(repo: str, run_id: str):
    resp = requests.get(f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs', headers=GITHUB_HEADERS)
    print(json.dumps(resp.json(), indent=2))
    return [parse_job(job) for job in resp.json()['jobs']]

if __name__ == "__main__":
    repo = os.environ.get("GITHUB_REPOSITORY")
    commit = os.environ.get("GITHUB_SHA")
    ref = os.environ.get("GITHUB_REF")
    run_id = os.environ.get("GITHUB_RUN_ID")

    result = {
        'GITHUB_REF': ref,
        'GITHUB_REPOSITORY': repo,
        'GITHUB_RUN_ID': run_id,
        'GITHUB_SHA': commit,
        'jobs': fetch_jobs(repo, run_id),
    }
    print(json.dumps(result, indent=2))
