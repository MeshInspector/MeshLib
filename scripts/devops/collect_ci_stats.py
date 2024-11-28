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

def parse_step(step: dict):
    def parse_datetime(s):
        return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f%z')
    return {
        'name': step['name'],
        'status': step['status'],
        'duration_s': (parse_datetime(step['completed_at']) - parse_datetime(step['started_at'])).total_seconds(),
    }

def parse_job(job: dict):
    def parse_datetime(s):
        return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')
    return {
        'name': job['name'],
        'status': job['status'],
        'duration_s': (parse_datetime(job['completed_at']) - parse_datetime(job['started_at'])).total_seconds(),
        'steps': [parse_step(step) for step in job['steps']],
    }

def fetch_jobs(repo: str, run_id: str):
    resp = requests.get(f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs', headers=GITHUB_HEADERS)
    print(json.dumps(resp.json(), indent=2))
    return [parse_job(job) for job in resp.json()['jobs']]

if __name__ == "__main__":
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")

    result = {
        'GITHUB_REPOSITORY': repo,
        'GITHUB_RUN_ID': run_id,
        'jobs': fetch_jobs(repo, run_id),
    }
    print(json.dumps(result, indent=2))
