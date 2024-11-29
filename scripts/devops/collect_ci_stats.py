#!/usr/bin/env python3
import datetime
import json
import os
import pprint
import re
from dataclasses import dataclass
from typing import List

import requests

@dataclass
class OsJobConfig:
    name: str
    matrix: List[str]

KNOWN_OS = {
    'windows-build-test / windows-build-test': OsJobConfig(name='windows', matrix=['config', 'runner', 'full_config_build']),
    'ubuntu-arm64-build-test / ubuntu-arm-build-test': OsJobConfig(name='ubuntu-arm64', matrix=['os', 'config', 'compiler']),
    'ubuntu-x64-build-test / ubuntu-x64-build-test': OsJobConfig(name='ubuntu-x64', matrix=['os', 'config', 'compiler', 'full_config_build']),
    'fedora-build-test / fedora-build-test': OsJobConfig(name='fedora', matrix=['config', 'compiler', 'full_config_build']),
    'emscripten-build-test / emscripten-build': OsJobConfig(name='emscripten', matrix=['config']),
    'macos-build-test / macos-build-test': OsJobConfig(name='macos', matrix=['os', 'compiler']),
}
JOB_NAME_PATTERN = re.compile(r"(?P<name>.+) \((?P<config>.+)\)")

def parse_iso8601(s):
    return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S%z')

def parse_step(step: dict):
    return {
        'name': step['name'],
        'conclusion': step['conclusion'],
        'duration_s': (parse_iso8601(step['completed_at']) - parse_iso8601(step['started_at'])).total_seconds() if step['conclusion'] else None,
    }

def parse_job_name(name: str):
    match = JOB_NAME_PATTERN.match(name)
    job_name, job_config = match.groups()
    os_config = KNOWN_OS[job_name]
    return {
        'os_type': os_config.name,
        'config': dict(zip(os_config.matrix, job_config.split(', '))),
    }

def parse_job(job: dict):
    job_id = job['id']
    runner_stats = {}
    with open(f'RunnerSysStats-{job_id}.json', 'r') as f:
        runner_stats = json.load(f)
    return {
        'name': job['name'],
        **parse_job_name(job['name']),
        'id': job['id'],
        'conclusion': job['conclusion'],
        'duration_s': (parse_iso8601(job['completed_at']) - parse_iso8601(job['started_at'])).total_seconds() if job['conclusion'] else None,
        'steps': [parse_step(step) for step in job['steps']],
        'head_branch': job['head_branch'],
        'head_sha': job['head_sha'],
        'runner_name': job['runner_name'],
        'runner_group_name': job['runner_group_name'],
        **runner_stats,
    }

def parse_jobs(jobs: List[dict]):
    return [
        parse_job(job)
        for job in jobs
        if any(
            job['name'].startswith(job_prefix)
            for job_prefix in KNOWN_OS.keys()
        )
    ]

def fetch_jobs(repo: str, run_id: str):
    return requests.get(f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs', headers={
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'Bearer {os.environ.get("GITHUB_TOKEN")}',
        'X-GitHub-Api-Version': '2022-11-28',
    })

if __name__ == "__main__":
    repo = os.environ.get("GITHUB_REPOSITORY")
    ref = os.environ.get("GITHUB_REF")
    run_id = os.environ.get("GITHUB_RUN_ID")

    resp = fetch_jobs(repo, run_id)

    result = {
        'GITHUB_REF': ref,
        'GITHUB_REPOSITORY': repo,
        'GITHUB_RUN_ID': run_id,
        'jobs': parse_jobs(resp.json()['jobs']),
    }
    pprint.pp(result, indent=2, width=120)
