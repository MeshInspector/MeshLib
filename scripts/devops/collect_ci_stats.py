#!/usr/bin/env python3
import datetime
import json
import os
import pprint
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests

@dataclass
class OsJobConfig:
    name: str
    matrix: List[str]

KNOWN_OS = {
    'windows-build-test / windows-build-test': OsJobConfig(name='windows', matrix=['config', 'runner', 'full_config_build']),
    'ubuntu-arm64-build-test / ubuntu-arm-build-test': OsJobConfig(name='ubuntu-arm64', matrix=['os', 'config', 'compiler']),
    'ubuntu-x64-build-test / ubuntu-x64-build-test': OsJobConfig(name='ubuntu-x64', matrix=['os', 'config', 'compiler', 'cxx-compiler', 'c-compiler', 'cxx-standard', 'build_mrcuda']),
    'fedora-build-test / fedora-build-test': OsJobConfig(name='fedora', matrix=['config', 'compiler', 'full_config_build']),
    'emscripten-build-test / emscripten-build': OsJobConfig(name='emscripten', matrix=['config']),
    'macos-build-test / macos-build-test': OsJobConfig(name='macos', matrix=['os', 'compiler']),
}
JOB_NAME_PATTERN = re.compile(r"(?P<name>.+) \((?P<config>.+)\)")

def parse_iso8601(s):
    return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S%z')

def parse_step(step: dict):
    return {
        'number': step['number'],
        'name': step['name'],
        'conclusion': step['conclusion'],
        'duration_s': (parse_iso8601(step['completed_at']) - parse_iso8601(step['started_at'])).seconds if step['conclusion'] else None,
    }

def parse_job_name(name: str):
    job_name, job_config = JOB_NAME_PATTERN.match(name).groups()
    os_config = KNOWN_OS[job_name]
    matrix_config = dict(zip(os_config.matrix, job_config.split(', ')))

    target_os = None
    target_arch = None
    compiler = None
    build_config = None
    if os_config.name == "macos":
        target_os = "macos"
        if matrix_config['os'] == "x64":
            target_arch = "x64"
        else:
            target_arch = "arm64"
        if matrix_config['os'] == "github-arm":
            build_config = "debug"
        else:
            build_config = "release"
    elif os_config.name in ("ubuntu-arm64", "ubuntu-x64") :
        target_os = matrix_config['os']
        if os_config.name == "ubuntu-arm64":
            target_arch = "arm64"
        else:
            target_arch = "x64"
        build_config = matrix_config['config'].lower()
    elif os_config.name == "emscripten":
        if matrix_config['config'] == "Singlethreaded":
            target_os = "emscripten-singlethreaded"
        else:
            target_os = "emscripten"
        target_arch = "wasm"
        compiler = "clang"
        build_config = "release"
    elif os_config.name == "fedora":
        target_os = "fedora39"
        target_arch = "x64"
        build_config = matrix_config['config'].lower()
    elif os_config.name == "windows":
        target_os = "windows"
        target_arch = "x64"
        compiler = matrix_config['runner'].replace("windows", "msvc")
        build_config = matrix_config['config'].lower()

    return {
        'target_os': target_os,
        'target_arch': target_arch,
        'compiler': compiler,
        'build_config': build_config,
    }

def parse_job(job: dict):
    job_id = job['id']
    runner_stats = {}
    stats_filename = Path(f'RunnerSysStats-{job_id}.json')
    if stats_filename.is_file():
        with open(stats_filename, 'r') as f:
            runner_stats = json.load(f)
    job_config = parse_job_name(job['name'])
    return {
        'id': job['id'],
        'conclusion': job['conclusion'],
        'duration_s': (parse_iso8601(job['completed_at']) - parse_iso8601(job['started_at'])).seconds if job['conclusion'] else None,
        'steps': [parse_step(step) for step in job['steps']],
        'target_os': job_config['target_os'],
        'target_arch': job_config['target_arch'],
        'compiler': job_config['compiler'] or runner_stats['compiler'],
        'build_config': job_config['build_config'],
        'runner_name': job['runner_name'],
        'runner_group_name': job['runner_group_name'],
        'runner_cpu_count': runner_stats.get('runner_cpu_count', None),
        'runner_ram_mb': runner_stats.get('runner_ram_mb', None),
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
    branch = os.environ.get('GIT_BRANCH')
    commit = os.environ.get('GIT_COMMIT')
    repo = os.environ.get("GITHUB_REPOSITORY")
    ref = os.environ.get("GITHUB_REF")
    run_id = os.environ.get("GITHUB_RUN_ID")

    resp = fetch_jobs(repo, run_id)

    result = {
        'id': int(run_id),
        'git_commit': commit,
        'git_branch': branch,
        'github_ref': ref,
        'github_repo': repo,
        'jobs': parse_jobs(resp.json()['jobs']),
    }
    pprint.pp(result, indent=2, width=150)

    requests.post("https://api.meshinspector.com/ci-stats/v1/log", json=result, headers={
        'Authorization': f'Bearer {os.environ.get("CI_STATS_AUTH_TOKEN")}',
    })
