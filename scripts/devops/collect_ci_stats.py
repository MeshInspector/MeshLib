#!/usr/bin/env python3
import datetime
import json
import os
import pprint
import time
from pathlib import Path
from typing import List

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import requests

API_URL = "https://api.meshinspector.com/ci-stats/v2/log"

def parse_iso8601(s):
    return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S%z')

def get_duration(obj):
    if obj.get('started_at') and obj.get('completed_at'):
        return parse_iso8601(obj['completed_at']) - parse_iso8601(obj['started_at'])

def get_duration_s(obj):
    dur = get_duration(obj)
    return dur.seconds if dur else None

def parse_step(step: dict):
    return {
        'number':     step['number'],
        'name':       step['name'],
        'conclusion': step['conclusion'],
        'duration_s': get_duration_s(step),
    }

def parse_job(job: dict):
    job_id = job['id']

    runner_type = "self-hosted"
    runner_name = job['runner_name'] or ""
    if job['runner_group_name'] == "GitHub Actions" or runner_name.startswith("GitHub Actions"):
        runner_type = "github actions"
        runner_name = None
    elif runner_name.startswith("i-0"):
        runner_type = "aws instance"
        runner_name = None

    stats_filename = Path(f"RunnerSysStats-{job_id}.json")
    artifact_stats_filename = Path(f"ArtifactStats-{job_id}.json")
    if not stats_filename.exists():
        return None

    runner_stats = None
    try:
        with open(stats_filename, 'r') as f:
            runner_stats = json.load(f)

        artifact_size = None
        if artifact_stats_filename.exists():
            with open(artifact_stats_filename, 'r') as f:
                artifact_stats = json.load(f)
                artifact_size = sum(artifact_stats.values())

        return {
            'id':                job['id'],
            'conclusion':        job['conclusion'],
            'duration_s':        get_duration_s(job),
            'steps':             [parse_step(step) for step in job['steps']],
            'target_os':         runner_stats['target_os'],
            'target_arch':       runner_stats['target_arch'],
            'compiler':          runner_stats['compiler'],
            'build_config':      runner_stats['build_config'],
            'runner_type':       runner_type,
            'runner_name':       runner_name,
            'runner_cpu_count':  runner_stats['cpu_count'],
            'runner_cpu_model':  runner_stats.get('cpu_model'),
            'runner_ram_mb':     runner_stats['ram_mb'],
            'runner_free_disk_mb': runner_stats.get('free_disk_mb'),
            'build_system':      runner_stats['build_system'],
            'aws_instance_type': runner_stats['aws_instance_type'],
            'artifact_size':     artifact_size,
        }
    except:
        print("Something went wrong while parsing the job/runner info. Debug info:")
        pprint.pp(job)
        pprint.pp(runner_stats)
        # re-throw the exception
        raise

def parse_jobs(jobs: List[dict]):
    return [
        job
        for job in [
            parse_job(job)
            for job in jobs
        ]
        if job is not None
    ]

def fetch_page(url, headers, attempts=3, cooldown=30):
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, headers=headers)
        except requests.exceptions.RequestException as e:
            if attempt == attempts:
                raise
            print(f'fetch_page: attempt {attempt}/{attempts} failed ({e}); retrying in {cooldown}s...')
            time.sleep(cooldown)
            continue
        resp.raise_for_status()
        return resp

def fetch_jobs(repo: str, run_id: str):
    url = f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100'
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'Bearer {os.environ.get("GITHUB_TOKEN")}',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    jobs = []
    while url:
        resp = fetch_page(url, headers)
        jobs += resp.json()['jobs']
        url = resp.links.get('next', {}).get('url')
    return jobs

def sign_api_request(url, method, headers, body, region, service):
    # Use the credentials from the assumed role
    session = boto3.Session()
    credentials = session.get_credentials().get_frozen_credentials()

    request = AWSRequest(
        method=method,
        url=url,
        headers=headers,
        data=json.dumps(body)
    )

    # Sign the request with the SigV4Auth class
    SigV4Auth(credentials, service, region).add_auth(request)

    return request

if __name__ == "__main__":
    branch = os.environ.get('GIT_BRANCH')
    commit = os.environ.get('GIT_COMMIT')
    repo = os.environ.get("GITHUB_REPOSITORY")
    ref = os.environ.get("GITHUB_REF")
    run_id = os.environ.get("GITHUB_RUN_ID")

    result = {
        'id':          int(run_id),
        'git_commit':  commit,
        'git_branch':  branch,
        'github_ref':  ref,
        'github_repo': repo,
        'jobs':        parse_jobs(fetch_jobs(repo, run_id)),
    }
    pprint.pp(result, indent=2, width=150)

    stats_file_count = len(list(Path('.').glob('RunnerSysStats-*.json')))
    if stats_file_count != len(result['jobs']):
        print(f"WARNING: found {stats_file_count} RunnerSysStats files but the payload has {len(result['jobs'])} jobs")

    headers = {
        'Content-Type': 'application/json',
    }

    if os.environ.get("CI_STATS_AUTH_TOKEN"):
        headers['Authorization'] = f'Bearer {os.environ.get("CI_STATS_AUTH_TOKEN")}'

    signed_request = sign_api_request(
        API_URL,
        'POST',
        headers,
        result,
        'us-east-1',
        'execute-api' # Service name for API Gateway
    )

    response = requests.post(
        API_URL,
        headers=dict(signed_request.headers.items()),  # Use signed headers
        data=signed_request.body
    )

    if response.status_code == 200:
        print("Successfully sent the CI stats to the API")
    else:
        raise RuntimeError(f'{response.status_code}: {response.text}')
