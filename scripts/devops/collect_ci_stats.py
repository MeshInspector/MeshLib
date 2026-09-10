#!/usr/bin/env python3
import datetime
import hashlib
import hmac
import json
import os
import pprint
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List

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
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request) as resp:
                return json.loads(resp.read()), resp.headers.get('Link', '')
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as e:
            if attempt == attempts:
                raise
            print(f'fetch_page: attempt {attempt}/{attempts} failed ({e}); retrying in {cooldown}s...')
            time.sleep(cooldown)

def next_page_url(link_header: str):
    match = re.search(r'<([^>]+)>\s*;\s*rel="next"', link_header)
    return match.group(1) if match else None

def fetch_jobs(repo: str, run_id: str):
    url = f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100'
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'Bearer {os.environ.get("GITHUB_TOKEN")}',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    jobs = []
    while url:
        page, link_header = fetch_page(url, headers)
        jobs += page['jobs']
        url = next_page_url(link_header)
    return jobs

def sign_api_request(url, method, headers, body: bytes, region, service):
    """Return the headers of an AWS SigV4-signed request; credentials come from the environment."""
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime('%Y%m%dT%H%M%SZ')
    scope = f"{now.strftime('%Y%m%d')}/{region}/{service}/aws4_request"

    signed = {k: v for k, v in headers.items() if k.lower() != 'authorization'}
    signed['Host'] = urllib.parse.urlsplit(url).netloc
    signed['X-Amz-Date'] = amz_date
    if os.environ.get('AWS_SESSION_TOKEN'):
        signed['X-Amz-Security-Token'] = os.environ['AWS_SESSION_TOKEN']

    names = sorted(k.lower() for k in signed)
    values = {k.lower(): ' '.join(v.split()) for k, v in signed.items()}
    canonical_request = '\n'.join([
        method,
        urllib.parse.quote(urllib.parse.urlsplit(url).path or '/', safe='/~'),
        urllib.parse.urlsplit(url).query,
        ''.join(f'{name}:{values[name]}\n' for name in names),
        ';'.join(names),
        hashlib.sha256(body).hexdigest(),
    ])
    to_sign = '\n'.join([
        'AWS4-HMAC-SHA256',
        amz_date,
        scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])

    key = f"AWS4{os.environ['AWS_SECRET_ACCESS_KEY']}".encode()
    for part in scope.split('/'):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()

    signed['Authorization'] = (
        f"AWS4-HMAC-SHA256 Credential={os.environ['AWS_ACCESS_KEY_ID']}/{scope}, "
        f"SignedHeaders={';'.join(names)}, Signature={signature}"
    )
    return signed

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

    body = json.dumps(result).encode()
    headers = sign_api_request(
        API_URL,
        'POST',
        {'Content-Type': 'application/json'},
        body,
        'us-east-1',
        'execute-api' # Service name for API Gateway
    )

    request = urllib.request.Request(API_URL, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request) as resp:
            if resp.status != 200:
                raise RuntimeError(f'{resp.status}: {resp.read().decode(errors="replace")}')
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'{e.code}: {e.read().decode(errors="replace")}')
    print("Successfully sent the CI stats to the API")
