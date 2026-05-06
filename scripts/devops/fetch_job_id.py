import json
import os
import ssl
import subprocess
import urllib.error
import urllib.request


GITHUB_HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'Bearer {os.environ.get("GITHUB_TOKEN")}',
    'X-GitHub-Api-Version': '2022-11-28',
}


def _fetch_jobs_urllib(repo: str, run_id: str):
    url = f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100'
    req = urllib.request.Request(url, headers=GITHUB_HEADERS)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _fetch_jobs_gh_cli(repo: str, run_id: str):
    # Fallback for runners whose Python has no working CA bundle
    # (notably brew Python 3.14 on the macos-x64 self-hosted runner). gh
    # ships its own certificates and uses GITHUB_TOKEN/GH_TOKEN automatically.
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def fetch_jobs(repo: str, run_id: str):
    # TODO: pagination support
    # more info: https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api
    try:
        return _fetch_jobs_urllib(repo, run_id)
    except (ssl.SSLError, urllib.error.URLError) as e:
        # urllib's URLError wraps the underlying SSLError; only fall back
        # for cert-verification problems, not for network/HTTP failures
        # we'd rather see surfaced.
        cause = getattr(e, "reason", e)
        if not isinstance(cause, ssl.SSLError):
            raise
        print(f"urllib failed with {cause!r}; retrying via gh CLI")
        return _fetch_jobs_gh_cli(repo, run_id)


def filter_job(job, job_name, runner_name):
    return job['status'] == "in_progress" and job_name in job['name'] and runner_name in [job['runner_name'], f"GitHub-Actions-{job['runner_id']}"]

def get_job_id():
    job_name = os.environ.get("GITHUB_JOB")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    runner_name = os.environ.get("RUNNER_NAME")

    resp = fetch_jobs(repo, run_id)
    if int(resp['total_count']) > 100:
        print("Total job count has exceeded 100; consider enabling the pagination support")
    jobs = [
        job
        for job in resp['jobs']
        if filter_job(job, job_name, runner_name)
    ]
    if len(jobs) == 0:
        raise RuntimeError(f"No jobs found for {job_name}")
    elif len(jobs) > 1:
        raise RuntimeError(f"Multiple jobs found for {job_name}: {jobs}")
    else:
        return jobs[0]['id']

if __name__ == "__main__":
    if os.environ.get('CI'):
        job_id = get_job_id()
        with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
            print(f"job_id={job_id}", file=out)
