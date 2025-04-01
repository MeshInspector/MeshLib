import json
import os
import urllib.request

GITHUB_HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'Bearer {os.environ.get("GITHUB_TOKEN")}',
    'X-GitHub-Api-Version': '2022-11-28',
}

def fetch_jobs(repo: str, run_id: str):
    url = f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs'
    req = urllib.request.Request(url, headers=GITHUB_HEADERS)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def filter_job(job, job_name, runner_name):
    return job['status'] == "in_progress" and job_name in job['name'] and runner_name in [job['runner_name'], f"GitHub-Actions-{job['runner_id']}"]

def get_job_id():
    job_name = os.environ.get("GITHUB_JOB")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    runner_name = os.environ.get("RUNNER_NAME")

    resp = fetch_jobs(repo, run_id)
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
        with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
            print(f"job_id={get_job_id()}", file=out)
