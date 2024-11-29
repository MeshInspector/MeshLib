import json
import os
import pprint
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
    return job['status'] == "in_progress" and job['name'].find(job_name) != -1 and job['runner_name'] == runner_name

if __name__ == "__main__":
    job_name = os.environ.get("GITHUB_JOB")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    runner_name = os.environ.get("RUNNER_NAME")

    resp = fetch_jobs(repo, run_id)
    pprint.pp({
        'job_name': job_name,
        'runner_name': runner_name,
        'jobs': [
            {
                'id': job['id'],
                'name': job['name'],
                'runner_name': job['runner_name'],
            }
            for job in resp['jobs']
            if filter_job(job, job_name, runner_name)
        ],
    }, indent=2, width=120)
