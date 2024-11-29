import json
import multiprocessing
import os
import platform
import subprocess
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
    jobs = [
        job
        for job in resp['jobs']
        if filter_job(job, job_name, runner_name)
    ]
    if len(jobs) == 0:
        raise Exception(f"No jobs found for {job_name}")
    elif len(jobs) > 1:
        raise Exception(f"Multiple jobs found for {job_name}: {jobs}")
    else:
        print(f"job id: {jobs[0]['id']}")

    print(f"cpu count: {multiprocessing.cpu_count()}")

    system = platform.system()
    if system == "Darwin":
        print(f"memory amount: {subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True).stdout.decode()}")
    elif system == "Linux":
        print(f"memory amount: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')}")
    elif system == "Windows":
        print(f"memory amount: {subprocess.run(['wmic', 'ComputerSystem', 'get', 'TotalPhysicalMemory'], capture_output=True).stdout.decode()}")
    else:
        raise Exception(f"Unknown system {system}")
