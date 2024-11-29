import json
import math
import multiprocessing
import os
import platform
import pprint
import re
import subprocess
import urllib.request

def get_ram_amount():
    system = platform.system()
    if system == "Darwin":
        output = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout
        return int(output)
    elif system == "Linux":
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    elif system == "Windows":
        output = subprocess.run(['wmic', 'ComputerSystem', 'get', 'TotalPhysicalMemory'], capture_output=True, text=True).stdout
        return int(re.search(r'\d+', output).group())
    else:
        raise Exception(f"Unknown system {system}")

def get_system_stats():
    return {
        'cpu_count': multiprocessing.cpu_count(),
        'ram_mb': math.floor(get_ram_amount() / 1024 / 1024),
    }

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
        raise Exception(f"No jobs found for {job_name}")
    elif len(jobs) > 1:
        raise Exception(f"Multiple jobs found for {job_name}: {jobs}")
    else:
        return jobs[0]['id']

if __name__ == "__main__":
    stats = get_system_stats()
    pprint.pprint(stats)

    job_id = get_job_id()
    os.environ['GITHUB_JOB_ID'] = job_id

    stats_filename = f"RunnerSysStats-{job_id}.json"
    with open(stats_filename, 'w') as f:
        json.dump(stats, f)
    print(f"Report is available at {stats_filename}")
