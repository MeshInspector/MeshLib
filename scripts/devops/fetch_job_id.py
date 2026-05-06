import json
import os
import ssl
import subprocess
import sys
import urllib.request


def _resolve_ca_bundle():
    """Return a path to a CA bundle, or None to use the stdlib default.

    Brew Python on the self-hosted macos-x64 runner ships without a working
    CA bundle, so urllib's default context fails the TLS handshake to
    api.github.com with CERTIFICATE_VERIFY_FAILED. certifi provides one;
    pip-install it on the fly if it isn't already there.
    """
    try:
        import certifi
        return certifi.where()
    except ImportError:
        pass
    pip_install = [sys.executable, "-m", "pip", "install",
                   "--quiet", "--disable-pip-version-check", "--user", "certifi"]
    try:
        subprocess.check_call(pip_install)
    except subprocess.CalledProcessError:
        try:
            subprocess.check_call(pip_install + ["--break-system-packages"])
        except subprocess.CalledProcessError:
            return None
    try:
        import certifi
        return certifi.where()
    except ImportError:
        return None


_SSL_CONTEXT = ssl.create_default_context(cafile=_resolve_ca_bundle())

GITHUB_HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'Bearer {os.environ.get("GITHUB_TOKEN")}',
    'X-GitHub-Api-Version': '2022-11-28',
}

def fetch_jobs(repo: str, run_id: str):
    # TODO: pagination support
    # more info: https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api
    url = f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100'
    req = urllib.request.Request(url, headers=GITHUB_HEADERS)
    with urllib.request.urlopen(req, context=_SSL_CONTEXT) as resp:
        return json.loads(resp.read().decode())

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
