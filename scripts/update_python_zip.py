#!/usr/bin/env python3
"""Regenerates thirdparty/python/python3XY.zip, the zipped standard library shipped next to
python3XY.dll on Windows, from the CPython sources of vcpkg's python3 port.

Usage: update_python_zip.py [vcpkg-tag]   (default: the vcpkg version in .github/workflows/config.yml)

The port's version, source checksum and patches are read at that vcpkg tag, so the zip matches the
Python that vcpkg builds. Run it with that Python minor version (e.g. `uv run --python 3.12`), or
have `uv` on PATH and the script re-runs itself under the right version.
"""
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# the packages python.org's Windows embeddable package leaves out, plus test data and caches
EXCLUDE_DIRS = {'test', 'tests', 'idlelib', 'tkinter', 'turtledemo', 'ensurepip', 'venv', '__pycache__', 'site-packages'}
EXCLUDE_FILES = {'turtle.py'}


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def vcpkg_tag_from_config():
    with open(os.path.join(ROOT, '.github', 'workflows', 'config.yml')) as f:
        return re.search(r'vcpkg_version:.*?value:\s*"([^"]+)"', f.read(), re.S).group(1)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else vcpkg_tag_from_config()
    port = f'https://raw.githubusercontent.com/microsoft/vcpkg/{tag}/ports/python3'
    version = re.search(r'"version":\s*"([^"]+)"', fetch(f'{port}/vcpkg.json').decode()).group(1)
    major, minor = (int(x) for x in version.split('.')[:2])
    if sys.version_info[:2] != (major, minor):
        if shutil.which('uv'):
            sys.exit(subprocess.call(['uv', 'run', '--no-project', '--python', f'{major}.{minor}',
                                      'python', os.path.abspath(__file__), tag]))
        sys.exit(f'vcpkg {tag} has Python {version}; run this script with Python {major}.{minor}')

    portfile = fetch(f'{port}/portfile.cmake').decode()
    sha512 = re.search(r'REPO python/cpython.*?SHA512\s+([0-9a-f]+)', portfile, re.S).group(1)
    patches = re.findall(r'^\s+(\d{4}-[\w.-]+\.patch)', re.search(r'set\(PATCHES(.*?)\)', portfile, re.S).group(1), re.M)
    archive = fetch(f'https://github.com/python/cpython/archive/v{version}.tar.gz')
    if hashlib.sha512(archive).hexdigest() != sha512:
        sys.exit(f'SHA512 of CPython {version} does not match vcpkg {tag}')

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            # GitHub archives start with a pax_global_header entry, so find the root via Lib/os.py
            prefix = next(n for n in tar.getnames() if n.endswith('/Lib/os.py'))[:-len('os.py')]
            members = [m for m in tar.getmembers() if m.name.startswith(prefix)]
            for m in members:
                m.name = 'Lib/' + m.name[len(prefix):]
            tar.extractall(tmp, members, filter='data')
        for patch in patches:
            path = os.path.join(tmp, patch)
            with open(path, 'wb') as f:
                f.write(fetch(f'{port}/{patch}'))
            subprocess.check_call(['git', 'apply', '--include=Lib/*', path], cwd=tmp)

        lib = os.path.join(tmp, 'Lib')
        for address, dirs, files in os.walk(lib):
            for d in [d for d in dirs if d in EXCLUDE_DIRS]:
                shutil.rmtree(os.path.join(address, d))
                dirs.remove(d)
            for f in files:
                if f in EXCLUDE_FILES and address == lib:
                    os.remove(os.path.join(address, f))
        # as in the embeddable package: no docstrings or asserts (-OO), and unchecked-hash .pyc,
        # since there is no source in the zip to check against; file names relative to Lib (-s)
        subprocess.check_call([sys.executable, '-I', '-m', 'compileall', '-q', '-b', '-o', '2',
                               '--invalidation-mode', 'unchecked-hash', '-s', lib, lib])

        zip_path = os.path.join(ROOT, 'thirdparty', 'python', f'python{major}{minor}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for address, dirs, files in sorted(os.walk(lib)):
                dirs.sort()
                for f in sorted(files):
                    if f.endswith('.py'):
                        continue
                    src = os.path.join(address, f)
                    info = zipfile.ZipInfo(os.path.relpath(src, lib).replace(os.sep, '/'))
                    with open(src, 'rb') as data:
                        z.writestr(info, data.read(), zipfile.ZIP_DEFLATED, 9)
    print(f'Wrote {zip_path} from CPython {version} (vcpkg {tag})')


main()
