## Release branches

`master` is the development line. A `release/<major>.<minor>` branch carries a shipped line, so a fix
can go out without also shipping whatever else landed on `master`.

Only the newest release branch receives updates. Older ones are frozen — see
[Freezing a line](#freezing-a-line).

### Versioning

Both lines use the same scheme, `vA.B.C.N`, computed from the nearest reachable `vA.B.C` tag. A
release branch adds an `-rc` suffix to the tag of its draft releases, and nothing else: the version
stamped into wheels, npm packages, NuGet packages and `.deb`/`.rpm` files is the plain `A.B.C.N`.

The suffix exists because both lines count from the same tag, so the same `vA.B.C.N` can come up on
each. It keeps their draft releases apart. Publishing drops it.

### Cutting a branch

Branch from the commit you want to ship and push it:

```bash
git switch -c release/3.1 <commit>
git push -u origin release/3.1
```

Every push to it then runs the same CI as `master`, plus the wheel build, and attaches the artifacts
to a draft release tagged `vA.B.C.N-rc`. Take fixes onto the branch with `git cherry-pick` — land
them on `master` first.

Do not merge `master` into a release branch. A merge pulls `master`'s newer tags into the branch's
history and moves its version.

### Publishing

Go to Actions, choose **Publish release**, and pick the release branch in the "Run workflow" branch
selector. There are no inputs.

It takes the newest `-rc` draft on the branch and:

1. creates the plain `vA.B.C` tag on the draft's commit,
2. republishes the draft under `vA.B.C.N`, without the `-rc`,
3. writes release notes covering only this patch, diffed against the previous release on the same
   line.

Publishing then triggers one workflow per target — **Publish to PyPI**, **Sign and Upload NuGet**,
**Publish Wasm module** and **Publish documentation**. Each ships what is already attached to the
release, so what reaches the registries is what was built and tested on that commit. Nothing is
rebuilt. The C++ distributions need no publish step; the release itself is the channel.

The workflow refuses to run if the branch is not a `release/` one, if the draft's commit is not on the
branch, or if the `vA.B.C` tag already exists on a different commit.

### Post-release patches

A branch is not one-shot. Cherry-pick the next fix, push, and publish again: the `vA.B.C` tag from the
previous publish is what makes the next build come out as `vA.B.(C+1).1-rc`.

### Freezing a line

When a newer release branch is cut, stop publishing from the old one. Publishing an older line after a
newer one would point npm's `latest` tag and the unversioned CDN path backwards.

Publishing checks for this and will still upload the packages, but leaves `latest` and the unversioned
CDN path on the newer release, and puts the npm packages under a `release-<major>.<minor>` tag
instead.

### When a publish half-finishes

The tags and the release are created before anything is uploaded, so a failure partway leaves the
release published and some registries missing it. Do not publish again — the release already exists.

Re-run the failed publisher for the tag that was published. Each takes the tag as its only input and
picks the packages up from the release, so re-running is safe and ships the same bytes.
