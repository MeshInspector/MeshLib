#!/bin/bash
#
# Wrapper for `brew` that suppresses the spurious GitHub Actions workflow
# commands Homebrew emits when GITHUB_ACTIONS=true. In particular, Homebrew
# prints `::error::` workflow commands for benign cases like
# "<pkg> is already installed" (rendered as red-cross failure annotations
# even though brew exits 0 because of HOMEBREW_NO_INSTALL_UPGRADE=1) and
# `::warning::` for the sibling "<pkg> is already installed and up-to-date"
# case (yellow-triangle warning annotations).
#
# We bracket the brew call with the `::stop-commands::` workflow command so
# the GHA runner ignores any annotation directives Homebrew prints during
# this call, then re-enable command parsing with the matching sentinel
# afterwards. The exit status of brew is preserved.
#
# Outside CI this script is a transparent passthrough — no annotations would
# be parsed anyway.
#
# Usage: scripts/brew_quiet.sh <subcommand> [args...]
#        e.g. scripts/brew_quiet.sh install gettext

if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  echo "::stop-commands::brew-quiet"
  brew "$@"
  rc=$?
  echo "::brew-quiet::"
  exit $rc
else
  exec brew "$@"
fi
