#!/bin/bash
# Configure VCPKG_BINARY_SOURCES to use the shared S3 vcpkg binary cache.
#
# Meant to be *sourced* (not executed) from the Linux vcpkg Dockerfiles, so the
# exported variables reach the subsequent `vcpkg install`. Mirrors the Windows
# setup in thirdparty/install.bat, using the same bucket and key layout:
#   s3://vcpkg-export/<vcpkg version>/<triplet>/
#
# Requires VCPKG_VERSION and VCPKG_TRIPLET in the environment. AWS credentials
# are read from optional BuildKit secrets (/run/secrets/AWS_*); when present the
# cache is read-write, otherwise it falls back to anonymous read-only access.

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

S3_URL="s3://vcpkg-export/${VCPKG_VERSION}/${VCPKG_TRIPLET}/"

if [ -s /run/secrets/AWS_ACCESS_KEY_ID ] && [ -s /run/secrets/AWS_SECRET_ACCESS_KEY ]; then
    export AWS_ACCESS_KEY_ID="$(cat /run/secrets/AWS_ACCESS_KEY_ID)"
    export AWS_SECRET_ACCESS_KEY="$(cat /run/secrets/AWS_SECRET_ACCESS_KEY)"
    # session token is only present for temporary (OIDC) credentials
    if [ -s /run/secrets/AWS_SESSION_TOKEN ]; then
        export AWS_SESSION_TOKEN="$(cat /run/secrets/AWS_SESSION_TOKEN)"
    fi
    echo "vcpkg S3 binary cache: read-write (${S3_URL})"
    export VCPKG_BINARY_SOURCES="clear;x-aws,${S3_URL},readwrite"
else
    echo "vcpkg S3 binary cache: anonymous read-only (${S3_URL})"
    export VCPKG_BINARY_SOURCES="clear;x-aws-config,no-sign-request;x-aws,${S3_URL},read"
fi
