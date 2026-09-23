#!/bin/bash

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

S3_URL="s3://vcpkg-export/${VCPKG_VERSION}/${VCPKG_TRIPLET}/"
# reading via HTTP is batched and thus usually faster than via x-aws
HTTP_URL="https://vcpkg-export.s3.${AWS_DEFAULT_REGION}.amazonaws.com/${VCPKG_VERSION}/${VCPKG_TRIPLET}/{sha}.zip"

# pick up credentials from Docker secrets if available
if [ -s /run/secrets/AWS_ACCESS_KEY_ID ] && [ -s /run/secrets/AWS_SECRET_ACCESS_KEY ]; then
    export AWS_ACCESS_KEY_ID="$(cat /run/secrets/AWS_ACCESS_KEY_ID)"
    export AWS_SECRET_ACCESS_KEY="$(cat /run/secrets/AWS_SECRET_ACCESS_KEY)"
    if [ -s /run/secrets/AWS_SESSION_TOKEN ]; then
        export AWS_SESSION_TOKEN="$(cat /run/secrets/AWS_SESSION_TOKEN)"
    fi
fi

if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "vcpkg S3 binary cache: read ${HTTP_URL}, write ${S3_URL}"
    export VCPKG_BINARY_SOURCES="clear;http,${HTTP_URL},read;x-aws,${S3_URL},write"
else
    echo "vcpkg S3 binary cache: anonymous read-only (${HTTP_URL})"
    export VCPKG_BINARY_SOURCES="clear;http,${HTTP_URL},read"
fi
