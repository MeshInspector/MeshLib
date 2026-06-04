#!/bin/bash

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

S3_URL="s3://vcpkg-export/${VCPKG_VERSION}/${VCPKG_TRIPLET}/"

# pick up credentials from Docker secrets if available
if [ -s /run/secrets/AWS_ACCESS_KEY_ID ] && [ -s /run/secrets/AWS_SECRET_ACCESS_KEY ]; then
    export AWS_ACCESS_KEY_ID="$(cat /run/secrets/AWS_ACCESS_KEY_ID)"
    export AWS_SECRET_ACCESS_KEY="$(cat /run/secrets/AWS_SECRET_ACCESS_KEY)"
    if [ -s /run/secrets/AWS_SESSION_TOKEN ]; then
        export AWS_SESSION_TOKEN="$(cat /run/secrets/AWS_SESSION_TOKEN)"
    fi
fi

if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "vcpkg S3 binary cache: read-write (${S3_URL})"
    export VCPKG_BINARY_SOURCES="clear;x-aws,${S3_URL},readwrite"
else
    echo "vcpkg S3 binary cache: anonymous read-only (${S3_URL})"
    export VCPKG_BINARY_SOURCES="clear;x-aws-config,no-sign-request;x-aws,${S3_URL},read"
fi
