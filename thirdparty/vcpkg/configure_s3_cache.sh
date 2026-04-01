#!/bin/bash

VCPKG_TAG="$(echo "${VCPKG_VERSION}" | tr '.' '-')"

# pick up credentials from Docker secrets if available
if [ -s /run/secrets/aws_access_key_id ] && [ -s /run/secrets/aws_secret_access_key ]; then
    export AWS_ACCESS_KEY_ID=$(cat /run/secrets/aws_access_key_id)
    export AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/aws_secret_access_key)
    export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
fi

if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    export VCPKG_BINARY_SOURCES="clear;x-aws,s3://vcpkg-export/${VCPKG_TAG}/${VCPKG_TRIPLET}/,readwrite"
else
    export VCPKG_BINARY_SOURCES="clear;x-aws-config,no-sign-request;x-aws,s3://vcpkg-export/${VCPKG_TAG}/${VCPKG_TRIPLET}/,read"
fi
