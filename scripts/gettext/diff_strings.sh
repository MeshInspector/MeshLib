#!/bin/sh

APP_NAME=$1

LOCALE_DIR=$PWD/locale

git diff \
    --unified=0 \
    --ignore-matching-lines="POT-Creation-Date.*" \
    $LOCALE_DIR/$APP_NAME.pot
