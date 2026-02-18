#!/bin/bash

APP_NAME=$1
LOCALE=$(echo $2 | cut -d'.' -f1 | sed -e 's/-/_/')

LOCALE_DIR=$PWD/locale

if [ ! -d $LOCALE_DIR/$LOCALE ] ; then
    mkdir -p $LOCALE_DIR/$LOCALE
fi

msginit \
    --input=$LOCALE_DIR/$APP_NAME.pot \
    --locale=$LOCALE.UTF-8 \
    --output=$LOCALE_DIR/$LOCALE/$APP_NAME.po \
    --no-translator
