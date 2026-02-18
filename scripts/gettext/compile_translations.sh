#!/bin/sh

APP_NAME=$1
OUTPUT_DIR=$2

LOCALE_DIR=$PWD/locale

for LOCALE in $(ls $LOCALE_DIR) ; do
  if [ -d $LOCALE_DIR/$LOCALE ] && [ -f $LOCALE_DIR/$LOCALE/$APP_NAME.po ] ; then
    if [ ! -d $OUTPUT_DIR/$LOCALE/LC_MESSAGES ] ; then
      mkdir -p $OUTPUT_DIR/$LOCALE/LC_MESSAGES
    fi
    msgfmt \
      $LOCALE_DIR/$LOCALE/$APP_NAME.po \
      --output-file=$OUTPUT_DIR/$LOCALE/LC_MESSAGES/$APP_NAME.mo
  fi
done
