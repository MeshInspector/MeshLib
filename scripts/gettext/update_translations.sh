#!/bin/sh

APP_NAME=$1

LOCALE_DIR=$PWD/locale

for LOCALE in $(ls $LOCALE_DIR) ; do
  if [ -d $LOCALE_DIR/$LOCALE ] && [ -f $LOCALE_DIR/$LOCALE/$APP_NAME.po ] ; then
    msgmerge \
      --update \
      $LOCALE_DIR/$LOCALE/$APP_NAME.po \
      $LOCALE_DIR/$APP_NAME.pot
  fi
done
