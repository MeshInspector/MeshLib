#!/bin/sh

APP_NAME=$1
shift

LOCALE_DIR=$PWD/locale

XGETTEXT_OPTIONS=$(cat $LOCALE_DIR/xgettext_options.txt | tr '\n' ' ')
xgettext \
    $XGETTEXT_OPTIONS \
    --default-domain=$APP_NAME \
    --output=$LOCALE_DIR/$APP_NAME.pot \
    "$@"

for LOCALE in $(ls $LOCALE_DIR) ; do
  if [ -d $LOCALE_DIR/$LOCALE ] && [ -f $LOCALE_DIR/$LOCALE/$APP_NAME.po ] ; then
    msgmerge \
      --update \
      $LOCALE_DIR/$LOCALE/$APP_NAME.po \
      $LOCALE_DIR/$APP_NAME.pot
  fi
done
