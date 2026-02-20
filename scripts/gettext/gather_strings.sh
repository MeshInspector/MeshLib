#!/bin/sh

APP_NAME=$1
shift

LOCALE_DIR=$PWD/locale

xgettext \
    -ktranslate:1,1t \
    -ktranslate:1c,2,2t \
    -ktranslate:1,2,3t \
    -ktranslate:1c,2,3,4t \
    -k_t:1,1t \
    -k_t:1c,2,2t \
    -k_t:1,2,3t \
    -k_t:1c,2,3,4t \
    -k_tr:1,1t \
    -k_tr:1c,2,2t \
    -k_tr:1,2,3t \
    -k_tr:1c,2,3,4t \
    -kf_tr:1,1t \
    -kf_tr:1c,2,2t \
    -kf_tr:1,2,3t \
    -kf_tr:1c,2,3,4t \
    --language=C++ \
    --from-code=UTF-8 \
    --add-comments="TRANSLATORS:" \
    --default-domain=$APP_NAME \
    --output=$LOCALE_DIR/$APP_NAME.pot \
    "$@"
# xgettext doesn't set any encoding if the messages contain only ASCII symbols
# which leads to choosing ASCII as a default encoding for .po files
# so we need to set the encoding manually
sed -i 's|Content-Type: text/plain; charset=CHARSET|Content-Type: text/plain; charset=UTF-8|' $LOCALE_DIR/$APP_NAME.pot
