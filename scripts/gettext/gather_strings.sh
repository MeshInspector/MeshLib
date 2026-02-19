#!/bin/sh

APP_NAME=$1
shift

LOCALE_DIR=$PWD/locale

xgettext \
    -k_t \
    -k_tr \
    -kf_tr \
    -kn_t:1,2 \
    -kn_tr:1,2 \
    -kfn_tr:1,2 \
    -kp_t:1c,2 \
    -kp_tr:1c,2 \
    -kfp_tr:1c,2 \
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
