#!/bin/bash

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "${SCRIPT}")
DOXYGEN_DIR=$(readlink -f "${SCRIPTPATH}/../../doxygen")

END_URL=".html"

# edit main block
cp ${DOXYGEN_DIR}/layout_templates/base_struct.xml ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
if [ "$1" = "Main" ]; then
    sed -e "s|__MAIN_PAGE_TAB__|<tab type=\"mainpage\" visible=\"yes\" title=\"About\"/>|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
    sed -e "s|__BEGIN_URL__|@ref |" -e "s|__END_URL__||" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
    # fix complex refs
    sed -e "s|@ref [^\"]*#|@ref |" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
else
    sed -e "s|__MAIN_PAGE_TAB__|<tab type=\"user\" url=\"../index.html\" title=\"About\"/>|"  -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
    sed -e "s|__BEGIN_URL__|../|" -e "s|__END_URL__|${END_URL}|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
fi

# edit API block
MODULE_PARAMETERS_MATRIX=(
  #TEMPLATE NAME TITLE
  "CPP:Cpp:C++"
  "PY:Py:Python"
  "C:C:C"
)

for MODULES_ROW in "${MODULE_PARAMETERS_MATRIX[@]}"; do
  IFS=':' read -r TEMPLATE NAME TITLE <<< "$MODULES_ROW"

  if [ "$1" = "Main" ]; then
    sed -e "s|      <!-- API_${TEMPLATE}_PAGE -->|      <tab type=\"user\" url=\"${NAME}/API${NAME}Page.html\" title=\"${TITLE}\"/>|" -i "${DOXYGEN_DIR}/DoxygenLayout${1}.xml"
  elif [ "$1" = "$NAME" ]; then
    sed -e "/      <!-- API_${TEMPLATE}_PAGE -->/r ${DOXYGEN_DIR}/layout_templates/API_part.xml" -i "${DOXYGEN_DIR}/DoxygenLayout${1}.xml"
    sed -e "s|__API_PAGE_URL__|@ref API${NAME}Page|" -e "s|__API_PAGE_NAME__|${TITLE}|" -i "${DOXYGEN_DIR}/DoxygenLayout${1}.xml"
  else
    sed -e "s|      <!-- API_${TEMPLATE}_PAGE -->|      <tab type=\"user\" url=\"../${NAME}/API${NAME}Page.html\" title=\"${TITLE}\"/>|" -i "${DOXYGEN_DIR}/DoxygenLayout${1}.xml"
  fi
done

# edit API titles
if [ "$1" = "Py" ]; then
    sed \
        -e "s|__NAMESPACES_TITLE__|Modules|" \
        -e "s|__NAMESPACES_LIST_TITLE__|Modules List|" \
        -e "s|__NAMESPACES_MEMBERS_TITLE__|Modules Members|" \
        -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
else
    sed \
        -e "s|__NAMESPACES_TITLE__||" \
        -e "s|__NAMESPACES_LIST_TITLE__||" \
        -e "s|__NAMESPACES_MEMBERS_TITLE__||" \
        -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
fi
