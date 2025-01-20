#!/bin/sh

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
    sed -e "s|@ref .*#|@ref |" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
else
    sed -e "s|__MAIN_PAGE_TAB__|<tab type=\"user\" url=\"../index.html\" title=\"About\"/>|"  -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
    sed -e "s|__BEGIN_URL__|../|" -e "s|__END_URL__|${END_URL}|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
fi

# edit C++ API block
if [ "$1" = "Main" ]; then
    sed -e "s|      <!-- API_CPP_PAGE -->|      <tab type=\"user\" url=\"Cpp/APICppPage.html\" title=\"C++\"/>|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
elif [ "$1" = "Py" ]; then
    sed -e "s|      <!-- API_CPP_PAGE -->|      <tab type=\"user\" url=\"../Cpp/APICppPage.html\" title=\"C++\"/>|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
else
    sed -e "/      <!-- API_CPP_PAGE -->/r ${DOXYGEN_DIR}/layout_templates/API_part.xml" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
    sed -e "s|__API_PAGE_URL__|@ref APICppPage|" -e "s|__API_PAGE_NAME__|C++|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
fi

# edit Python API block
if [ "$1" = "Main" ]; then
    sed -e "s|      <!-- API_PY_PAGE -->|      <tab type=\"user\" url=\"Py/APIPyPage.html\" title=\"Python\"/>|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
elif [ "$1" = "Cpp" ]; then
    sed -e "s|      <!-- API_PY_PAGE -->|      <tab type=\"user\" url=\"../Py/APIPyPage.html\" title=\"Python\"/>|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
else
    sed -e "/      <!-- API_PY_PAGE -->/r ${DOXYGEN_DIR}/layout_templates/API_part.xml" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
    sed -e "s|__API_PAGE_URL__|@ref APIPyPage|" -e "s|__API_PAGE_NAME__|Python|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
fi

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
