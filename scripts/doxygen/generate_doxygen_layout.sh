#!/bin/sh

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "${SCRIPT}")
DOXYGEN_DIR=$(readlink -f "${SCRIPTPATH}/../../doxygen")

BEGIN_URL=""
END_URL=".html"

if [ "$2" == "127.0.0.1:8000/MeshLib/local" ]; then
    BEGIN_URL="http://${2}"
else
    BEGIN_URL="https://${2}"
fi

sed -e "s|__BEGIN_URL__|${BEGIN_URL}/Main/html/|" -e "s|__END_URL__|${END_URL}|" ${DOXYGEN_DIR}/layout_templates/base_struct.xml > ${DOXYGEN_DIR}/DoxygenLayout${1}.xml

if [ "$1" != "Cpp" ]; then
    sed -e "s|      <!-- API_CPP_PAGE -->|      <tab type=\"user\" url=\"${BEGIN_URL}/Cpp/html/APICppPage.html\" title=\"C++\"/>|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
else
    sed -e "/      <!-- API_CPP_PAGE -->/r ${DOXYGEN_DIR}/layout_templates/API_part.xml" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
    sed -e "s|__API_PAGE_URL__|${BEGIN_URL}/Cpp/html/APICppPage.html|" -e "s|__API_PAGE_NAME__|C++|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
fi

if [ "$1" != "Py" ]; then
    sed -e "s|      <!-- API_PY_PAGE -->|      <tab type=\"user\" url=\"${BEGIN_URL}/Py/html/APIPyPage.html\" title=\"Python\"/>|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
else
    sed -e "/      <!-- API_PY_PAGE -->/r ${DOXYGEN_DIR}/layout_templates/API_part.xml" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
    sed -e "s|__API_PAGE_URL__|${BEGIN_URL}/Py/html/APIPyPage.html|" -e "s|__API_PAGE_NAME__|Python|" -i ${DOXYGEN_DIR}/DoxygenLayout${1}.xml
fi


