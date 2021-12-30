cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project (OpenCTM C)

file(GLOB SOURCES "*.c")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set(OPENCTM_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}
    CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)
