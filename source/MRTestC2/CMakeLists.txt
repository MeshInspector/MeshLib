cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# On Mac, strip `-stdlib=libc++` from linker flags to avoid the `unused flag` warning. Unused because this is a C project.
IF (APPLE)
  string(REPLACE "-stdlib=libc++" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
ENDIF()

project(MRTestC2 C)

file(GLOB HEADERS "*.h")
file(GLOB SOURCES "*.c")

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ../MeshLibC2/include)

IF(MR_EMSCRIPTEN)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")
ENDIF()

IF(NOT MR_EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MeshLibC2
  )
ELSE()
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  target_link_libraries(${PROJECT_NAME} PRIVATE
    gtest
    -Wl,--whole-archive
    MeshLibC2
    -Wl,--no-whole-archive
  )
ENDIF()

add_test(
  NAME ${PROJECT_NAME}
  COMMAND ${PROJECT_NAME}
)

IF(NOT WIN32)
  target_link_libraries( ${PROJECT_NAME} PRIVATE m )
ENDIF()
