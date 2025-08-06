cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# On Mac, strip `-stdlib=libc++` from linker flags to avoid the `unused flag` warning. Unused because this is a C project.
IF (APPLE)
  string(REPLACE "-stdlib=libc++" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
ENDIF()

project(MRTestC C)

file(GLOB HEADERS "*.h")
file(GLOB SOURCES "*.c")

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})

IF(MR_EMSCRIPTEN)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")
ENDIF()

IF(MR_EMSCRIPTEN)
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  target_link_libraries(${PROJECT_NAME} PRIVATE
    gtest
    -Wl,--whole-archive
    MRMeshC
    -Wl,--no-whole-archive
  )
ELSEIF(WIN32)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMeshC
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMeshC
    m
  )
ENDIF()

add_test(
  NAME ${PROJECT_NAME}
  COMMAND ${PROJECT_NAME}
)
