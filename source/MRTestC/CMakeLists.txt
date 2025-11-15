project(MRTestC C)

file(GLOB HEADERS "*.h")
file(GLOB SOURCES "*.c")

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})

IF(EMSCRIPTEN)
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")

  target_link_libraries(${PROJECT_NAME} PRIVATE
    gtest
    -Wl,--whole-archive
    MRMeshC
    -Wl,--no-whole-archive
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMeshC
  )
  IF(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE m)
  ENDIF()
ENDIF()

add_test(
  NAME ${PROJECT_NAME}
  COMMAND ${PROJECT_NAME}
)
