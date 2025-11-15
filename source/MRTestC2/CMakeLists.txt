project(MRTestC2 C)

file(GLOB HEADERS "*.h")
file(GLOB SOURCES "*.c")

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ../MeshLibC2/include)

IF(NOT MR_EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MeshLibC2
  )
  IF(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE m)
  ENDIF()
ELSE()
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")

  target_link_libraries(${PROJECT_NAME} PRIVATE
    -Wl,--whole-archive
    MeshLibC2
    -Wl,--no-whole-archive
  )
ENDIF()

add_test(
  NAME ${PROJECT_NAME}
  COMMAND ${PROJECT_NAME}
)
