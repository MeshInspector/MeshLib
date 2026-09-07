project(MRTestCuda CXX)

file(GLOB SOURCES "*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

find_package(GTest REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  MRCuda
  GTest::gtest
)

add_test(
  NAME ${PROJECT_NAME}
  COMMAND ${PROJECT_NAME}
)

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
