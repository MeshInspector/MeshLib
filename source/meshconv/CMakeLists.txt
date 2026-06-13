project(meshconv CXX)

find_package(Boost CONFIG COMPONENTS program_options REQUIRED)

add_executable(${PROJECT_NAME} meshconv.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  MRIOExtras
  Boost::program_options
)

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
  IF(MR_PCH_CODEGEN)
    mr_attach_pch_object(${PROJECT_NAME})
  ENDIF()
ENDIF()

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION "${MR_BIN_DIR}"
)
