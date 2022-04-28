cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (meshconv CXX)

IF (NOT MR_EMSCRIPTEN)
  find_package(Boost COMPONENTS program_options REQUIRED )
  if(Boost_PROGRAM_OPTIONS_FOUND)
    add_definitions( "-DCGAL_USE_BOOST_PROGRAM_OPTIONS" )
    link_libraries( ${Boost_PROGRAM_OPTIONS_LIBRARY} )
  endif()
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")
ENDIF()

add_executable(${PROJECT_NAME} meshconv.cpp)

IF(${distro} STREQUAL "Fedora Linux")
  target_link_libraries(${PROJECT_NAME} PRIVATE
          mrealgorithmspy
          MREAlgorithms
          MRMesh
          spdlog
          mrmeshpy
          Boost::boost
          )
ELSEIF (MR_EMSCRIPTEN)
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  target_link_libraries(${PROJECT_NAME} PRIVATE
          gtest
          -Wl,--whole-archive
          MREAlgorithms
          MRMesh
          -Wl,--no-whole-archive
          )
ELSE() # Ubuntu and macOS
  target_link_libraries(${PROJECT_NAME} PRIVATE
          mrealgorithmspy
          MREAlgorithms
          MRMesh
          fmt
          gtest
          spdlog
          mrmeshpy
          Boost::boost
          )
ENDIF()
