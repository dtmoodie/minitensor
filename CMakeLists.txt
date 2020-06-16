cmake_minimum_required(VERSION 3.5)

project(minitensor)


option(BUILD_TESTS ON "Build tests")

set(CMAKE_CXX_VERSION 11)
set(CMAKE_CXX_VERSION_REQUIRED ON)

add_library(minitensor
  INTERFACE
)

target_include_directories(minitensor
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)

export(TARGETS minitensor
    FILE minitensor-targets.cmake
)

install(TARGETS minitensor
      EXPORT minitensor
      DESTINATION lib
) # ignored

install(EXPORT minitensor FILE minitensor-config.cmake
    DESTINATION share/minitensor
)
install(DIRECTORY include/minitensor DESTINATION include)

if(BUILD_TESTS)
  ADD_SUBDIRECTORY (googletest)
  enable_testing()
  set(GTEST_INCLUDE googletest/googletest/include)

  file(GLOB_RECURSE test_src "tests/*.cpp")

  add_executable(test_minitensor ${test_src})
  target_link_libraries(test_minitensor minitensor gtest gtest_main)
  target_include_directories(test_minitensor SYSTEM PRIVATE ${GTEST_INCLUDE})
endif(BUILD_TESTS)



