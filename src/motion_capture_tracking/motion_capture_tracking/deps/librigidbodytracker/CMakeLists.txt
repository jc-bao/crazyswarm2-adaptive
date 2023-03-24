cmake_minimum_required(VERSION 3.5)
project(librigidbodytracker)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL REQUIRED)
# find_package(PkgConfig)
# pkg_check_modules(YamlCpp yaml-cpp)

###########
## Build ##
###########

## Additional include folders
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

## Declare a cpp library
add_library(librigidbodytracker
  src/rigid_body_tracker.cpp
)
target_link_libraries(librigidbodytracker
  ${PCL_LIBRARIES}
)

# add_executable(playclouds
#   src/playclouds.cpp
# )
# target_link_libraries(playclouds
#   librigidbodytracker
#   ${PCL_LIBRARIES}
#   yaml-cpp
# )

# add_executable(profile
#   src/profile.cpp
# )
# target_link_libraries(profile
#   ${PCL_LIBRARIES}
# )