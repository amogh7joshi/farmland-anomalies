cmake_minimum_required(VERSION 3.17)
project(cc)

set(CMAKE_CXX_STANDARD 14)

# We need the OpenCV libraries from the system.
find_package( OpenCV 4.1.2 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(/usr/local/include/)

add_executable( cc generate.cpp)

target_link_libraries( cc ${OpenCV_LIBS} )