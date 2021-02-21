cmake_minimum_required(VERSION 3.17)
project(cc)

set(CMAKE_CXX_STANDARD 14)

# We need the OpenCV libraries from the system.
find_package( OpenCV 4.1.2 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(/usr/local/include/)
include_directories(/usr/local/Cellar)

# We also need the nlohmann/json library from the system.
find_package(nlohmann_json REQUIRED)

add_executable( cc generate.cc )

target_link_libraries( cc ${OpenCV_LIBS} )
target_link_libraries( cc nlohmann_json::nlohmann_json )