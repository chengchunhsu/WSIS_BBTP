cmake_minimum_required (VERSION 2.6)
project(probimage)

add_library( probimage probimage.cpp )

add_executable( compress compress.cpp )
target_link_libraries( compress probimage )

add_executable( decompress decompress.cpp )
target_link_libraries( decompress probimage )
