cmake_minimum_required(VERSION 3.10)

project(centerpool_plugin)

enable_language(CUDA)


include_directories(/usr/include /data/sunkx/TensorRT-10.4.0.26/include /usr/local/cuda/include)
link_directories(/data/sunkx/TensorRT-10.4.0.26/lib /usr/local/cuda/lib64)

add_library(centerpool_plugin SHARED centerpool_plugin.cpp centerpool_plugin.cu)

target_link_libraries(centerpool_plugin nvinfer cuda)
