cmake_minimum_required(VERSION 3.10)

project(centerpool_infer)

set(CMAKE_CXX_STANDARD 14)


include_directories(/data/sunkx/TensorRT-10.4.0.26/include /usr/local/cuda/include)


link_directories(/data/sunkx/TensorRT-10.4.0.26/lib /usr/local/cuda/lib64 /data/sunkx/workspace/trt_centerpool_demo/plugin/build)

add_executable(infer infer.cpp)

target_link_libraries(infer nvinfer cudart -Wl,--no-as-needed centerpool_plugin -Wl,--as-needed)
