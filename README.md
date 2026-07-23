# 自定义算子及TensorRT插件demo

1. 编译torch自定义算子
    * cd cuda_op
    * python setup.py build_ext --inplace
    * cd /data/sunkx/workspace/trt_centerpool_demo
2. 模型测试、导出onnx
    * python model_symbolic.py
3. TensorRT插件编译
    * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/sunkx/TensorRT-10.4.0.26/lib
    * export PATH=$PATH:/usr/local/cuda/bin
    * cd plugin
    * mkdir build
    * cd build
    * cmake ..
    * make -j
    * cd /data/sunkx/workspace/trt_centerpool_demo
3. 模型转换
    * /data/sunkx/TensorRT-10.4.0.26/bin/trtexec --onnx=./model_symbolic.onnx --plugins=./plugin/build/libcenterpool_plugin.so --saveEngine=model.engine --device=2
4. 模型推理
    * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/sunkx/workspace/trt_centerpool_demo/plugin/build
    * cd infer
    * mkdir build
    * cmake ..
    * make -j
    * ./infer ../../model.engine
