#include "NvInfer.h"
#include "NvInferPlugin.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <iostream>
#include <string>

using namespace nvinfer1;
/*
ONNX
 | 发现自定义算子 CenterPool
 v
Plugin Registry 查找
 |
 v
CenterPoolPluginCreator
 |-- createPlugin()
 v
CenterPoolPlugin
 |-- getNbOutputs()
 |-- getOutputDimensions()
 |-- getOutputDataType()
 |-- supportsFormatCombination()
 |-- configurePlugin()
 |-- getWorkspaceSize()
 v
serialize()
 |
 v
保存 engine
============================
加载 engine
 |
 v
CenterPoolPluginCreator
 |-- deserializePlugin()
 v
CenterPoolPlugin
 |-- initialize()
 v
推理执行
 |-- enqueue()
 v
CUDA Kernel
*/

// CUDA kernel入口
extern "C"
void launch_centerpool(const float* input, float* output, int C, int H, int W, cudaStream_t stream);

/**************************************************
 *
 * CenterPool Plugin
 *
 **************************************************/
class CenterPoolPlugin : public IPluginV2DynamicExt
{
public:
    CenterPoolPlugin()
    {
    }

    // 反序列化构造
    CenterPoolPlugin(const void* data, size_t length)
    {
    }

    /******** Plugin基本信息 ********/
    const char* getPluginType() const noexcept override
    {
        return "CenterPool";
    }


    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    int getNbOutputs() const noexcept override
    {
        return 1;
    }

    /******** 输出shape ********/
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        DimsExprs output;

        // 输入:
        // N,C,H,W inputs: CenterPool算子的所有输入(>=1); nbInputs:输入格式
        //
        // 输出:
        // N,C

        output.nbDims = 2;

        output.d[0] = inputs[0].d[0];

        output.d[1] = inputs[0].d[1];

        return output;
    }

    /******** 数据类型 ********/

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        return DataType::kFLOAT;
    }

    /******** 格式支持 ********/

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override
    {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    /******** 配置 ********/

    void configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs, DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
    }

    /******** workspace ********/
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        return 0;
    }

    /******** 核心执行 ********/
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        /*
        input:
        
        [N,C,H,W]
        
        */
        int C = inputDesc[0].dims.d[1];
        int H = inputDesc[0].dims.d[2];
        int W = inputDesc[0].dims.d[3];
        launch_centerpool(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), C, H, W, stream);
        return 0;
    }

    /******** 生命周期 ********/
    int initialize() noexcept override
    {
        return 0;
    }

    void terminate() noexcept override
    {
    }

    void destroy() noexcept override
    {
        delete this;
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        return new CenterPoolPlugin();
    }

    /******** 序列化 ********/
    size_t getSerializationSize() const noexcept override
    {
        return 0;
    }

    void serialize(void* buffer) const noexcept override
    {
    }

    /******** namespace ********/

    void setPluginNamespace(const char* pluginNamespace) noexcept override
    {
    }
    const char* getPluginNamespace() const noexcept override
    {
        return "";
    }
};

/**************************************************
 *
 * Creator
 *
 **************************************************/

class CenterPoolPluginCreator : public IPluginCreator
{
public:
    CenterPoolPluginCreator()
    {
        mFC.nbFields = 0;
        mFC.fields = nullptr;
    }
    const char* getPluginName() const noexcept override
    {
        return "CenterPool";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    const PluginFieldCollection*getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        return new CenterPoolPlugin();
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        return new CenterPoolPlugin(serialData, serialLength);
    }

    void setPluginNamespace( const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    PluginFieldCollection mFC{};
    std::string mNamespace;
};

/**************************************************
 *
 * 注册 Plugin
 *
 **************************************************/
REGISTER_TENSORRT_PLUGIN(
    CenterPoolPluginCreator
);