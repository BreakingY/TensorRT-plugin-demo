#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
using namespace nvinfer1;
class Logger : public ILogger
{

public:

    void log(Severity severity, const char* msg) noexcept override
    {

        if(severity <= Severity::kWARNING)
        {
            std::cout<< msg<< std::endl;
        }
    }

};
Logger gLogger;

std::vector<char> loadEngine(const std::string& file){

    std::ifstream f(file, std::ios::binary);

    f.seekg(0, f.end);

    size_t size = f.tellg();
    f.seekg(0, f.beg);

    std::vector<char> buffer(size);

    f.read(buffer.data(), size);
    return buffer;
}

int main(int argc, char** argv)
{
    if(argc < 2){
        std::cout << "./bin engine" << std::endl;
        return -1;
    }
	cudaSetDevice(2);
    std::cout << "start TRT inference" << std::endl;

    auto runtime = createInferRuntime(gLogger);
    if(!runtime){
        std::cout <<"runtime failed" << std::endl;
        return -1;
    }
    auto engineData =loadEngine(argv[1]);

    auto engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if(!engine)
    {
        std::cout << "deserialize engine failed" << std::endl;
        return -1;
    }
    std::cout << "engine load success" << std::endl;

    auto context = engine->createExecutionContext();
    if(!context)
    {
        std::cout << "context failed"<< std::endl;
        return -1;
    }

    int nb = engine->getNbIOTensors();
    std::cout<< "tensor num:" << nb << std::endl;
    for(int i=0;i<nb;i++)
    {
        const char* name = engine->getIOTensorName(i);

        auto mode = engine->getTensorIOMode(name);


        std::cout << name << " " << (mode==TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << std::endl;

    }

    int inputSize = 1*64*80*80;

    std::vector<float> input(inputSize);

    for(int i=0;i<inputSize;i++)
    {

        input[i]=0.5f;

    }

    int outputSize = 1*64;

    std::vector<float> output(outputSize);

    void* buffers[2];

    cudaMalloc(&buffers[0], inputSize*sizeof(float));

    cudaMalloc(&buffers[1], outputSize*sizeof(float));

    cudaMemcpy(buffers[0], input.data(), inputSize*sizeof(float), cudaMemcpyHostToDevice);


    context->setTensorAddress("input", buffers[0]);
    context->setTensorAddress("output",buffers[1]);


    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bool ret = context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    if(!ret)
    {
        std::cout << "enqueue failed" << std::endl;
        return -1;
    }

    std::cout << "inference success" << std::endl;

    cudaMemcpy(output.data(), buffers[1], outputSize*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "output:" << std::endl;

    for(int i=0;i<10;i++)
    {

        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(buffers[0]);
    cudaFree(buffers[1]);


    cudaStreamDestroy(stream);

    delete context;
    delete engine;
    delete runtime;
    return 0;

}
