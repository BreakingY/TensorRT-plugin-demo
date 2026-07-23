#include <torch/extension.h>


void launch_centerpool(float* input, float* output, int C, int H, int W);

torch::Tensor centerpool(torch::Tensor input){

    int C=input.size(1);
    int H=input.size(2);
    int W=input.size(3);

    auto output =torch::zeros({1,C}, input.options());

    launch_centerpool(input.data_ptr<float>(), output.data_ptr<float>(), C,H,W);

    return output;
}



PYBIND11_MODULE(centerpool_cuda, m){
    m.def("centerpool",centerpool);

}
