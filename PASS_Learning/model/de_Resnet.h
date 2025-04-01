#include <torch/torch.h>
#include <torch/script.h>
#include <c10/util/ArrayRef.h>
#include <iostream>

namespace nn = torch::nn;

nn::ConvTranspose2dOptions deconv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
    nn::ConvTranspose2dOptions deconv_options = nn::ConvTranspose2dOptions(in_planes, out_planes, kerner_size).stride(stride).bias(with_bias);
    //conv_options.stride_ = stride;
    //conv_options.padding_ = padding;
    //conv_options.with_bias_ = with_bias;
    return deconv_options;
}

nn::ConvTranspose2dOptions create_deconv2x2_options(int64_t in_planes,
    int64_t out_planes,
    int64_t stride = 1,
    int64_t groups = 1,
    int64_t dilation = 1)
{
    nn::ConvTranspose2dOptions deconv_options = nn::ConvTranspose2dOptions(in_planes, out_planes, 2).stride(stride).padding(0).bias(false).groups(groups).dilation(dilation);

    return deconv_options;
}

class deBasicBlock : public nn::Module {
public:
    static const int expansion;

    int64_t stride;
    nn::ConvTranspose2d conv1{ nullptr };
    nn::Conv2d conv1_1{ nullptr };

    nn::BatchNorm2d bn1{ nullptr };
    nn::Conv2d conv2{ nullptr };
    nn::BatchNorm2d bn2{ nullptr };
    nn::Sequential upsample{ nullptr };

    deBasicBlock(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
        nn::Sequential upsample_ = nn::Sequential(), int64_t base_width = 64, int64_t groups = 1)
    {
        if ((groups != 1) || (base_width != 64))
        {
            throw std::invalid_argument{
                "BasicBlock only supports groups=1 and base_width=64" };
        }

        if (stride_ == 2)
            conv1 = nn::ConvTranspose2d(create_deconv2x2_options(inplanes, planes, stride_, 1, 1));
        else
            conv1_1 = nn::Conv2d(create_conv3x3_options(inplanes, planes, stride_, 1, 1));
        bn1 = nn::BatchNorm2d(planes);
        conv2 = nn::Conv2d(create_conv3x3_options(planes, planes, 1, 1, 1));
        bn2 = nn::BatchNorm2d(planes);
        upsample = upsample_;
        stride = stride_;

        if (stride_ == 2)
            register_module("conv1", conv1);
        else
            register_module("conv1", conv1_1);

        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        stride = stride_;
        if (!upsample->is_empty()) {
            register_module("upsample", upsample);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        at::Tensor residual(x.clone());

        if (stride == 2)
            x = conv1->forward(x);
        else
            x = conv1_1->forward(x);

        x = bn1->forward(x);
        x = torch::relu(x);

        x = conv2->forward(x);
        x = bn2->forward(x);

        if (!upsample->is_empty()) {
            residual = upsample->forward(residual);
        }

        x += residual;
        x = torch::relu(x);

        return x;
    }
};

const int deBasicBlock::expansion = 1;


class deBottleneck : public nn::Module {
public:
    static const int expansion;

    int64_t stride;
    nn::Conv2d conv1{ nullptr };
    nn::BatchNorm2d bn1{ nullptr };
    nn::ConvTranspose2d conv2{ nullptr };
    nn::Conv2d conv2_1{ nullptr };

    nn::BatchNorm2d bn2{ nullptr };
    nn::Conv2d conv3{ nullptr };
    nn::BatchNorm2d bn3{ nullptr };
    nn::Sequential upsample{ nullptr };

    deBottleneck(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
        nn::Sequential upsample_ = nn::Sequential(), int64_t base_width = 64, int64_t groups = 1)
    {
        int64_t width = planes * (base_width / 64) * groups;
        int64_t dilation = 1;

        conv1 = nn::Conv2d(create_conv1x1_options(inplanes, width, 1));
        bn1 = nn::BatchNorm2d(width);
        if (stride_ == 2)
            conv2 = nn::ConvTranspose2d(create_deconv2x2_options(width, width, stride_, groups, dilation));
        else
            conv2_1 = nn::Conv2d(create_conv3x3_options(width, width, stride_, groups, dilation));
        bn2 = nn::BatchNorm2d(width);
        conv3 = nn::Conv2d(create_conv1x1_options(width, planes * expansion, 1));
        bn3 = nn::BatchNorm2d(planes * expansion);
        upsample = upsample_;

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        if (stride_ == 2)
            register_module("conv2", conv2);
        else
            register_module("conv2", conv2_1);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        stride = stride_;
        if (!upsample->is_empty()) {
            register_module("upsample", upsample);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        at::Tensor residual(x.clone());

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        if (stride == 2)
            x = conv2->forward(x);
        else
            x = conv2_1->forward(x);

        x = bn2->forward(x);
        x = torch::relu(x);

        x = conv3->forward(x);
        x = bn3->forward(x);

        if (!upsample->is_empty()) {
            residual = upsample->forward(residual);
        }

        x += residual;
        x = torch::relu(x);

        return x;
    }
};

const int deBottleneck::expansion = 4;


template <class Block>
class deResNet : public nn::Module {
public:
    int64_t inplanes = 64;
    nn::Sequential layer1{ nullptr };
    nn::Sequential layer2{ nullptr };
    nn::Sequential layer3{ nullptr };
    nn::Sequential layer4{ nullptr };
    int64_t m_base_width = 64;
    int64_t dilation = 1;

    torch::Tensor feature_a;
    torch::Tensor feature_b;
    torch::Tensor feature_c;

    deResNet(c10::ArrayRef<int64_t> layers, int64_t num_classes = 1000, int64_t width_per_group = 64) {
        m_base_width = width_per_group;
        inplanes = 512 * Block::expansion;
        dilation = 1;

        layer1 = _make_layer(256, layers[0], 2);
        layer2 = _make_layer(128, layers[1], 2);
        layer3 = _make_layer(64, layers[2], 2);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
    };

    std::vector<torch::Tensor> forward(torch::Tensor x) {
        feature_a = layer1->forward(x);  //feature_a
        feature_b = layer2->forward(feature_a);  //feature_b
        feature_c = layer3->forward(feature_b);  //feature_c

        std::vector<torch::Tensor> output = {};
        output.push_back(feature_c);
        output.push_back(feature_b);
        output.push_back(feature_a);

        return output;
    }

private:
    nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1)
    {
        nn::Sequential upsample;

        if (stride != 1 || inplanes != planes * Block::expansion)
        {
            upsample = nn::Sequential(
                nn::ConvTranspose2d(create_deconv2x2_options(inplanes, planes * Block::expansion, stride, 1, 1)),
                nn::BatchNorm2d(planes * Block::expansion)
            );
        }
        nn::Sequential layers;
        layers->push_back(Block(inplanes, planes, stride, upsample, m_base_width, 1));
        inplanes = planes * Block::expansion;
        for (int64_t i = 1; i < blocks; i++) {
            layers->push_back(Block(inplanes, planes, 1, nn::Sequential(), m_base_width, 1));
        }
        return layers;
    }
};