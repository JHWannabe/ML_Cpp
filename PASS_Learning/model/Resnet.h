#include <torch/torch.h>
#include <torch/script.h>
#include <c10/util/ArrayRef.h>
#include <iostream>

namespace nn = torch::nn;

nn::Conv2dOptions conv_options_legacy(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 1, bool with_bias = false) {
    nn::Conv2dOptions conv_options = nn::Conv2dOptions(in_planes, out_planes, kerner_size).stride(stride).padding(padding).bias(with_bias);
    //conv_options.stride_ = stride;
    //conv_options.padding_ = padding;
    //conv_options.with_bias_ = with_bias;
    return conv_options;
}

nn::Conv2dOptions
create_conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, int64_t groups = 1,
    int64_t dilation = 1, bool bias = false)
{
    nn::Conv2dOptions conv_options =
        nn::Conv2dOptions(in_planes, out_planes, kerner_size)
        .stride(stride)
        .padding(padding)
        .bias(bias)
        .groups(groups)
        .dilation(dilation);

    return conv_options;
}

nn::Conv2dOptions create_conv3x3_options(int64_t in_planes,
    int64_t out_planes,
    int64_t stride = 1,
    int64_t groups = 1,
    int64_t dilation = 1)
{
    nn::Conv2dOptions conv_options = create_conv_options(
        in_planes, out_planes, /*kerner_size = */ 3, stride,
        /*padding = */ dilation, groups, /*dilation = */ dilation, false);
    return conv_options;
}

nn::Conv2dOptions create_conv1x1_options(int64_t in_planes,
    int64_t out_planes,
    int64_t stride = 1)
{
    nn::Conv2dOptions conv_options = create_conv_options(
        in_planes, out_planes,
        /*kerner_size = */ 1, stride,
        /*padding = */ 0, /*groups = */ 1, /*dilation = */ 1, false);
    return conv_options;
}


class BasicBlock : public nn::Module
{
public:
    static const int expansion;

    int64_t stride;
    nn::Conv2d conv1{ nullptr };
    nn::BatchNorm2d bn1{ nullptr };
    nn::Conv2d conv2{ nullptr };
    nn::BatchNorm2d bn2{ nullptr };
    nn::Sequential m_downsample = nn::Sequential();
    nn::ReLU relu{ nullptr };

    BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
        nn::Sequential downsample = nn::Sequential(), int64_t groups = 1, int64_t base_width = 64, int64_t dilation = 1)
    {
        if ((groups != 1) || (base_width != 64))
        {
            throw std::invalid_argument{
                "BasicBlock only supports groups=1 and base_width=64" };
        }

        conv1 = nn::Conv2d(create_conv3x3_options(inplanes, planes, stride));
        bn1 = nn::BatchNorm2d(planes);
        relu = nn::ReLU(true);
        conv2 = nn::Conv2d(create_conv3x3_options(planes, planes));
        bn2 = nn::BatchNorm2d(planes);

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        if (!downsample->is_empty()) {
            m_downsample = register_module("downsample", downsample);
        }
        register_module("relu", relu);
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor identity = x;

        torch::Tensor out = conv1->forward(x);
        out = bn1->forward(out);
        out = relu->forward(out);

        out = conv2->forward(out);
        out = bn2->forward(out);

        if (!m_downsample->is_empty()) {
            identity = m_downsample->forward(x);
        }

        out += identity;
        out = relu->forward(out);

        return out;
    }
};

const int BasicBlock::expansion = 1;


class Bottleneck : public nn::Module {
public:
    static const int expansion;

    int64_t stride;
    nn::Conv2d conv1{ nullptr };
    nn::BatchNorm2d bn1{ nullptr };
    nn::Conv2d conv2{ nullptr };
    nn::BatchNorm2d bn2{ nullptr };
    nn::Conv2d conv3{ nullptr };
    nn::BatchNorm2d bn3{ nullptr };
    nn::Sequential m_downsample = nn::Sequential();
    nn::ReLU relu{ nullptr };

    Bottleneck(int64_t inplanes, int64_t planes, int64_t stride = 1,
        nn::Sequential downsample = nn::Sequential(),
        int64_t groups = 1, int64_t base_width = 64,
        int64_t dilation = 1)
    {
        int64_t width = planes * (base_width / 64) * groups;

        conv1 = nn::Conv2d(create_conv1x1_options(inplanes, width));
        bn1 = nn::BatchNorm2d(width);

        conv2 = nn::Conv2d(create_conv3x3_options(width, width, stride, groups, dilation));
        bn2 = nn::BatchNorm2d(width);

        conv3 = nn::Conv2d(create_conv1x1_options(width, planes * expansion));
        bn3 = nn::BatchNorm2d(planes * expansion);

        relu = nn::ReLU(true);

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        register_module("relu", relu);

        if (!downsample->is_empty())
        {
            m_downsample = register_module("downsample", downsample);
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor identity = x;
        torch::Tensor out = conv1->forward(x);

        out = bn1->forward(out);
        out = relu->forward(out);

        out = conv2->forward(out);
        out = bn2->forward(out);

        out = relu->forward(out);

        out = conv3->forward(out);
        out = bn3->forward(out);

        if (!m_downsample->is_empty())
        {
            identity = m_downsample->forward(x);
        }
        out += identity;
        out = relu->forward(out);

        return out;
    }
};

const int Bottleneck::expansion = 4;


template <class Block>
class ResNet : public nn::Module {

public:
    nn::Conv2d conv1{ nullptr };
    nn::BatchNorm2d bn1{ nullptr };
    nn::Sequential layer1{ nullptr };
    nn::Sequential layer2{ nullptr };
    nn::Sequential layer3{ nullptr };
    nn::Sequential layer4{ nullptr };
    nn::ReLU relu{ nullptr };
    nn::MaxPool2d maxpool{ nullptr };
    nn::AdaptiveAvgPool2d avgpool{ nullptr };
    nn::Linear fc{ nullptr };

    int64_t inplanes = 64;
    int64_t m_dilation = 1;
    int64_t m_groups = 1;
    int64_t m_base_width = 64;

    torch::Tensor feature_a;
    torch::Tensor feature_b;
    torch::Tensor feature_c;
    torch::Tensor feature_d;

    ResNet(const std::vector<int64_t> layers, int64_t num_classes = 1000,
        bool zero_init_residual = false, int64_t groups = 1,
        int64_t width_per_group = 64,
        std::vector<int64_t> replace_stride_with_dilation = {})
    {

        if (replace_stride_with_dilation.size() == 0)
        {
            // Each element in the tuple indicates if we should replace
            // the 2x2 stride with a dilated convolution instead.
            replace_stride_with_dilation = { false, false, false };
        }
        if (replace_stride_with_dilation.size() != 3)
        {
            throw std::invalid_argument{
                "replace_stride_with_dilation should be empty or have exactly "
                "three elements." };
        }

        m_groups = groups;
        m_base_width = width_per_group;

        conv1 = nn::Conv2d(create_conv_options(
            /*in_planes = */ 3, /*out_planes = */ inplanes,
            /*kerner_size = */ 7, /*stride = */ 2, /*padding = */ 3,
            /*groups = */ 1, /*dilation = */ 1, /*bias = */ false));

        bn1 = nn::BatchNorm2d(inplanes);
        relu = nn::ReLU(true);
        maxpool = nn::MaxPool2d(nn::MaxPool2dOptions({ 3, 3 }).stride({ 2, 2 }).padding({ 1, 1 }));
        avgpool = nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({ 1, 1 }));
        fc = nn::Linear(512 * Block::expansion, num_classes);

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        layer1 = register_module("layer1", _make_layer(64, layers.at(0)));
        layer2 = register_module(
            "layer2", _make_layer(128, layers.at(1), 2,
                replace_stride_with_dilation.at(0)));
        layer3 = register_module(
            "layer3", _make_layer(256, layers.at(2), 2,
                replace_stride_with_dilation.at(1)));
        layer4 = register_module(
            "layer4", _make_layer(512, layers.at(3), 2,
                replace_stride_with_dilation.at(2)));
        register_module("relu", relu);
        register_module("maxpool", maxpool);
        register_module("avgpool", avgpool);
        register_module("fc", fc);
    };

    std::vector<torch::Tensor> forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = bn1->forward(x);
        std::vector<torch::Tensor> output = {};
        x = relu->forward(x);
        output.push_back(x);  // act1 : 0
        x = maxpool->forward(x);

        feature_a = layer1->forward(x);
        feature_b = layer2->forward(feature_a);
        feature_c = layer3->forward(feature_b);
        feature_d = layer4->forward(feature_c);

        output.push_back(feature_a);  // layer1 : 1
        output.push_back(feature_b);  // layer2 : 2
        output.push_back(feature_c);  // layer3 : 3
        output.push_back(feature_d);  // layer4 : 4

        return output;
    }

private:
    nn::Sequential _make_layer(int64_t planes, int64_t blocks,
        int64_t stride = 1, bool dilate = false)
    {
        nn::Sequential downsample = nn::Sequential();
        int64_t previous_dilation = m_dilation;
        if (dilate)
        {
            m_dilation *= stride;
            stride = 1;
        }
        int64_t temp = Block::expansion;
        if ((stride != 1) || (inplanes != planes * Block::expansion))
        {
            downsample = nn::Sequential(
                nn::Conv2d(create_conv1x1_options(
                    inplanes, planes * Block::expansion, stride)),
                nn::BatchNorm2d(planes * Block::expansion));

        }

        nn::Sequential layers;

        layers->push_back(Block(inplanes, planes, stride, downsample,
            m_groups, m_base_width, previous_dilation));
        inplanes = planes * Block::expansion;
        for (int64_t i = 1; i < blocks; i++)
        {
            layers->push_back(Block(inplanes, planes, 1,
                nn::Sequential(), m_groups,
                m_base_width, m_dilation));
        }

        return layers;
    }
};


class AttnBasicBlock : public nn::Module {
public:
    static const int expansion;

    int64_t stride;
    nn::Conv2d conv1{ nullptr };
    nn::BatchNorm2d bn1{ nullptr };
    nn::Conv2d conv2{ nullptr };
    nn::BatchNorm2d bn2{ nullptr };
    nn::Sequential downsample{ nullptr };

    AttnBasicBlock(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
        nn::Sequential downsample_ = nn::Sequential(), int64_t base_width = 64, int64_t groups = 1)
    {
        if ((groups != 1) || (base_width != 64))
        {
            throw std::invalid_argument{
                "BasicBlock only supports groups=1 and base_width=64" };
        }

        conv1 = nn::Conv2d(create_conv3x3_options(inplanes, planes, stride_));
        bn1 = nn::BatchNorm2d(planes);
        conv2 = nn::Conv2d(create_conv3x3_options(planes, planes));
        bn2 = nn::BatchNorm2d(planes);
        downsample = downsample_;

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);

        if (!downsample->is_empty()) {
            register_module("downsample", downsample);
        }
    }

    torch::Tensor forward(torch::Tensor x) {

        torch::Tensor identity = x;

        torch::Tensor out = conv1->forward(x);
        out = bn1->forward(out);
        out = torch::relu(out);

        out = conv2->forward(out);
        out = bn2->forward(out);

        if (!downsample->is_empty()) {
            identity = downsample->forward(x);
        }

        out += identity;
        out = torch::relu(out);

        return out;
    }
};

const int AttnBasicBlock::expansion = 1;



class AttnBottleneck : public nn::Module {
public:

    static const int expansion;

    int64_t stride;
    nn::Conv2d conv1{ nullptr };
    nn::BatchNorm2d bn1{ nullptr };
    nn::Conv2d conv2{ nullptr };
    nn::BatchNorm2d bn2{ nullptr };
    nn::Conv2d conv3{ nullptr };
    nn::BatchNorm2d bn3{ nullptr };
    nn::Sequential downsample{ nullptr };

    AttnBottleneck(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
        nn::Sequential downsample_ = nn::Sequential(), int64_t base_width = 64, int64_t groups = 1)
    {
        int64_t width = planes * (base_width / 64) * groups;
        int64_t dilation = 1;

        conv1 = nn::Conv2d(create_conv1x1_options(inplanes, width));
        bn1 = nn::BatchNorm2d(width);
        conv2 = nn::Conv2d(create_conv3x3_options(width, width, stride_, groups, dilation));
        bn2 = nn::BatchNorm2d(width);
        conv3 = nn::Conv2d(create_conv1x1_options(width, planes * expansion));
        bn3 = nn::BatchNorm2d(planes * expansion);
        downsample = downsample_;

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        stride = stride_;
        if (!downsample->is_empty()) {
            register_module("downsample", downsample);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor identity = x;

        torch::Tensor out = conv1->forward(x);
        out = bn1->forward(out);
        out = torch::relu(out);
        out = conv2->forward(out);
        out = bn2->forward(out);
        out = torch::relu(out);

        out = conv3->forward(out);
        out = bn3->forward(out);

        if (!downsample->is_empty()) {
            identity = downsample->forward(x);
        }

        out += identity;
        out = torch::relu(out);

        return out;
    }
};

const int AttnBottleneck::expansion = 4;


template <class Block>
class BN_layer : public nn::Module {
public:
    int64_t inplanes = 64;
    int64_t dilation = 1;
    nn::Conv2d conv1{ nullptr };
    nn::Conv2d conv2{ nullptr };
    nn::Conv2d conv3{ nullptr };
    nn::Conv2d conv4{ nullptr };

    nn::BatchNorm2d bn1{ nullptr };
    nn::BatchNorm2d bn2{ nullptr };
    nn::BatchNorm2d bn3{ nullptr };
    nn::BatchNorm2d bn4{ nullptr };

    nn::Sequential bn_layer{ nullptr };

    int64_t m_base_width = 64;

    BN_layer(int64_t layers, int64_t num_classes = 1000, int64_t width_per_group = 64) {
        m_base_width = width_per_group;
        inplanes = 256 * Block::expansion;
        dilation = 1;
        bn_layer = _make_layer(512, layers, 2);

        conv1 = nn::Conv2d(create_conv3x3_options(64 * Block::expansion, 128 * Block::expansion, 2, 1, 1));
        bn1 = nn::BatchNorm2d(128 * Block::expansion);
        conv2 = nn::Conv2d(create_conv3x3_options(128 * Block::expansion, 256 * Block::expansion, 2, 1, 1));
        bn2 = nn::BatchNorm2d(256 * Block::expansion);
        conv3 = nn::Conv2d(create_conv3x3_options(128 * Block::expansion, 256 * Block::expansion, 2, 1, 1));
        bn3 = nn::BatchNorm2d(256 * Block::expansion);
        conv4 = nn::Conv2d(create_conv1x1_options(1024 * Block::expansion, 512 * Block::expansion, 1));
        bn4 = nn::BatchNorm2d(512 * Block::expansion);

        register_module("bn_layer", bn_layer);
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        register_module("conv4", conv4);
        register_module("bn4", bn4);
    };

    torch::Tensor forward(std::vector<torch::Tensor> x) {
        torch::Tensor temp = conv1->forward(x[0]);
        temp = bn1->forward(temp);
        temp = torch::relu(temp);
        temp = conv2->forward(temp);
        temp = bn2->forward(temp);
        torch::Tensor l1 = torch::relu(temp);

        temp = conv3->forward(x[1]);
        temp = bn3->forward(temp);
        torch::Tensor l2 = torch::relu(temp);

        torch::Tensor feature = torch::cat({ l1,l2,x[2] }, 1);
        torch::Tensor output = bn_layer->forward(feature);

        return output.contiguous();
    }

private:
    nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1)
    {
        nn::Sequential downsample;

        if (stride != 1 || inplanes != planes * Block::expansion)
        {
            downsample = nn::Sequential(
                nn::Conv2d(create_conv1x1_options(inplanes * 3, planes * Block::expansion, stride)),
                nn::BatchNorm2d(planes * Block::expansion)
            );
        }
        nn::Sequential layers;
        layers->push_back(Block(inplanes * 3, planes, stride, downsample, m_base_width, 1));
        inplanes = planes * Block::expansion;
        for (int64_t i = 1; i < blocks; i++) {
            layers->push_back(Block(inplanes, planes, 1, nn::Sequential(), m_base_width, 1));
        }
        return layers;
    }
};