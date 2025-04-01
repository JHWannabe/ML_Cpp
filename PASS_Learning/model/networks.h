#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include "Resnet.h"
#include "de_Resnet.h"
#include "decoder.h"

void weights_init(nn::Module& m) {
	if ((typeid(m) == typeid(nn::Conv2d)) || (typeid(m) == typeid(nn::Conv2dImpl)) || (typeid(m) == typeid(nn::ConvTranspose2d)) || (typeid(m) == typeid(nn::ConvTranspose2dImpl))) {
		auto p = m.named_parameters(false);
		auto w = p.find("weight");
		auto b = p.find("bias");
		if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.02);
		if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
	}
	else if ((typeid(m) == typeid(nn::BatchNorm2d)) || (typeid(m) == typeid(nn::BatchNorm2dImpl))) {
		auto p = m.named_parameters(false);
		auto w = p.find("weight");
		auto b = p.find("bias");
		if (w != nullptr) nn::init::normal_(*w, /*mean=*/1.0, /*std=*/0.02);
		if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
	}
	return;
}

std::vector<torch::Tensor> diffmap(std::vector<torch::Tensor> fs_list, std::vector<torch::Tensor> ft_list)
{

	std::vector<torch::Tensor> diff_list = {};
	for (size_t i = 0; i < ft_list.size(); ++i) {
		torch::Tensor fs_norm = nn::functional::normalize(fs_list[i], nn::functional::NormalizeFuncOptions().p(2));
		torch::Tensor ft_norm = nn::functional::normalize(ft_list[i], nn::functional::NormalizeFuncOptions().p(2));

		torch::Tensor diff_features = nn::functional::mse_loss(fs_norm, ft_norm, nn::functional::MSELossFuncOptions().reduction(torch::kNone));
		diff_list.push_back(diff_features);
	}

	return diff_list;
}

class Supervised : public nn::Module
{
public:
	nn::ModuleHolder<ResNet<BasicBlock>> feature_extractor{ nullptr };
	nn::ModuleHolder<Decoder> decoder{ nullptr };
public:
	Supervised(mINI::INIStructure& ini)
	{
		int64_t width_per_group = 64;
		const std::vector<int64_t> layers{ 2, 2, 2, 2 };
		std::vector<int64_t> replace_stride_with_dilation = {};

		feature_extractor = std::make_shared<ResNet<BasicBlock>>(layers, 1000, false, 1, width_per_group, replace_stride_with_dilation);

		decoder = std::make_shared<Decoder>();

		feature_extractor = register_module("feature_extractor", feature_extractor);
		decoder = register_module("decoder", decoder);

	}


	torch::Tensor forward(torch::Tensor img)
	{
		std::vector<torch::Tensor> features = feature_extractor(img);
		torch::Tensor f_in = features[0];
		//torch::save(f_in, "libtorch_x.pt");
		torch::Tensor f_out = features[4];

		std::vector<torch::Tensor> f_ii = {};
		f_ii.push_back(features[1]);
		f_ii.push_back(features[2]);
		f_ii.push_back(features[3]);

		torch::Tensor predicted_mask = decoder(f_out, f_in, f_ii);

		return predicted_mask;
	}

};



class Unsupervised : public nn::Module
{
public:
	nn::ModuleHolder<ResNet<BasicBlock>> feature_extractor{ nullptr };
	nn::ModuleHolder<BN_layer<AttnBasicBlock>> RD4AD_bn{ nullptr };
	nn::ModuleHolder<deResNet<deBasicBlock>> RD4AD_decoder{ nullptr };
	nn::ModuleHolder<Decoder> decoder{ nullptr };

public:
	Unsupervised()
	{
		int64_t width_per_group = 64;
		const std::vector<int64_t> layers{ 2, 2, 2, 2 };
		std::vector<int64_t> replace_stride_with_dilation = {};

		feature_extractor = std::make_shared<ResNet<BasicBlock>>(layers, 1000, false, 1, width_per_group, replace_stride_with_dilation);
		RD4AD_bn = std::make_shared<BN_layer<AttnBasicBlock>>(2, 1000, width_per_group);
		RD4AD_decoder = std::make_shared<deResNet<deBasicBlock>>(layers, 1000, width_per_group);
		decoder = std::make_shared<Decoder>();

		feature_extractor = register_module("feature_extractor", feature_extractor);
		RD4AD_bn = register_module("RD4AD_bn", RD4AD_bn);
		RD4AD_decoder = register_module("RD4AD_decoder", RD4AD_decoder);
		decoder = register_module("decoder", decoder);
	}


	torch::Tensor forward(torch::Tensor normal_img)
	{
		std::vector<torch::Tensor> inputs = feature_extractor(normal_img);
		torch::Tensor f_in = inputs[0];
		torch::Tensor f_out = inputs[4];

		std::vector<torch::Tensor> inputs_ = {};
		inputs_.push_back(inputs[1]);
		inputs_.push_back(inputs[2]);
		inputs_.push_back(inputs[3]);

		torch::Tensor output1 = RD4AD_bn(inputs_);

		std::vector<torch::Tensor> RD4AD_outputs = RD4AD_decoder->forward(output1);

		std::vector<torch::Tensor> difference_features = diffmap(inputs_, RD4AD_outputs);

		torch::Tensor predicted_mask = decoder(f_out, f_in, difference_features);

		return predicted_mask;
	}
};