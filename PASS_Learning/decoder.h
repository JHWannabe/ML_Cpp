#include <torch/torch.h>
#include <torch/script.h>

class UpConvBlock : public torch::nn::Module
{
	torch::nn::Sequential blk{ nullptr };
public:
	UpConvBlock(int in_channel, int out_channel)
	{
		torch::nn::Sequential blk_(
			torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2,2 })).mode(torch::kBilinear).align_corners(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, out_channel, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channel)),
			torch::nn::ReLU(torch::nn::ReLUOptions(false))
		);
		blk = register_module("blk", blk_.ptr());
	}

	torch::Tensor forward(torch::Tensor x)
	{
		return blk->forward(x);
	}
};

class Decoder : public torch::nn::Module
{
	torch::nn::Conv2d conv{ nullptr };
	torch::nn::ModuleHolder<UpConvBlock> upconv3{ nullptr };
	torch::nn::ModuleHolder<UpConvBlock> upconv2{ nullptr };
	torch::nn::ModuleHolder<UpConvBlock> upconv1{ nullptr };
	torch::nn::ModuleHolder<UpConvBlock> upconv0{ nullptr };
	torch::nn::ModuleHolder<UpConvBlock> upconv2mask{ nullptr };
	torch::nn::Conv2d final_conv{ nullptr };
public:
	Decoder()
	{
		conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 48, 3).stride(1).padding(1));
		upconv3 = std::make_shared<UpConvBlock>(512, 256);
		upconv2 = std::make_shared<UpConvBlock>(512, 128);
		upconv1 = std::make_shared<UpConvBlock>(256, 64);
		upconv0 = std::make_shared<UpConvBlock>(128, 48);
		upconv2mask = std::make_shared<UpConvBlock>(96, 48);
		final_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(48, 1, 3).stride(1).padding(1));

		register_module("conv", conv);
		register_module("upconv3", upconv3);
		register_module("upconv2", upconv2);
		register_module("upconv1", upconv1);
		register_module("upconv0", upconv0);
		register_module("upconv2mask", upconv2mask);
		register_module("final_conv", final_conv);
	}


	torch::Tensor forward(torch::Tensor encoder_output, torch::Tensor f_in, std::vector<torch::Tensor> concat_features)
	{
		torch::Tensor f0 = f_in;
		torch::Tensor f1 = concat_features[0];
		torch::Tensor f2 = concat_features[1];
		torch::Tensor f3 = concat_features[2];

		torch::Tensor x_up3 = upconv3(encoder_output);
		x_up3 = torch::cat({ x_up3 ,f3 }, 1);

		torch::Tensor x_up2 = upconv2(x_up3);
		x_up2 = torch::cat({ x_up2 ,f2 }, 1);

		torch::Tensor x_up1 = upconv1(x_up2);
		x_up1 = torch::cat({ x_up1 ,f1 }, 1);

		torch::Tensor x_up0 = upconv0(x_up1);
		f0 = conv(f0);
		torch::Tensor x_up2mask = torch::cat({ x_up0 ,f0 }, 1);

		torch::Tensor x_mask = upconv2mask(x_up2mask);
		x_mask = final_conv(x_mask);

		return x_mask;
	}
};