
#include <string>
#include <iostream>
#include <cmath>

#include <boltview/host_image.h>
#include <boltview/array_view.h>
#include <boltview/device_image.h>
#include <boltview/unified_image.h>
#include <boltview/texture_image.h>
#include <boltview/create_view.h>
#include <boltview/image_io.h>
#include <boltview/loop_utils.h>
#include <boltview/reduce.h>
#include <boltview/for_each.h>
#include <boltview/subview.h>
#include <boltview/geometrical_transformation.h>
#include <boltview/interpolated_view.h>
#include "io.h"

using namespace bolt;

template<typename TType, int tDim>
// using Image = UnifiedImage<TType, tDim>;
using Image = HostImage<TType, tDim>;

using RGBAf = Vector<float, 4>;
using RGBA8 = Vector<uint8_t, 4>;

std::vector<RGBAf> readTransferFunction(boost::filesystem::path file) {
	std::ifstream f;
	f.exceptions ( std::ifstream::badbit );
	f.open(file.string());

	std::vector<RGBAf> tf;
	tf.reserve(4096);

	int v;
	float r, g, b, a;
	while (f >> v >> r >> g >> b >> a) {
		tf.emplace_back(r, g, b, a);
		// std::cout << tf.back() << "\n";
	}
	return tf;
}

template<typename TView>
BOLT_DECL_DEVICE
Float3 computeGradient(TView volume, float value, Float3 current_position, float epsilon) {
	return {
		(value - volume.access(current_position - Float3(epsilon, 0.0f, 0.0f))) / epsilon,
		(value - volume.access(current_position - Float3(0.0f, epsilon, 0.0f))) / epsilon,
		(value - volume.access(current_position - Float3(0.0f, 0.0f, epsilon))) / epsilon
	};
}

BOLT_DECL_DEVICE
RGBAf frontToBackBlend(RGBAf current, RGBAf sample, float step_size) {
	auto sample_alpha = 1.0f - pow(1.0f - sample[3], step_size);
	auto sample_rgb = swizzle<0, 1, 2>(sample);

	auto current_alpha = current[3];
	auto current_rgb = swizzle<0, 1, 2>(current);

	current_rgb   = current_rgb   + sample_alpha * (1 - current_alpha) * sample_rgb;
	current_alpha = current_alpha + sample_alpha * (1 - current_alpha);

	return RGBAf(current_rgb, current_alpha);
}

struct Material
{
	Float3	Ka;
	Float3	Kd;
	Float3	Ks;
	float	shininess;
};

struct Light {
	Float3 color;
	Float3 ambient;
	Float3 position;
};

BOLT_DECL_DEVICE
Float3 lit(float NdotL, float NdotH, float m)
{
  float specular = (NdotL > 0) ? pow(max(0.0f, NdotH), m) : 0.0f;
  return {1.0f, max(0.0f, NdotL), specular};
}


BOLT_DECL_DEVICE
Float3 blinnPhongShading(Float3 N, Float3 V, Float3 L, Material material, Light light)
{
	//half way vector
	Float3 H = normalize( L + V );

	//compute ambient term
	Float3 ambient = product(material.Ka, light.ambient);

	Float3 koef = lit(dot(N, L), dot(N, H), material.shininess);
	Float3 diffuse = koef[1] * product(material.Kd, light.color);
	Float3 specular = koef[2] * product(material.Ks, light.color);

	return ambient + diffuse + specular;
}

BOLT_DECL_DEVICE
RGBAf doShading(
	Float3 aPosition,
	RGBAf color,
	Light aLight,
	Material material,
	Float3 aEyePosition,
	Float3 gradient
	)
{
	Float3 N = normalize( gradient );

	Float3 L = normalize( aLight.position - aPosition );
	Float3 V = normalize( aEyePosition - aPosition );

	auto rgb = swizzle<0,1,2>(color);
	rgb += blinnPhongShading(N, V, L, material, aLight);
	return RGBAf(rgb, color[3]);
}


struct Camera {
	Float3 lookat;
	Float3 eye;
	Float3 up;
	float fov = 50.0f;
};


template<typename TView, typename TTransFunction>
struct RayTraversal {
	static constexpr int kSampleCount = 25000;
	static constexpr float kGradientEpsilon = 0.2f;

	RayTraversal(TView data, TTransFunction tf, Int2 resolution, Camera camera) :
		volume_(data),
		tf_(tf),
		resolution_(resolution),
		eye_(camera.eye),
		lookat_(camera.lookat),
		up_(camera.up)
	{
		auto look_dir = normalize(lookat_ - eye_);
		plane_center_ = eye_ + look_dir;
		auto binorm = normalize(cross(look_dir, up_));
		// up_ = cross(look_dir, binorm);
		up_ = cross(binorm, look_dir);

		float factor = tan(camera.fov/2 * kPi / 180.0f) / (resolution[0] / 2.0f);
		x_step_ = factor * binorm;
		y_step_ = factor * up_;

		std::cout << "Lookat " << lookat_ << "\n";
		std::cout << "Up " << up_ << "\n";
		std::cout << "X " << x_step_ << "\n";
		std::cout << "Y " << y_step_ << "\n";
		std::cout << "C " << plane_center_ << "\n";
		std::cout << "Eye " << eye_ << "\n";


		material_.Ka = Float3(0.1f, 0.1f, 0.1f);
		material_.Kd = Float3(0.6f, 0.6f, 0.6f);
		material_.Ks = Float3(0.2f, 0.2f, 0.2f);
		material_.shininess = 100;
	}

	BOLT_DECL_DEVICE
	Float3 computeStep(Int2 coords) const {
		Float2 plane_pos = coords - (0.5f * resolution_);
		Float3 pos = plane_center_ + plane_pos[0] * x_step_ + plane_pos[1] * y_step_;
		return normalize(pos - eye_);
	}

	BOLT_DECL_DEVICE
	void operator()(RGBAf &color, Int2 coords) const {
		Float3 step = step_size_ * computeStep(coords);
		Float3 current_position = eye_ + 800*step;
		RGBAf res;
		for (int i = 0; i < kSampleCount; ++i) {
			// auto sample = clamp((volume_.access(current_position) - 1100.0f) /3500.0f, 0.0f, 1.0f);
			auto sample = volume_.access(current_position);
			auto gradient = computeGradient(volume_, sample, current_position, kGradientEpsilon);
			RGBAf color = tf_[round(sample)];
			color = doShading(
				current_position,
				color,
				light_,
				material_,
				eye_,
				gradient);
			// RGBAf color{ 1.0f, 1.0f, 1.0f, sample };

			res = frontToBackBlend(res, color, step_size_);
			current_position += step;
		}
		// color = {1.0f, 1.0f, 1.0f, res[3]};
		color = res;
	}

	TView volume_;
	TTransFunction tf_;
	Int2 resolution_;
	Float3 eye_;
	Float3 lookat_;
	Float3 up_;

	Float3 plane_center_;
	Float3 x_step_;
	Float3 y_step_;
	float step_size_ = 0.1f;
	Material material_;
	Light light_ = { Float3(1.0f, FillTag{}), Float3(0.1f, FillTag{}), Float3(1000.0f, -1000.0f, 1000.0f)};
};

template<typename TView, typename TTransferFunc>
void renderImage(TView image, TTransferFunc tf, Camera camera, Int2 output_size, std::string output_path) {

	UnifiedImage<RGBAf, 2> render(output_size);

	forEachPosition(
		view(render),
		RayTraversal<TView, TTransferFunc>(image, tf, render.size(), camera));

	UnifiedImage<RGBA8, 2> render_out(render.size());
	copy(255.0f * constView(render), view(render_out));
	saveImage(output_path, constView(render_out));
}

TextureImage<float, 3> loadInTexture(std::string prefix, Int3 size) {
	TextureImage<float, 3> tex_image(size);

	{
		Image<uint16_t, 3> input_image(size);
		load(view(input_image), prefix);
		// forEach(view(input_image), [](auto &v){
		// 		uint16_t v2 = ((v & 0xFF) << 8) | ((v & 0xFF00) >> 8);
		// 		v = v2;
		// 	});
		// dump(view(input_image), "stagbeetle");
		// std::cout << "MIMAX " << minMax(view(input_image)) << "\n";

		Image<float, 3> fimage(input_image.size());
		copy(constView(input_image), view(fimage));
		copy(constView(fimage), view(tex_image));
	}
	return tex_image;
}

int main(int argc, char** argv) {
	try {
		std::string prefix = "../../data/intestine";
		auto size = Int3(512, 512, 548);
		auto eye_offset = Float3(-200.0f, 800.0f, -100.0f);

		// std::string prefix = "../../data/stagbeetle";
		// auto size = Int3(832, 832, 494);
		// auto eye_offset = Float3(600.0f, 800.0f, -500.0f);
		TextureImage<float, 3> tex_image = loadInTexture(prefix, size);

		auto tf_data = readTransferFunction(prefix + "_tf.txt");
		DeviceImage<RGBAf, 1> tf(tf_data.size());
		copy(constView(tf_data), view(tf));

		auto camera = Camera{
			0.5f * size,
			0.5f * size + eye_offset,
			Float3(0.0f, 0.0f, 1.0f),
			55.0f
		};
		renderImage(constView(tex_image), constView(tf), camera, Int2{1500, 1500}, "render.png");


	} catch (CudaError &e) {
		std::cout << "boost::diagnostic_information(e):\n" << boost::diagnostic_information(e) << std::endl;
		return 1;

	}
	catch (std::exception &e) {
		std::cout << "boost::diagnostic_information(e):\n" << boost::diagnostic_information(e) << std::endl;
		return 1;

	}

	return 0;

}
