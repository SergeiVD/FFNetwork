#ifndef LAYER_H
#define LAYER_H

#include "activation_fn.h"
#include <bits/c++config.h>
#include <cstddef>
#include <vector>
#include <string>

#include "../mlm/matrix.h"
#include "../mlm/vec.h"

enum class activate_fn
{
	sigmoid,
	relu,
	tanh
}; 

class Layer
{
public:
	Layer() = default;
    Layer(std::size_t inputSize, std::size_t outputSize, activate_fn activationFunction);
    Layer(const Layer& layer);
    Layer(Layer&& layer) noexcept;
    Layer& operator=(const Layer& layer);
    Layer& operator=(Layer&& layer) noexcept;
	~Layer();

	const mlm::vecd& activate(const mlm::vecd& input);

	void update_weights(const int& neuronNum, const int& weightNum, const double& weight_gradient);
    void update_bias(const int& neuronNum, const double& bias_gradient);
	
	mlm::vecd deriveActivation(const mlm::vecd& val);
	const double derive_activation(const std::size_t& num_output)const;
	const mlm::vecd derive_activation()const;

	const mlm::vecd& get_input()const {return input_;}
	const mlm::vecd& get_output()const {return output_;}
	const mlm::matrixd& get_weights()const { return weights_; }
	const mlm::vecd& get_biases()const { return biases_; }

	const std::size_t size()const {return num_out_;}

    void save_weights_to_file(const std::string& filename) const;
    void load_weights_from_file(const std::string& filename);

private:
	std::size_t num_input_{0};
	std::size_t num_out_{0};

	mlm::vecd input_;
	mlm::vecd output_;

    mlm::matrixd weights_;
	mlm::vecd biases_;

	activate_fn activationFunction_{activate_fn::sigmoid};
};

#endif /* LAYER_H */
