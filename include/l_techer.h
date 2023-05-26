#ifndef L_TECHER_H
#define L_TECHER_H

#include "layer.h"
#include "neuralnetwork.h"
#include <vector>

#include <iostream>

class L_Teacher
{
public:

	
	L_Teacher(NeuralNetwork& network,
			  std::vector<std::vector<double>>& training_data,
			  std::vector<std::vector<double>>& target_data,
			  double learning_rate = 0.1,
			  int num_epochs = 1000) :
		network_{network},
		training_data_{training_data},
		target_data_{target_data},
		learning_rate_{learning_rate},
		num_epochs_{num_epochs}
		{}


    virtual ~L_Teacher()=default;

	void train()
		{
			for (int epoch = 0; epoch < num_epochs_; ++epoch)
			{
				double epoch_error = 0.0;
				for (int i = 0; i < training_data_.size(); ++i)
				{
					// Получение выхода сети
					auto output = network_.feed_forward(training_data_[i]);

					// Вычитание ответа сети из правильного ответа
					auto error =  [output,this,i]
						{
							std::vector<double> err(output.size());
							for (size_t j = 0; j < target_data_[i].size(); ++j)
							{
								err[j] = target_data_[i].at(j) - output[j];
							}
							return err;
						}();

					// Расчет функции ошибки (как вариант среднеквадратичная ошибка)
					epoch_error += [error]()
						{
							double sum_sq{0.0};
							for(int i = 0; i < error.size(); ++i)
							{
								sum_sq += error[i] * error[i];
							}
							return sum_sq/error.size();
						}();

					// auto gradients = backpropagate(error);
					auto gradients = backpropagate(error);
					update_weights(gradients);
				}
				epoch_error /= training_data_.size();
				std::cout << "Epoch " << epoch+1 << ", Error = " << epoch_error << std::endl;
			}


		}

private:
	NeuralNetwork& network_;
	std::vector<std::vector<double>>& training_data_;
	std::vector<std::vector<double>>& target_data_;
	double learning_rate_;
	int num_epochs_;

	// Вычисление градиента для весов
	std::vector<std::vector<double>> calc_grad_weights(const std::vector<double>& delta, const std::vector<double>& input)
		{
			std::vector<std::vector<double>> grad_weights;
			for (int i = 0; i < delta.size(); ++i)
			{
				std::vector<double> grad_ij(input.size());
				for (int j = 0; j < input.size(); ++j)
				{
					grad_ij[j] = delta[i] * input[j];
				}
				grad_weights.push_back(grad_ij);
			}
			return grad_weights;
		}

	// Вычисление градиента для смещений
	std::vector<std::vector<double>> calculate_grad_bias(const std::vector<double>& delta)
		{
			std::vector<std::vector<double>> grad_bias;
			for (int i = 0; i < delta.size(); ++i)
			{
				std::vector<std::vector<double>> grad_bias(delta.size());
				for (int i = 0; i < delta.size(); ++i)
				{
					std::vector<double> grad_i(delta.size());
					grad_i[i] = delta[i];
					grad_bias[i]= grad_i;
				}
				return grad_bias;
			}
			return grad_bias;
		}

	  // Обратное распространение ошибки
	std::vector<std::vector<std::vector<double>>> backpropagate(const std::vector<double>& error)
		{
			std::vector<std::vector<std::vector<double>>> gradients(network_.get_layers().size());
			int num_last_layer = network_.get_layers().size()-1;

			// Вычисление градиента для последнего слоя
			// auto delta = last_layer.derive_activation(last_layer.get_output()) * error;

			std::vector<double> delta = [error,this,num_last_layer]()  // Ошибка выхоного слоя ()
					{
						std::vector<double> last_layer_derivative{network_.layer_derivative(num_last_layer)};
						std::vector<double> delta(last_layer_derivative.size());						
						if (error.size() != last_layer_derivative.size())
						{
							throw std::runtime_error("multiplicate fun no correct size vector last_layer_derivative: " +
													 std::to_string(last_layer_derivative.size()) +
													 " size vector error: " +
													 std::to_string(error.size()));							
						}

						for (size_t i = 0; i < last_layer_derivative.size(); ++i)
						{
							delta[i] = last_layer_derivative[i] * error[i];
						}
						
						return delta;
					}();

			// Получаем входы в последний слой
			std::vector<double> input_last_layer{network_.get_output_layers(num_last_layer-1)}; // Выход предполседного слоя является входом последнего!

			// Градиент по весам связей
			std::vector<std::vector<double>> grad_weights = [delta, input_last_layer]()
				{
					std::vector<std::vector<double>> grad_weights;
					for (int i = 0; i < delta.size(); ++i)
					{
						std::vector<double> grad_ij(input_last_layer.size());
						for (int j = 0; j < input_last_layer.size(); ++j)
						{
							grad_ij[j] = delta[i] * input_last_layer[j];
						}
						grad_weights.push_back(grad_ij);
					}
					return grad_weights;
				}();

			// Градиент по смещениям
			std::vector<std::vector<double>> grad_bias = [delta]()
				{
					std::vector<std::vector<double>> grad_bias(delta.size());
					for (int i = 0; i < delta.size(); ++i)
					{
						std::vector<double> grad_i(delta.size());
						grad_i[i] = delta[i];
						grad_bias[i]= grad_i;
					}
					return grad_bias;
				}();

			gradients.push_back(grad_weights);
			gradients.push_back(grad_bias);


			// Вычисление градиентов для остальных слоев
			for (int i = num_last_layer-1; i >= 0; --i)
			{
				auto num_neuron = network_.get_weights().at(i+1).size();
				auto derive_action = network_.layer_derivative(i);

				for (size_t j = 0; j < num_neuron; ++j)
				{
					std::vector<double> rez;
					int weigth = network_.get_weights().at(i+1).at(j).size();
					double sum{0};
					for (size_t k = 0; k < weigth; ++k)
					{
						rez.push_back(derive_action[j] * network_.get_weights().at(i+1).at(j).at(k) * delta[j]);
					}
					delta = rez;
				}


				


				grad_weights = calc_grad_weights(delta, network_.get_layers().at(i).get_input());
				grad_bias = calculate_grad_bias(delta);

				gradients.insert(gradients.begin(), grad_bias);
				gradients.insert(gradients.begin(), grad_weights);

			


			}
			
			print(gradients);

			// // Вычисление градиентов для остальных слоев
			// for (int i = network_.get_layers().size()-2; i >= 0; --i) {
			// 	auto layer = network_.get_layers()[i];
			// 	auto next_layer = network_.get_layers()[i+1];

			// 	delta = layer.derive_activation(layer.get_output()) * (next_layer.get_weights().transpose() * delta);
			// 	grad_weights = calculate_grad_weights(delta, layer.get_input());
			// 	grad_bias = calculate_grad_bias(delta);
			// 	gradients.insert(gradients.begin(), grad_bias);
			// 	gradients.insert(gradients.begin(), grad_weights);
			// }

			return gradients;
		}

	// Обновление весов и смещений
	void update_weights(const std::vector<std::vector<std::vector<double>>>& gradients)
		{
			auto layers = network_.get_layers();
			for (int i = 0; i < layers.size(); ++i)
			{
				auto layer = layers[i];
				auto grad_weights = gradients[i*2];
				auto grad_bias = gradients[i*2 + 1];

				// Обновление весов
				for (int j = 0; j < layer.get_num_outputs(); ++j)
				{
					for (int k = 0; k < layer.get_num_inputs(); ++k)
					{
						layer.set_weights(j, k, layer.get_weights().at(j).at(k) + learning_rate_ * grad_weights[j][k]);
					}
				}

				// Обновление смещений
				for (int j = 0; j < layer.get_num_outputs(); ++j)
				{
					layer.set_bias(j, layer.get_biases().at(j) + learning_rate_ * grad_bias[j][0]);
				}
			}

		}



	void print(std::vector<std::vector<std::vector<double>>> vec)
		{
			for (const auto& element : vec)
			{
				for(const auto& elem : element)
				{
					for(const auto& ele : elem)
					{
						std::cout<<ele<<std::endl;
					}
					std::cout<<"-----------"<<std::endl;
				}
			}

		}
	
};




#endif /* L_TECHER_H */
