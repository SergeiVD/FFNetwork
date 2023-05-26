
#include <iostream>
#include <vector>
#include "include/layer.h"
// #include "include/activation_fn.h"
// #include "lib/eigen/Eigen/Dense"

// #include "include/neuralnetwork.h"

#include "include/l_techer.h"

int main() 
{
	NeuralNetwork net({3,2,2,3},{activate_fn::sigmoid, activate_fn::sigmoid, activate_fn::sigmoid});
	
	std::vector<std::vector<double>>test_in{{0.3,0.1,1.0},
		{0.5,1.0,0.2},
		{1.0,0.3,0.82},
		{0.9,1.0,3.2},
		{1.3,1.0,0.57}
	};
	std::vector<std::vector<double>>target_1{{0.0,0.0,1.0},
		                                     {0.0,1.0,0.0},
		                                     {1.0,0.0,0.0},
		                                     {0.0,1.0,0.0},
		                                     {0.0,1.0,0.0}
	};

	L_Teacher teatcher(net, test_in, target_1, 0.1, 3);
	teatcher.train();

	// std::vector<double> test_in{0.3,0.1,1.0};

	// auto out = net.feed_forward(test_in);
	// net.layer_derivative(1);
	
	// for(const auto& element : net.get_output_layers(2))
	// {
	// 	std::cout<<element<<std::endl;
	// }

	// for(const auto& element : out)
	// {
	// 	std::cout<<element<<std::endl;
	// }



		
	return 0;
}


  // Eigen::MatrixXd m(2,2);
  // m(0,0) = 1;
  // m(1,0) = 0;
  // m(0,1) = 0;
  // m(1,1) = 1;
  
  // std::cout << m << std::endl;

  // Eigen::MatrixXd m2(3,3);
  // m2.setRandom();
  // std::cout<<m2<<std::endl;
