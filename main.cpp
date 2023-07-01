#include <iostream>

#include "include/neuralnetwork.h"
#include "include/l_teacher.h"
#include "mnist/include/mnist_reader.h"

#include <vector>

#include <stdio.h>
#include <unistd.h>
#include <termios.h>

int getch()
{
	struct termios oldattr, newattr;
	int ch;
	tcgetattr(STDIN_FILENO, &oldattr);
	newattr = oldattr;
	newattr.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &newattr);
	ch = getchar();
	tcsetattr(STDIN_FILENO, TCSANOW, &oldattr);
	return ch;
}

std::vector<std::vector<double>> target_net_label(std::vector<int> datase_lables)
{
	std::vector<std::vector<double>> target_for_network(datase_lables.size(), std::vector<double>(10));
	for (size_t i = 0; i < datase_lables.size(); ++i)
	{
		switch (datase_lables[i])
		{
		case 0: target_for_network[i] = {1,0,0,0,0,0,0,0,0,0}; break;
		case 1: target_for_network[i] = {0,1,0,0,0,0,0,0,0,0}; break;
		case 2: target_for_network[i] = {0,0,1,0,0,0,0,0,0,0}; break;
		case 3: target_for_network[i] = {0,0,0,1,0,0,0,0,0,0}; break;
		case 4: target_for_network[i] = {0,0,0,0,1,0,0,0,0,0}; break;
		case 5: target_for_network[i] = {0,0,0,0,0,1,0,0,0,0}; break;
		case 6: target_for_network[i] = {0,0,0,0,0,0,1,0,0,0}; break;
		case 7: target_for_network[i] = {0,0,0,0,0,0,0,1,0,0}; break;
		case 8: target_for_network[i] = {0,0,0,0,0,0,0,0,1,0}; break;
		case 9: target_for_network[i] = {0,0,0,0,0,0,0,0,0,1}; break;
		}
	}
	return target_for_network;
}

int answer_net(const mlm::vec<double>& answer_net)
{
	double max_val{0.0};
	std::size_t index{0};
	for (size_t i = 0; i < answer_net.size(); ++i)
	{
		if(answer_net[i] > max_val)
		{
			max_val = answer_net[i];
			index = i;
		}
	}

	return index;
}

template<typename Pixel = uint8_t, typename Label = uint8_t, typename Net>
void test_net_image(const mnist::MNIST_dataset<Pixel,Label>& dataset, const mnist::type& type, Net& net)
{
	uint32_t rows{28};
	uint32_t cols{28};

	auto data_image = std::vector<std::vector<Pixel>>{};
	auto data_label = std::vector<Label>{};
		
	if(type == mnist::type::training)
	{
		data_image = dataset.training_images;
		data_label = dataset.training_labels;
	}			

	if(type == mnist::type::test)
	{
		data_image = dataset.test_images;
		data_label = dataset.test_labels;
	}			
	std::string label;
	for (size_t item_id = 0; item_id < dataset.test_images.size(); ++item_id)
	{

		label = "label: " + std::to_string(static_cast<int>(data_label[item_id])) + " net answer: " + std::to_string(answer_net(net.feedForward(data_image[item_id])));
		auto* pixels = data_image[item_id].data();

		cv::Mat image;
		switch (sizeof(Pixel))
		{
		case 8: {image =  cv::Mat(rows, cols, CV_64FC1, pixels); break;}
		case 4: {image =  cv::Mat(rows, cols, CV_32FC1, pixels); break;}
		case 1: {image =  cv::Mat(rows, cols, CV_8UC1, pixels); break;}
		}

		// resize bigger for showing
		cv::resize(image, image, cv::Size(300,300));
			
		cv::setWindowTitle("MNIST_VIEWER", label);
		cv::imshow("MNIST_VIEWER", image);

		int key = cv::waitKey(0);
		if(key == 27 || key == 'q')break;

	}
	cv::destroyWindow("MNIST_VIEWER");
}


int main() 
{
	mnist::MNIST_dataset<double,int> dataset = mnist::read_dataset<double,int>();
	// mnist::show_dataset(dataset, mnist::type::training);

	std::vector<std::vector<double>> target = target_net_label(dataset.training_labels);

	NeuralNetwork net({784,10,10,10},{activate_fn::sigmoid, activate_fn::sigmoid, activate_fn::sigmoid});
	L_Teacher teatcher(net, dataset.training_images, target, 0.01, 100);
		
	teatcher.train();

	std::cout<<"Test network"<<std::endl;

	test_net_image(dataset, mnist::type::test, net);

	
	// for (size_t i = 0; i < dataset.test_labels.size(); ++i)
	// {		
	// 	std::cout<<answer_net(net.feedForward(dataset.test_images[i]))<<" : "<<dataset.test_labels[i]<<std::endl;
	// 	getch();
	// }

	std::cout<<"hellot"<<std::endl;
	
	return 0;
}



