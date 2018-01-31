#include "NN.h"

NN::NN(std::string input_file, int label_col=-1)
{
    this->input_file = input_file;
    this->data = std::vector <std::vector <double>>();
    this->predictions = std::vector <std::string>();

    //TODO Read input file and populate data
}

void NN::train_test_split()
{
}

void NN::addData(std::vector <std::vector <double>>& data)
{
    for (std::vector <double>& data_point : data)
        addDataPoint(data_point);
}

void NN::addDataPoint(std::vector <double>& data_point)
{
    this->data.push_back(data_point);
}

void NN::train(std::string file_name="", int iterations=1000)
{

}

std::vector <std::string> NN::predict(std::vector <double>& data)
{
    std::vector <std::string> predictions;
    return predictions;
}

std::string NN::predict(double data_point)
{
    std::string prediction;
    return prediction;
}

double NN::sigmoid(double n)
{
    return 1.0 / (1.0 + exp(-n));
}

double NN::sigmoid_derivative(double n)
{
    return sigmoid(n) * (1 - sigmoid(n));
}

/* Average squared error cost function */
double NN::cost(std::vector <double>& targets, std::vector <double>& predictions)
{
    double average_cost;
    for (int i = 0; i < targets.size(); i++)
        average_cost += cost(targets[i], predictions[i]); 
    average_cost /= targets.size();
    return average_cost;
}

/* Squared error cost function */
double NN::cost(double target, double prediction)
{
    return (prediction - target) * (prediction - target);
}
