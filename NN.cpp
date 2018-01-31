#include "NN.h"
#include <fstream>

NN::NN(std::string input_file)
{
    this->input_file = input_file;
    this->data = std::vector <std::vector <double>>();
    this->predictions = std::vector <std::string>();
    this->alpha = 0.01;
    

    /* Populate data, weights and bias term */
    srand(time(NULL));

    if (input_file != "") {
        populateData();
        populateWeights();

        original_weights = weights;
    }
    bias = randomDouble();
    original_bias = bias;
}

void NN::populateWeights()
{
    for (int i = 0; i < data[0].size(); i++)
        weights.push_back(randomDouble());
}

void NN::setAlpha(double alpha)
{
    this->alpha = alpha;
}

double NN::randomDouble()
{
    return (double) rand()*2 / (double) RAND_MAX - 1;
}

void NN::populateData()
{
    std::ifstream data_file;
    data_file.open(input_file);
    if (!data_file)
        perror(input_file.c_str());

    std::string line;

    while (getline(data_file, line)) {
        std::stringstream parser(line);

        double x;
        std::string y;
        std::vector <double> data_point;

        while (parser >> y) {
            if (atof(y.c_str())) {
                x = atof(y.c_str());
                data_point.push_back(x);
            } else { 
                known_labels.push_back(y);
            }
        }

        data.push_back(data_point);
    }
}

void NN::train_test_split()
{
}

double NN::getPerfectVal(Label& label)
{
    return getPerfectVal(label.label);
}

double NN::getPerfectVal(std::string& label_name)
{
    double perfect_val = -1;
    for (int i = 0; i < labels.size(); i++)
        if (labels[i].label == label_name)
            perfect_val = labels[i].perfect_value;
    return perfect_val;
}

void NN::createLabel(double perfect_value, double lower_bound, double upper_bound, std::string& label_name) 
{
    Label label;
    label.lower_bound = lower_bound;
    label.upper_bound = upper_bound;
    label.perfect_value = perfect_value;
    label.label = label_name;
    labels.push_back(label);
}

std::string NN::getLabel(double prediction)
{
    std::string pred_label = "";
    for (Label l : labels)
        if (prediction >= l.lower_bound && prediction <= l.upper_bound)
            pred_label = l.label;
    return pred_label;
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

/* Train neural network on data by using backpropagation with variable
 * iterations */
void NN::train(int iterations)
{
    std::vector <double> costs;
    std::cout << "Before training, b = " << bias << " w = {";
    for (int i = 0; i < weights.size(); i++)
        std::cout << weights[i] << ", ";
    std::cout << "}" << std::endl;

    for (int i = 0; i < iterations; i++) {
        int random_index = rand() % data.size();
        std::vector <double> data_point = data[random_index];

		double z = 0;
        for (int j = 0; j < data_point.size(); j++)
            z += data_point[j] * weights[j];
        z += bias;

        double prediction = sigmoid(z);
        double target = getPerfectVal(known_labels[random_index]);

        double pred_cost = cost(prediction, target);
        costs.push_back(pred_cost);

        /* Each variable of the name "da_db" is the derivative of a
         * with respect to b. This is to perform gradient descent in
         * order to adjust our weights and bias terms. This is the
         * part where we are actually training our network. */
        double dcost_dpred, dpred_dz, dz_db, dcost_dz,
               dcost_dw1, dcost_dw2, dcost_db;
        std::vector <double> dz_dweight, dcost_dweight;

        dcost_dpred = 2 * (prediction - target);
        dpred_dz = sigmoid_derivative(z);

        for (int j = 0; j < weights.size(); j++)
            dz_dweight.push_back(data_point[j]);
        dz_db = 1;

        dcost_dz = dcost_dpred * dpred_dz;
        for (int j = 0; j < weights.size(); j++)
            dcost_dweight.push_back(dcost_dz * dz_dweight[j]);

        dcost_db = dcost_dz * dz_db;
    
        /* Adjust weights and bias */
        for (int j = 0; j < weights.size(); j++)
            weights[j] -= alpha * dcost_dweight[j];

        bias -= alpha * dcost_db;
    }
    std::cout << "Costs " << costs[0] << " --> " << costs[costs.size() - 1] << std::endl;
}

std::vector <double> NN::predict(std::vector <std::vector <double>>& data)
{
    std::vector <double> predictions;

    for (std::vector <double>& data_point : data)
        predictions.push_back(predict(data_point));

    return predictions;
}

double NN::predict(std::vector <double>& data_point)
{
    double prediction = 0.0;

    for (int i = 0; i < data_point.size(); i++)
        prediction += data_point[i] * weights[i];

    prediction += bias;
    prediction = sigmoid(prediction);
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

double NN::test(int num) 
{
    if (num > data.size()) num = data.size();
    double accuracy = 0;
    for (int i = 0; i < num; i++) {
        if (getLabel(predict(data[i])) == known_labels[i])
            accuracy++;
    }
    accuracy /= num;
    return accuracy;
}

void NN::reset(bool keep_weights)
{
    if (keep_weights) {
        weights = original_weights;
        bias = original_bias;
    } else {
        populateWeights();
        bias = randomDouble();
    }
}
