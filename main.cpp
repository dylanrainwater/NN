#include <iostream>
#include "NN.h"
using namespace std;

/*
 * This is the driver file that shows possible usage of the
 * neural network library.
 */
int main(int argc, char **argv) {
    if (argc <= 3) {
        cerr << "Usage: NN filename label1 label2 [label3] ..." << endl;
        return -1;
    }

    string file_name(argv[1]);

    NN classifier(file_name);

    int num_labels = argc - 2;
    vector <string> labels;
    for (int i = 2; i < argc; i++)
        labels.push_back(string(argv[i]));

    int j = 0;
    for (double i = 0; i < 1; i += (1.0 / num_labels), j++) {
        double perfect_value, lower_bound, upper_bound;
        lower_bound = i;
        upper_bound = i + (1.0 / num_labels);
        perfect_value = (lower_bound + upper_bound) / 2.0; 
        if (lower_bound == 0)
            perfect_value = 0;
        if (upper_bound == 1)
            perfect_value = 1;
        classifier.createLabel(perfect_value, lower_bound, upper_bound, labels[j]);
    }

    classifier.train(100000);

    int num = 100;
    double accuracy = classifier.test(num) * 100;
    cout << "On " << num << " random trials, I am " << accuracy << "% accurate." << endl;
    return 0;
}
