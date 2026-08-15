// C++ port of the forward-propagation exercise built in ANN.ipynb.
// Builds a small feed-forward network with random weights/biases and
// propagates a random input vector through it using the sigmoid activation.

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

struct Node {
    std::vector<double> weights;
    double bias;
};

using Layer = std::vector<Node>;
using Network = std::vector<Layer>; // layers[0..num_hidden_layers-1] = hidden, layers.back() = output

static std::mt19937 rng(std::random_device{}());

double random_uniform_rounded(int decimals = 2) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double value = dist(rng);
    double factor = std::pow(10, decimals);
    return std::round(value * factor) / factor;
}

// Mirrors initialise_nwtwork(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output)
Network initialise_network(int num_inputs, int num_hidden_layers,
                            const std::vector<int>& num_nodes_hidden, int num_nodes_output) {
    Network network;
    int num_nodes_previous = num_inputs;

    for (int layer = 0; layer < num_hidden_layers + 1; ++layer) {
        int num_nodes = (layer == num_hidden_layers) ? num_nodes_output : num_nodes_hidden[layer];

        Layer current_layer;
        for (int node = 0; node < num_nodes; ++node) {
            Node n;
            n.weights.resize(num_nodes_previous);
            for (double& w : n.weights) w = random_uniform_rounded();
            n.bias = random_uniform_rounded();
            current_layer.push_back(n);
        }
        network.push_back(current_layer);
        num_nodes_previous = num_nodes;
    }
    return network;
}

double compute_weighted_sum(const std::vector<double>& inputs, const std::vector<double>& weights, double bias) {
    double sum = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) sum += inputs[i] * weights[i];
    return sum + bias;
}

double node_activation(double weighted_sum) {
    return 1.0 / (1.0 + std::exp(-weighted_sum));
}

std::vector<double> forward_propagate(const Network& network, const std::vector<double>& inputs) {
    std::vector<double> layer_inputs = inputs;

    for (size_t layer_idx = 0; layer_idx < network.size(); ++layer_idx) {
        const Layer& layer_data = network[layer_idx];
        std::vector<double> layer_outputs;

        for (const Node& node_data : layer_data) {
            double weighted_sum = compute_weighted_sum(layer_inputs, node_data.weights, node_data.bias);
            double output = node_activation(weighted_sum);
            layer_outputs.push_back(std::round(output * 10000.0) / 10000.0);
        }

        bool is_output_layer = (layer_idx == network.size() - 1);
        if (!is_output_layer) {
            std::cout << "The activations of the nodes in layer_" << (layer_idx + 1) << " are [";
            for (size_t i = 0; i < layer_outputs.size(); ++i) {
                std::cout << layer_outputs[i] << (i + 1 < layer_outputs.size() ? ", " : "");
            }
            std::cout << "]\n";
        }

        layer_inputs = layer_outputs;
    }

    return layer_inputs; // final layer_inputs holds the output layer's activations
}

int main() {
    // my_network = initialise_nwtwork(5, 3, [3, 2, 3], 1)
    Network my_network = initialise_network(5, 3, {3, 2, 3}, 1);

    // inputs = np.around(np.random.uniform(size=5), decimals=2)
    std::vector<double> inputs;
    for (int i = 0; i < 5; ++i) inputs.push_back(random_uniform_rounded());

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "The inputs to the network are [";
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::cout << inputs[i] << (i + 1 < inputs.size() ? ", " : "");
    }
    std::cout << "]\n";

    std::cout << std::setprecision(4);
    std::vector<double> predictions = forward_propagate(my_network, inputs);

    std::cout << "The predicted value by the network for the given input is " << predictions[0] << "\n";

    return 0;
}
