import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)  # You can choose any integer as the seed

# ---------------------------
# Define Constants and Hyper parameters
# ---------------------------
# Hyper Parameters 
LAMBDA_PARAM = 0.9
LEARNING_RATE_PARAM = 0.7
MOMENTUM_PARAM = 0.2
# Number of Neurons
NUM_HIDDEN_NEURONS = 7
NUM_OUTPUT_NEURONS = 2
NUM_INPUT_NEURONS = 2
# BIAS=0 with no bias BIAS=1 with bias
BIAS = 0
EPOCHS = 100

data_frame = pd.read_csv('ce889_dataCollection.csv')
#data_frame=data_frame.iloc[150000:,:]
print(data_frame.shape)


# ---------------------------
# Normalization using min-max
# ---------------------------
def normalize_min_max(df):
    df_min_max_scaled = (df - df.min()) / (df.max() - df.min())
    return df_min_max_scaled


# ---------------------------
# Spilt Data into 70% Training and 80% Testing
# ---------------------------
def data_split(df_scaled):
    #spliting into training and testing
    train, val = train_test_split(df_scaled, test_size=0.3, random_state=42, shuffle=True)
    input_dist_train = train.iloc[:, 0:2]
    input_dist_val = val.iloc[:, 0:2]
    output_vel_train = train.iloc[:, 2:4]
    output_vel_val = val.iloc[:, 2:4]
    return input_dist_train, input_dist_val, output_vel_train, output_vel_val


# ---------------------------
# Neural Network class
# ---------------------------
class NeuralNetworkLayer:
    def __init__(self, layer_type, weights=None, activation_function_type=None):
        if layer_type != 'input':
            # Layer specific parameters
            self.layer_type = layer_type
            self.activation_function_type = activation_function_type
            self.weights = weights
            # Neuron specific values
            self.values = np.zeros(weights.shape[0], dtype=float)
            self.local_gradient = np.zeros(weights.shape[0], dtype=float)
            # Delta weights for weight update in back propagation
            self.delta_weights_prev = np.zeros(weights.shape, dtype=float)
            self.delta_weights = np.zeros(weights.shape, dtype=float)
            # Error used only in output layer
            self.error = np.zeros(weights.shape[0], dtype=float)
        # if input layer save only values of neurons (input values )
        else:
            self.values = np.zeros(2, dtype=float)

    # ---------------------------
    # Activation Function used in Feed Forward
    # ---------------------------
    def activation_function(self, weighted_sum):
        
        # Variable to save neuron values
        values = np.zeros(weighted_sum.shape, dtype=float)
        
        # Check if sigmoid or linear
        if self.activation_function_type == 'sigmoid':
            for i in range(len(values)):
                # Sigmoid activation function for hidden layer
                values[i] = (1 / (1 + math.exp(-weighted_sum[i] * LAMBDA_PARAM)))
        else:
            # linear activation function for output layer
            values = weighted_sum
        return values

    # ---------------------------
    # Forward feed path function
    # ---------------------------
    # Inputs for the function are :  input layer for function (input or hidden layer)
    def forward(self, prev_layer):
        # Add BIAS if BIAS=1
        if BIAS != 0:
            # Add bias term to the input layer in index 0
            p_layer_values = np.insert(prev_layer.values, 0, BIAS)
        else:
            p_layer_values = prev_layer.values
        # Compute the forward pass
        # Calculate wighted sum (Vk=summation(WkixXi))
        weighted_sum = np.dot(self.weights, p_layer_values)
        self.values = self.activation_function(weighted_sum)

        return self.values

    # ---------------------------
    # Error Function
    # ---------------------------
    def error_fun(self, desired):
        self.error = desired - self.values
        return self.error

    # ---------------------------
    # Weight Update Function 
    # ---------------------------
    def weight_update(self, prev_layer_values):
        self.delta_weights = LEARNING_RATE_PARAM * np.outer(self.local_gradient,
                                                            prev_layer_values) + MOMENTUM_PARAM * self.delta_weights_prev
        # Save the new delta weights as previous ones
        self.delta_weights_prev = self.delta_weights
        # Apply the weight updates to the weights matrix
        self.weights += self.delta_weights

    # ---------------------------
    # Local Gradient Function
    # ---------------------------
    def get_local_gradient(self, next_layer):
        # Compute local gradients for the output layer
        if self.layer_type == 'output':
            # If the activation function is 'linear', the local gradient is simply the error (delta)
            # because differentiation  of the activation linear function is zero
            if self.activation_function_type == 'linear':
                self.local_gradient = self.error
            elif self.activation_function_type == 'sigmoid':
                # For sigmoid activation functions, calculate the gradient using the derivative of the activation function
                self.local_gradient = LAMBDA_PARAM * self.values * (1 - self.values) * self.error
        # Local Gradient for Hidden Layer 
        else:
            weighted_gradients = np.dot(next_layer.local_gradient,
                                        (next_layer.weights[:, BIAS:] - next_layer.delta_weights[:, BIAS:]))
            self.local_gradient = LAMBDA_PARAM * self.values * (1 - self.values) * weighted_gradients

    # ---------------------------
    # Back Propagation Function
    # ---------------------------
    def backward(self, prev_layer, next_layer=None):
        if BIAS != 0:
            p_layer_values = np.insert(prev_layer.values, 0, BIAS)
        else:
            p_layer_values = prev_layer.values
        self.get_local_gradient(next_layer)
        self.weight_update(p_layer_values)

# ---------------------------
# Neural Network Model 
# ---------------------------
class NeuralNetworkModel:
    def __init__(self, model_input, model_hidden, model_output):
        self.input_layer = model_input
        self.hidden_layer = model_hidden
        self.output_layer = model_output

    # ---------------------------
    # Train Function
    # ---------------------------
    def train(self, input_df_training, target_df_training):
        # Initialize avg_rmse_training to store RMSE values for each epoch
        # Reset errors at the start of each epoch
        total_error_x_training = 0
        total_error_y_training = 0
        num_iterations_training = len(input_df_training)
        for i in range(num_iterations_training):
            # Set input values for the input layer
            self.input_layer.values = input_df_training[i]
            # Forward pass
            self.hidden_layer.forward(self.input_layer)
            self.output_layer.forward(self.hidden_layer)
            # Compute the error
            error_training = self.output_layer.error_fun(target_df_training[i])
            # Accumulate the errors for each output
            total_error_x_training += error_training[0] ** 2
            total_error_y_training += error_training[1] ** 2
            # Backward pass (gradient calculation)
            self.output_layer.backward(self.hidden_layer)
            self.hidden_layer.backward(self.input_layer, self.output_layer)
        # Calculate RMSE for both outputs
        rmse_x_training = math.sqrt(
            total_error_x_training / num_iterations_training)  # Use the actual number of samples
        rmse_y_training = math.sqrt(
            total_error_y_training / num_iterations_training)  # Use the actual number of samples
        # Calculate the average RMSE for the epoch
        avg_rmse_training = (rmse_x_training + rmse_y_training) / 2
        return avg_rmse_training

    # ---------------------------
    # Validation Function
    # ---------------------------
    def validate(self, input_df_validation, target_df_validation):
        # Initialize avg_rmse_validation to store RMSE values for each epoch
        total_error_x_validation = 0
        total_error_y_validation = 0
        num_iterations_validation = len(input_df_validation)
        for i in range(num_iterations_validation):
            # Set input values for the input layer
            self.input_layer.values = input_df_validation[i]
            # Forward pass
            self.hidden_layer.forward(self.input_layer)
            self.output_layer.forward(self.hidden_layer)
            # Compute the error
            error_validation = self.output_layer.error_fun(target_df_validation[i])
            # Accumulate the errors for each output
            total_error_x_validation += (error_validation[0] ** 2)
            total_error_y_validation += (error_validation[1] ** 2)

        # Calculate RMSE for both outputs
        rmse_x_validation = math.sqrt(
            total_error_x_validation / num_iterations_validation)  # Use the actual number of samples
        rmse_y_validation = math.sqrt(
            total_error_y_validation / num_iterations_validation)  # Use the actual number of samples
        # Calculate the average RMSE for the epoch
        avg_rmse_validation = (rmse_x_validation + rmse_y_validation) / 2
        return avg_rmse_validation

    # ---------------------------
    # Fitting  Function
    # ---------------------------
    def train_validate(self, input_df_training, target_df_training, input_df_validation, target_df_validation,num_epochs):
        rmse_training = np.zeros(num_epochs)
        rmse_validation = np.zeros(num_epochs)
        error_counter = 0
        for epoch in range(num_epochs):
            rmse_training[epoch] = self.train(input_df_training, target_df_training)
            rmse_validation[epoch] = self.validate(input_df_validation, target_df_validation)

            # Stopping Condition
            if epoch > 1 and ((rmse_validation[epoch - 1] - rmse_validation[epoch]) < 0.00001):
                error_counter += 1
                if error_counter > 5:
                    break
            else:
                error_counter = 0
            print("Number of epoch : ", epoch)
        print("RMSE training : \n", rmse_training[:epoch])
        print("RMSE validation : \n", rmse_validation[:epoch])

        # Plot Training and Validation RMSE
        plt.figure(figsize=(10, 6))
        plt.plot(rmse_training[:epoch], label="Training RMSE")
        plt.plot(rmse_validation[:epoch], label="Validation RMSE")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.title(f"Training and Validation RMSE Over Epochs (Hidden Neurons: {NUM_HIDDEN_NEURONS})")
        plt.legend()
        plt.grid()
        plt.show()


# Reading the CSV file
scaled_data = normalize_min_max(data_frame)
training_input, test_input, training_output, test_output = data_split(scaled_data)

# Weights matrices dim = number of neurons in layer(k) x number of neuron in previous layer(i)
hidden_layer_weights = np.random.rand(NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS + BIAS)
output_layer_weights = np.random.rand(NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS + BIAS)  #this 1 refers to the bias

input_layer = NeuralNetworkLayer('input')
hidden_layer = NeuralNetworkLayer('hidden', hidden_layer_weights, 'sigmoid')
output_layer = NeuralNetworkLayer('output', output_layer_weights, 'sigmoid')

model = NeuralNetworkModel(input_layer, hidden_layer, output_layer)
model.train_validate(training_input.to_numpy(), training_output.to_numpy(), test_input.to_numpy(),
                     test_output.to_numpy(), EPOCHS)

print("output layer weights : \n", np.array2string(output_layer.weights, separator=', '))
print("hidden layer weights : \n", np.array2string(hidden_layer.weights, separator=', '))
