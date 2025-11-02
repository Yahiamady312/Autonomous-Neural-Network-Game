import numpy as np
import math
import pandas as pd

df = pd.read_csv("ce889_dataCollection.csv")
#df=df.iloc[150000:,:]
#print(df.shape)
X_DIST_MAX=df.iloc[:,0].max()
X_DIST_MIN=df.iloc[:,0].min()
Y_DIST_MAX=df.iloc[:,1].max()
Y_DIST_MIN=df.iloc[:,1].min()
X_VEL_MAX= df.iloc[:,2].max()
X_VEL_MIN=df.iloc[:,2].min()
Y_VEL_MAX=df.iloc[:,3].max()
Y_VEL_MIN= df.iloc[:,3].min()
BIAS=0
LAMBDA_PARAM = 0.9


class NeuralNetworkLayer :
   def __init__(self,layer_type,weights=None,activation_function_type=None):
       if layer_type!='input' :
            # Layer specific parameters
            self.layer_type = layer_type
            self.activation_function_type=activation_function_type
            self.weights = weights
            # Neuron specific values
            self.values = np.zeros(len(weights), dtype=float)
            self.local_gradient = np.zeros(len(weights), dtype=float)
            # Delta weights for weight update in back propagation
            self.delta_weights_prev = np.zeros(weights.shape, dtype=float)
            self.delta_weights = np.zeros(weights.shape, dtype=float)
            # Error used only in output layer
            self.error = np.zeros( len(weights), dtype=float)
       else:
           self.values=np.zeros(2, dtype=float)
   @staticmethod
   def activation_function(weighted_sum,activation_function_type):
       values=np.zeros(weighted_sum.shape,dtype=float)
       if activation_function_type == 'sigmoid':
           for i in range(len(values)):
               # Sigmoid activation function for hidden layer
                   values[i]=(1 / (1 + math.exp(-weighted_sum[i] * LAMBDA_PARAM)))
       elif activation_function_type == 'linear':
           # linear activation function for hidden layer
           values=weighted_sum
       return values
   # Forward feed path function
   # Inputs for the function are : 1. input layer for function (input or hidden layer)
   # Add bias to the input layer
   # Uses Weights of input layer to find the wighted sum
   # Uses layer type to detect which activation function to use
   # Uses lambda parameter as slope modifier for the sigmoid function
   def forward(self, prev_layer):
       if BIAS!=0:
           # Add bias term to the input layer in index 0
           b_layer_values=np.insert(prev_layer.values,0,BIAS)
       else:
           b_layer_values=prev_layer.values
       # Compute the forward pass
       # Calculate wighted sum (Vk=summation(WkixXi))
       weighted_sum = np.dot(self.weights,b_layer_values)
       self.values=self.activation_function(weighted_sum,self.activation_function_type)

       return self.values

   def error_fun(self, desired):
       self.error = desired - self.values
       return self.error

hidden_layer_weights=np.array(
    [[-5.18081784, 3.91616008],
     [-8.87674682, -5.07123742],
     [-2.8161381, -7.25870982],
     [-2.83071423, -8.36043618],
     [1.59274704, -8.80746],
     [-2.91680703, -8.69756992],
     [0.15186372, -20.74601987]]
)

output_layer_weights=np.array(
    [[-0.28627138, -0.41950672, 2.19799489, 0.02259822, -0.18558746,
      -0.76644525, 0.78382613],
     [0.06738215, 20.32115031, -12.07256951, -11.12304346, 6.49383274,
      -11.57786255, 5.28965008]]
)
input_layer = NeuralNetworkLayer('input')
hidden_layer = NeuralNetworkLayer('hidden',hidden_layer_weights,'sigmoid')
output_layer = NeuralNetworkLayer('output', output_layer_weights, 'sigmoid')

class NeuralNetHolder:

    def __init__(self):
        super().__init__()

    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        # Split the input row into a list of strings and convert to floats
        try:
            input_row = list(map(float, input_row.split(',')))
            print(input_row)
            # Output: [12.5, -4.2, 7.8, 45.0, 100.0, -50.0, 10.0]
        except ValueError as e:
            print(f"Error converting input row to floats: {e}")

        norm_x_dist=(input_row[0]-X_DIST_MIN)/(X_DIST_MAX-X_DIST_MIN)
        norm_y_dist=(input_row[1]-Y_DIST_MIN)/(Y_DIST_MAX-Y_DIST_MIN)
        norm_input=np.array([norm_x_dist,norm_y_dist])
        input_layer.values=norm_input
        hidden_layer.forward(input_layer)
        output_layer.forward(hidden_layer)
        x_vel=output_layer.values[0]
        y_vel=output_layer.values[1]
        scaled_x_vel = X_VEL_MIN + (x_vel * (X_VEL_MAX - X_VEL_MIN))
        scaled_y_vel = Y_VEL_MIN + (y_vel * (Y_VEL_MAX - Y_VEL_MIN))

        return scaled_x_vel,scaled_y_vel

        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        pass
