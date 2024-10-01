import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

class NeuralNetwork(BaseEstimator, ClassifierMixin):
    """
    ### Description 
    Implementation of an Artificial Neural Network designed for binary classification using numpy, and it implements 
    common techniques and algorithms to improve the performance and convergence.

    ### Features
    - Different activation functions
    - Weight initialization techniques
    - Implementation of common optimization algorithms
    - Learing rate decay strategies
    - Regularization to prevent overfitting
    - Plots of loss and accuracy curves
    - Outputs of nn_predict and nn_predict_proba are compatible with sklearn.metrics functions

    ### Usage
    1. Initialize NeuralNetwork Object with desired parameters, if none are specified, default parameters will be used.
    2. Train the model with nn_train(). Specify verbose=False as an argument in nn_train to hide loss and accuracy for each epoch
    3. Generate predictions with nn_predict(), and obtain prediction probabilities with nn_predict_proba()
    4. Visualize loss and accuracuy plots()

    ### Dependencies
    - numpy
        - np.seterr is used to prevent warning from showing during neural network training. All errors are appropriately handled.
    - matplotlib
    - sklearn

    ### Arguments
    - activation_func (str): Activation function for hidden layers and output (tanh, sigmoid, relu, linear)
        - Default: 'relu'
    - hidden_layer_sizes (List[int]): List of number of neurons in each hidden layer
        - Default: [100, 100, 100]
    - initialization (str): Method used to initalize the weights (he, xavier, random)
        - Initialization formula determines how weights are initialized
        - Default: 'he'
    - epochs (int): Number of training iterations
        - Default: 50
    - optimizer (str): Optimization algorithm (adam, sgd)
        - Optimizer used to update weights and biases during training
        - Default: 'adam'
    - learning_rate (float): Specifies the learning rate for gradient descent
        - Default: 0.01
    - lr_decay_type (str): Learning rate decay strategy (step, exponential, time)
        - Formula to dynamically adjust the learning rate during training
        - If lr_decrease or lr_epoch_drop is 0, then constant learning rate is used.
        - Default: 'constant'
    - lr_decrease (float): Value to decrease learning rate by in specified decay type
        - Use when decay_type is not constant
        - Default: 0.5
    - lr_epoch_drop (int): Epoch at which to drop the learning rate
        - Use when decay_type is not constant
        - Default: 5
    - alpha (float): Regularization term to prevent overfitting
        - Default: 0.01
    - regularization (str): Type of regularization (l1, l2, none)
        - Regularization method applied to the network to prevent overfitting
        - Default: 'l2'
    - tolerance (float): Tolerance used to determine early convergence
        - Default: 0.0001
    - tolerance_counter (int): Tolerance count to end Neural Network training
        - Default: 10
    - beta_1 (float): Control first moment value (mean) for adam optimizer
        - Use only when optimizer='adam'
        - Default: 0.9
    - beta_2 (float): Control second moment value (variance) for adam optimizer
        - Use only when optimizer='adam'
        - Default: 0.999
    - epsilon (float): To prevent division by 0 when using adam optimizer
        - Use only when optimizer='adam'
        - Default: 1e-8
    - random_state (int): Sets numpy seed to allow for reproducable results
        - Default: None

    ### Example:
    ```
    # Required imports
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score

    # Optional imports
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, confusion_matrix, classification_report, precision_recall_curve
    import scikitplot as skplot

    # Using X as dataframe with features and y as the target feature with binary class
    neural_net = NeuralNetwork(activation_func='sigmoid', hidden_layer_sizes=[20, 20, 20],
                               initialization='random', epochs=50, optimizer='adam', 
                               learning_rate=0.01, lr_decay_type='constant', random_state=18)
    
    X = df_features
    y = df_target.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

    fit = neural_net.nn_train(X_train, y_train)

    # get training predictions
    y_pred_train = neural_net.nn_train(X_train)

    # get testing predictions
    y_pred_test = neural_net.nn_predict(X_test)

    # get class probability predictions
    pred_proba = neural_net.nn_predict_proba(X_test)

    # plotting the loss and accuracy curves for training
    neural_net.plots()

    # confusion matrix, classification report, auRoc curves
    print(classification_report(y_test, y_pred_test))
    skplt.metrics.plot_confusion_matrix(y_test, y_pred_test)
    skplt.metrics.plot_roc(y_test, pred_proba)
    plot_precision_recall(y_test, pred_proba)
    plt.show()
    ```
    """

    def __init__(self, activation_func='relu', hidden_layer_sizes=[100, 100, 100],
                 initialization='he', epochs=100, optimizer='adam',
                 learning_rate=0.01, lr_decay_type='constant', lr_decrease=0.5,
                 lr_epoch_drop=5, alpha=0.01, regularization='l2',
                 tolerance = 0.0001, tolerance_counter = 10, beta_1 = 0.9,
                 beta_2 = 0.999, epsilon=1e-8, random_state=None):
        self.activation_func = activation_func
        self.hidden_layer_sizes = hidden_layer_sizes
        self.initialization = initialization
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_decay_type = lr_decay_type
        self.lr_decrease = lr_decrease
        self.lr_epoch_drop = lr_epoch_drop
        self.alpha = alpha
        self.regularization = regularization
        self.tolerance = tolerance
        self.tolerance_counter = tolerance_counter
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.random_state = random_state

        np.seterr(all='ignore')
        np.random.seed(self.random_state)
    
    # sets the parameters for the NeuralNetwork object. Argument passed is a dictionary
    def set_params(self, params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise KeyError(f"Invalid parameter: {k}")
        
        np.random.seed(self.random_state)

    # Returns the parameters of the object
    def get_params(self):
        params = {
            'activation_func': self.activation_func,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'initialization': self.initialization,
            'epochs': self.epochs,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'lr_decay_type': self.lr_decay_type,
            'lr_decrease': self.lr_decrease,
            'lr_epoch_drop': self.lr_epoch_drop,
            'alpha': self.alpha,
            'regularization': self.regularization,
            'tolerance': self.tolerance,
            'tolerance_counter': self.tolerance_counter,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'random_state': self.random_state
        }

        return params
    
    # Defining all the common activation functions and their derivatives
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_ddx(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_ddx(self, x):
        return (1 - (self.tanh(x))**2)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_ddx(self, x):
        return np.where(x < 0, 0, 1)

    def linear(self, x):
        return x

    def linear_ddx(self, x):
        return x**0
    
    # Allows the learning rate to vary with the number of epochs to allow for better convergence
    def learning_rate_scheduler(self, learning_rate, decay_type, decrease_value, epoch_drop):
        final_lr = 0

        if decrease_value == 0 or epoch_drop == 0:
            final_lr = learning_rate
        else:
            if decay_type == 'exponential':
                drop_factor = np.power(decrease_value, np.floor(self.epoch / epoch_drop))
                final_lr = learning_rate * drop_factor
            elif decay_type == 'time':
                drop_factor = (1 / (1 + decrease_value * np.floor(self.epoch / epoch_drop)))
                final_lr = learning_rate * drop_factor
            elif decay_type == 'step':
                drop_factor = np.power(decrease_value, np.floor((1 + self.epoch) / epoch_drop))
                final_lr = learning_rate * drop_factor
            else:
                final_lr = learning_rate

        return final_lr

    # Defining loss function for binary prediction
    def log_loss(self, y_actual, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred))
        
        return loss
    
    # Defining different initialization functions to help with convergence
    def xavier_initialization(self, input_size, output_size):
        bound = np.sqrt(6.0 / (input_size + output_size))
        weights = np.random.normal(-bound, bound, size=(input_size, output_size))

        return weights
    
    def he_initialization(self, input_size, output_size):
        std_dev = np.sqrt(2.0 / input_size)
        weights = np.random.normal(0, std_dev, size=(input_size, output_size))

        return weights
    
    def random_initialization(self, input_size, output_size):
        weights = np.random.randn(input_size, output_size)

        return weights

    # Initializing the weights and biases based on the number of sizes in input, hidden and output layers and
    # the initialization method
    def weights_biases(self, input_size, hidden_layer_sizes, output_size, initialization):
        size = [input_size] + hidden_layer_sizes + [output_size]
        
        initialization_formulas = {
            'xavier': self.xavier_initialization,
            'he': self.he_initialization,
            'random': self.random_initialization
        }
        weights = [initialization_formulas[initialization](size[i], size[i + 1]) for i in range(len(size) - 1)]
        biases = [np.zeros((1, size[i + 1])) for i in range(len(size) - 1)]

        return weights, biases

    # Runs the forward pass of the neural network to get to the output layer
    def forward_propagation(self, X, weights, biases, activation_func):
        activations = [X]

        activation_functions = {
            'tanh': self.tanh,
            'sigmoid': self.sigmoid,
            'relu': self.relu,
            'linear': self.linear
        }
        
        for i in range(len(weights)):
            input_weighted = np.dot(activations[-1], weights[i]) + biases[i]
            output_activation = activation_functions[activation_func](input_weighted)
            activations.append(output_activation)

        return activations

    # Runs the backward pass and computes the updated weights and biases, adn includes regularization
    # to prevent overfitting
    # Accounts for the derivative of the specified activation function
    def back_propagation(self, X, y, activations, weights, activation_func, alpha, regularization):
        len_X = X.shape[1]
        deltas = [activations[-1] - y]

        activation_functions_ddx = {
            'tanh': self.tanh_ddx,
            'sigmoid': self.sigmoid_ddx,
            'relu': self.relu_ddx,
            'linear': self.linear_ddx
        }
        
        for i in range(len(weights) -1 , 0, -1):
            input_weighted_ddx = np.dot(deltas[-1], weights[i].T)
            output_activation_ddx = activation_functions_ddx[activation_func](activations[i])
            
            delta = input_weighted_ddx * output_activation_ddx
            deltas.append(delta)

        deltas.reverse()

        if regularization == 'l1':
            weights_ddx = [(np.dot(activations[i].T, deltas[i]) + alpha * np.sign(weights[i])) / len_X for i in range(len(weights))]
        elif regularization == 'l2':
            weights_ddx = [(np.dot(activations[i].T, deltas[i]) + 2 * alpha * weights[i]) / len_X for i in range(len(weights))]
        else:
            weights_ddx = [(np.dot(activations[i].T, deltas[i])) / len_X for i in range(len(weights))]
        
        biases_ddx = [np.sum(deltas[i], axis=0, keepdims=True) / len_X for i in range(len(weights))]

        return weights_ddx, biases_ddx
    
    # Standard optimizer used to compute the weights adn biases
    def sgd_optimizer(self, weights, biases, weights_ddx, biases_ddx, learning_rate):
        for i in range(len(weights)):
            weights[i] -= learning_rate * weights_ddx[i]
            biases[i] -= learning_rate * biases_ddx[i]

        return weights, biases

    # A better optimizer that uses the moment generating function, means, and variances to converge. Hyperparameters
    # can be changes as needed when initalizing the object
    def adam_optimizer(self, weights, biases, weights_ddx, biases_ddx, learning_rate):
        mean_weights = [np.zeros_like(w) for w in weights]
        var_weights = [np.zeros_like(w) for w in weights]
        mean_biases = [np.zeros_like(b) for b in biases]
        var_biases = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            mean_weights[i] = self.beta_1 * mean_weights[i] + (1 - self.beta_1) * weights_ddx[i]
            var_weights[i] = self.beta_2 * var_weights[i] + (1 - self.beta_2) * (weights_ddx[i]**2)
            mean_biases[i] = self.beta_1 * mean_biases[i] + (1 - self.beta_1) * biases_ddx[i]
            var_biases[i] = self.beta_2 * var_biases[i] + (1 - self.beta_2) * (biases_ddx[i]**2)

            mean_weights_hat = mean_weights[i] / (1 - np.power(self.beta_1, (self.epoch + 1)))
            var_weights_hat = var_weights[i] / (1 - np.power(self.beta_2, (self.epoch + 1)))
            mean_biases_hat = mean_biases[i] / (1 - np.power(self.beta_1, (self.epoch + 1)))
            var_biases_hat = var_biases[i] / (1 - np.power(self.beta_2, (self.epoch + 1)))

            weights[i] -= learning_rate * mean_weights_hat / (np.sqrt(var_weights_hat) + self.epsilon)
            biases[i] -= learning_rate * mean_biases_hat / (np.sqrt(var_biases_hat) + self.epsilon)
        
        return weights, biases

    # Trains the neural network and runs the forward and backward passes. Early stopping was included to prevent
    # long run times
    # and for faster convergence
    def nn_train(self, X, y, verbose=True):
        input_size = X.shape[1]
        output_size = y.shape[1]
        
        self.losses = []
        self.accuracies = []
        self.epoch = 0
        self.weights, self.biases = self.weights_biases(input_size, self.hidden_layer_sizes, output_size, self.initialization)

        optimizers = {
            'adam': self.adam_optimizer,
            'sgd': self.sgd_optimizer
        }

        loss_prev = np.inf
        tol_counter = 0

        for _ in range(self.epochs):
            activations = self.forward_propagation(X, self.weights, self.biases, self.activation_func)
            weights_ddx, biases_ddx = self.back_propagation(X, y, activations, self.weights, self.activation_func, self.alpha, self.regularization)

            y_pred = activations[-1]
            loss = self.log_loss(y, y_pred)
            
            if np.isnan(y_pred).any() or np.isnan(loss):
                print("WARNING: Training stopped due to NaN values in computation. Please adjust Hyperparameters.")
                loss = 0
                break
            
            loss_change = loss_prev - loss
            if abs(loss_change) < self.tolerance:
                tol_counter += 1
            else:
                tol_counter = 0
            
            if tol_counter >= self.tolerance_counter:
                print(f"WARNING: Training stopped due to {tol_counter} consecutive epochs with no change in loss.")
                break

            if self.activation_func == 'sigmoid':
                accuracy = accuracy_score(y, np.round(y_pred))
            else:
                accuracy = accuracy_score(y, np.round(self.sigmoid(y_pred)))

            current_learning_rate = self.learning_rate_scheduler(self.learning_rate, self.lr_decay_type, self.lr_decrease, self.lr_epoch_drop)
            weights_ddx, biases_ddx = self.back_propagation(X, y, activations, self.weights, self.activation_func, self.alpha, self.regularization)
            self.weights, self.biases = optimizers[self.optimizer](self.weights, self.biases, weights_ddx, biases_ddx, learning_rate=current_learning_rate)
            
            if verbose == True:
                print(f"Epoch {self.epoch + 1:3d} ==================> Loss: {loss:7.4f}, Accuracy: {accuracy:6.4f}")
            
            loss_prev = loss
            self.accuracies.append(accuracy)
            self.losses.append(loss)

            self.epoch += 1
            
        return self.weights, self.biases

    # Runs a final forward pass with the best weights and biases to make predictions
    def nn_predict(self, X):
        outputs = self.forward_propagation(X, self.weights, self.biases, self.activation_func)
        self.class_outputs = outputs[-1]
        
        if self.activation_func == 'sigmoid':
            y_pred = np.nan_to_num(np.round(self.class_outputs))
        else:
            y_pred = np.nan_to_num(np.round(self.sigmoid(self.class_outputs)))
        
        return y_pred

    # Gets the predicition probabilites of the classes
    def nn_predict_proba(self, X=None):
        if X is not None:
            logits = self.forward_propagation(X, self.weights, self.biases, self.activation_func)[-1]
        else:
            logits = self.class_outputs
        
        sigmoid_logits = self.sigmoid(logits)

        prob_pos_class = sigmoid_logits
        prob_neg_class = 1 - sigmoid_logits

        prediction_probabilities = np.nan_to_num(np.column_stack((prob_neg_class, prob_pos_class)))

        return prediction_probabilities
    
    # Creates the plots of loss and training curves from nn_train
    def plots(self):
        if len(self.losses) > 100:
            fig, axs = plt.subplots(2, 1, figsize=(8, 4))
        else:
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].plot(self.losses, label='Loss', color='#1f77b4')
        axs[0].set_xlabel("Epochs")
        axs[0].set_title("Loss (Train)")

        axs[1].plot(self.accuracies, label='Accuracy', color='green')
        axs[1].set_xlabel("Epochs")
        axs[1].set_title("Accuracy (Train)")

        fig.tight_layout()
        plt.show()
