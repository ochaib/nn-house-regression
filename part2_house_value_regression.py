import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pickle
import random
import time
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from torch import nn
from torch.nn.init import xavier_uniform_
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ParameterGrid

class Net(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        # Create a basic NN model, initializing weights with xavier uniform
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for idx, size in enumerate(hidden_sizes):
            if idx == 0:
                self.layers.append(nn.Linear(input_size, size))
            else:
                self.layers.append(nn.Linear(hidden_sizes[idx-1], size))
            xavier_uniform_(self.layers[idx].weight)
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        

    def forward(self, hidden):
        output = hidden
        for layer in self.layers[:-1]:
            output = nn.functional.relu(layer(output))
        output = self.layers[-1](output)
        return output

    # def __init__(self, input_size, output_size):
    #     # Create a basic NN model, initializing weights with xavier uniform
    #     super(Net, self).__init__()
    #     self.hidden_1 = nn.Linear(input_size, 20)
    #     xavier_uniform_(self.hidden_1.weight)
    #     self.hidden_2 = nn.Linear(20, 32)
    #     xavier_uniform_(self.hidden_2.weight)
    #     self.hidden_3 = nn.Linear(32, 20)
    #     xavier_uniform_(self.hidden_3.weight)
    #     self.output_layer = nn.Linear(20, output_size)

    # def forward(self, input):
    #     hidden_1_output = nn.functional.relu(self.hidden_1(input))
    #     hidden_2_output = nn.functional.relu(self.hidden_2(hidden_1_output))
    #     hidden_3_output = nn.functional.relu(self.hidden_3(hidden_2_output))
    #     output = self.output_layer(hidden_3_output)
    #     return output


class Regressor():

    def __init__(self, x=np.zeros((10,10)), hidden_layers = [20, 32, 20], batch_size=1, learning_rate=0.01, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Cache values we impute into NaN values
        self.impute_value = None
        # Encoding objects to fit then transform on data.
        self.categorical_columns = ['ocean_proximity']
        self.label_binarizer = None
        self.category_less_x = x.select_dtypes(['number'])
        self.x_min_max_scaler = None
        # self.y_min_max_scaler = None

        X, _ = self._preprocessor(x, training = True)
        # Initialise neural network model and all attributes needed.
        # Apply preprocessor method to set dimensions of NN model.
        self.input_size = X.shape[1]
        self.hidden_sizes = hidden_layers
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Initialise neural net
        self.net = Net(self.input_size, self.hidden_sizes, self.output_size)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size
                (batch_size, 1).

        """


        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Impute/Fill the missing values with the mean:
        # Could also impute the value using median, KNN or MICE.
        if training:
            self.impute_value = x.mean()
        data = x.fillna(self.impute_value)

        # Use label binarizer to convert categorical values to one hot
        # encoded values.
        if training:
            # for col in categorical_columns:
            self.label_binarizer = preprocessing.LabelBinarizer().fit(data['ocean_proximity'])

        # Continue with transforming cat cols as label_binarizer has already been initialised
        one_hot_encodings = self.label_binarizer.transform(data['ocean_proximity'])

        column_labels = self.label_binarizer.classes_
        for i, column_label in enumerate(column_labels):
            data[column_label] = one_hot_encodings[:, i]
        # Drop categorical columns ('ocean_proximity_cat') column
        data = data.drop('ocean_proximity', axis=1)

        if training:
            # Min max scale each column individually
            self.x_min_max_scaler = preprocessing.MinMaxScaler().fit(data.values)
            # if isinstance(y, pd.DataFrame):
            #     self.y_min_max_scaler = preprocessing.MinMaxScaler().fit(y.values)

        # Apply min-max normalisation
        # Convert to numpy array
        column_labels = list(data.columns)
        # Transform the values of the array by the min max scaler existing
        data_np_scaled = self.x_min_max_scaler.transform(data.values)
        # Convert back to Pandas DataFrame
        data = pd.DataFrame(data_np_scaled, index=data.index, columns=data.columns)
        # Set columns back to column names
        data.columns = column_labels

        # if isinstance(y, pd.DataFrame):
        #     # Apply min-max normalisation
        #     # Convert to numpy array
        #     y_labels = list(y.columns)
        #     # Transform the values of the array by the min max scaler existing
        #     y_np_scaled = self.y_min_max_scaler.transform(y.values)
        #     # Convert back to Pandas DataFrame
        #     y = pd.DataFrame(y_np_scaled, index=y.index, columns=y.columns)
        #     # Set columns back to column names
        #     y.columns = y_labels

        # Convert pandas dataframes into torch tensors:
        x_tensor = torch.tensor(data.values)

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        return x_tensor, (torch.tensor(y.values) if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Seed necessary for reproducibility of randomisation
        torch.manual_seed(0)

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        # Related to batch processing
        # Convert the X, Y preprocessed datasets into a torch dataset:
        # Here we wrap the torch tensors X, Y in order to record operations
        # performed on them for automatic differentiation.
        torch_dataset = Data.TensorDataset(Variable(X), Variable(Y))

        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.batch_size)

        # Loss function
        loss_func = nn.MSELoss()

        # Adam optimiser with hyperparameter learning rate.
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Log gradients and model parameters

        # Perform the following nb_epoch times:
        for _ in range(self.nb_epoch):

            # Sets the module in training mode
            self.net.train()

            # Necessary for processing in batches
            for step, (batch_x, batch_y) in enumerate(loader):
                # Perform forward pass through the model given the input
                prediction = self.net(Variable(batch_x).float())

                # Clear previous gradients
                optimiser.zero_grad()

                # Compute loss based on this forward pass, comparing resulting prediction to
                # actual output label y.
                loss = loss_func(prediction, Variable(batch_y).float())

                # Perform backward pass to compute gradients of loss wrt params
                # Compute gradients using back propagation
                loss.backward()

                # Perform one step of gradient descent on the model params
                optimiser.step()


        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False)

        loader = Data.DataLoader(
            dataset=X,
            batch_size=self.batch_size)

        # The regressor model has been trained...

        # Now using a new unseen test dataset, predict test labels for each
        # row...

        # Outputs as numpy array
        predictions = []

        # Pass the data through the model to generate a prediction
        # Do not perform gradient descent while applying predictions
        with torch.no_grad():
            
            for sample in loader:

                pred = self.net(sample.float())

                predictions = np.append(predictions, pred)
                # predictions.append(np.asarray())



        return predictions

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def mse_r2_var(self, x_test, y_test):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X_test, Y_test = self._preprocessor(x_test, y = y_test, training = False) # Do not forget

        # Get prediction from x_test, x_test is preprocessed in predict so 
        # not necessary to pass in X_test here.
        y_preds = self.predict(x_test)
        # Scale predictions back
        # y_preds = self.y_min_max_scaler.inverse_transform(pd.DataFrame(y_preds))

        # Scale Y_test back
        # Y_test = self.y_min_max_scaler.inverse_transform(pd.DataFrame(Y_test.numpy()))
        Y_test = pd.DataFrame(Y_test.numpy())

        mse = metrics.mean_squared_error(Y_test, y_preds)
        rmse = metrics.mean_squared_error(Y_test, y_preds, squared=False)
        r2 = metrics.r2_score(Y_test, y_preds)
        variance = np.std(y_preds)

        print("MSE: ", mse)
        print("RMSE: ", rmse)
        print("R2: ", r2)
        print("Variance: ", variance)

        return (mse, r2, variance)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x_test, y_test):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        return self.mse_r2_var(x_test, y_test)[0]

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################



def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(X, y, x_test, y_test):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # Grid of hyperparameters that will be randomly sampled when calling
    # GridSearchCV().
    # Could also include optimisation algorithm e.g. SGD, Adam, RmsProp
    param_grid = {
        'hidden_layers': [[20, 32, 20]],
        'batch_size': [32, 64],
        'lr': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
        'nb_epoch': [10]
    }


    # Use ParameterGrid to perform a full parallelised parameter search.
    grid = ParameterGrid(param_grid)

    optimal_parameters = []
    lowest_error = math.inf
    best_regressor = None

    param_results = []

    for params in grid:
        row = {}
        
        for key in param_grid:
            row[key] = params[key]

        # print(params)
        regressor = Regressor(X, params['hidden_layers'], params['batch_size'], params['lr'], params['nb_epoch'])
        regressor.fit(X, y)
        # error = regressor.score(x_test, y_test)
        (error, r2, var) = regressor.mse_r2_var(x_test, y_test)
        # print(error)

        row['mse'] = error
        row['r2'] = r2
        row['var'] = var
        param_results.append(row)

        if error < lowest_error:
            lowest_error = error
            optimal_parameters = params
            best_regressor = regressor
    
    param_results = pd.DataFrame(param_results)

    for x in param_grid:
        if x == 'hidden_layers':
            continue
        for y in ['mse', 'r2', 'var']:
            # print(param_results[x])
            # print(param_results[y])
            plt.plot(param_results[x], param_results[y], '.b')
            plt.suptitle("{} vs. {}".format(y, x))
            plt.xlabel(x)
            plt.ylabel(y)
            plt.savefig('plots/{}vs{}.png'.format(y, x))
            plt.clf()
            # plt.show()

    save_regressor(best_regressor)

    return optimal_parameters

    # Why doesn't this work
    regressor = Regressor()
    # GridSearchCV exhaustively generates candidates from a grid of
    # paraemeters specified in param_grid
    # Specify the scoring parameter to GridSearchCV, in our case, 
    # MSE.
    grid = GridSearchCV(regressor, param_grid, cv=5)
    grid.fit(X, y)

    best_params = grid.best_params_
    print(f"Best Params: {best_params}")
    best_model = grid.best_estimator_
    print(f"Best Model: {best_model}")
    best_score = grid.best_score_
    print(f"Best Score: {best_score}")

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Retrieve non-numeric (categorical) columns of the dataset:
    # categorical_columns = list(data.select_dtypes(include=['object']).columns)[0]
    # Convert object dtype columns to category dtype columns
    data['ocean_proximity'] = data['ocean_proximity'].astype('category')

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Split dataset into training, validation and test data
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)
    # If we need a validation dataset then uncomment the following:
    # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, batch_size=32, learning_rate=0.005, nb_epoch = 1000)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

    # hyper_parameters = RegressorHyperParameterSearch(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
