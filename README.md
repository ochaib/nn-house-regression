# Neural Networks, Group 53

The full training script can be ran easily by calling the part 2 script as follows:

```
python3 part2_house_value_regression.py
```

This will train a model with our predefined hyperparameters.

## Training

First, simply create a regressor from your training data:

``` python
regressor = Regressor(x_train, batch_size=32, learning_rate=0.005, nb_epoch = 10000)
```

Then fit the regressor:
``` python
regressor.fit(x_train, y_train)
```

## Prediction and Evaluation

Predicting house prices is fairly straightforward. First normalize your features:
``` python
x_test = self._preprocessor(x_test)
```

Then simply use the provided `predict` method which will also de-normalize (scale up) the Y values back to housing prices:

``` python
y_preds = self.predict(x_test)
```

Evaluating is similarly simple, provided you have test data handy:
``` python
error = regressor.score(x_test, y_test)
```


## hyperparameter Search

Running grid search can be done via the helper function on your training and testing datasets:

``` python
params = RegressorHyperParameterSearch(x_train, y_train, x_test, y_test)
```

By default the following hyperparameter grid is used in testing:
``` python
param_grid = {
    'hidden_layers': [[15, 17, 15], [20, 32, 20], [100, 180, 100], [20, 32, 64, 32, 20]],
    'batch_size': [32, 64, 128, 256],
    'lr': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
    'nb_epoch': [100, 500, 1000]
}
```

You may alter these as you see fit to modify which parameters are fine-tuned by the grid search.