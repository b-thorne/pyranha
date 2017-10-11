# Fisher forecasts for B-modes and foreground removal

`pyranha` is an extremely simple Fisher code for forecasting constraints on the tensor-to-scalar ratio under incomplete foreground removal. The model for adding foreground residuals is described in [](Thorne et al 2017), which is itself based on ![]().

## Configuration files

An example configuration file, with all the current options labelled and explained, can be found in `configurations/example.py`.

## Example

An example of how to run the code, and the various facilities it provides, can be found in the [IPython notebook](example.ipynb) in this directory.

## Processing

- We will want to iterate through the parameters and calculate the derivative matrix for each parameter.
- Will need to assemble the Fisher matrix and invert it.
-
