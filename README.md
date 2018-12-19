# Scalar multiplication autoencoder
An implementation of scalar multiplication by autoencoder.
Our implementation of autoencoder refers to [[1]].
The architecture of the network is modified as follows.

* input layer size = 3
* hidden layer size = 3 (only 1 layer)
* output layer size = 3

From the following procedure, we demonstrate that
our autoencoder approximates the twice multiplication mapping.

1. Train autoencoder using input data as radius 1 sphere data
and target data as radius 2 sphere data.

2. Input radius 100 sphere data to the trained autoencoder.

3. Input a basis of 3-dimentional Euclidean space
[[1,0,0], [0,1,0], [0,0,1]] to the trained autoencoder.

As the result of procedure 2, we get data approximating
radius 200 sphere data as the output.
As the result of procedure 3, we get a basis of 3-dimentional Euclidean space
that approximates the basis [[2,0,0], [0,2,0], [0,0,2]].

## Requirements
* numpy==1.15.4
* matplotlib==3.0.0
* Keras==2.2.4

## Usage
We assume the current directory contains the data directory
data/. Enter the following command.
```console
$ python scalar_multiplication_autoencoder.py
```

## References
[[1]] Building Autoencoders in Keras by F. Chollet.
https://blog.keras.io/building-autoencoders-in-keras.html

[1]: https://blog.keras.io/building-autoencoders-in-keras.html