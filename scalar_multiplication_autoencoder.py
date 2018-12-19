import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

#__ create autoencoder model
def model_autoencoder(input_dim, encoding_dim):
    #.. input placeholder
    input_data = Input(shape=(input_dim,))
    #.. encoded representation of the input
    encoded = Dense(encoding_dim, activation='linear')(input_data)
    #.. lossy reconstruction of the input
    decoded = Dense(input_dim, activation='linear')(encoded)

    #.. autoencoder
    #.. this model maps an input to its reconstruction
    autoencoder = Model(input_data, decoded)

    #.. encoder
    #.. this model maps an input to its encoded representation
    encoder = Model(input_data, encoded)

    #.. create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    #.. retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]

    #.. decoder
    #.. create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='nadam', loss='mean_squared_error')

    return encoder, decoder, autoencoder

#__ read a text file and get plot data
def get_data_file(filename):
    import csv

    xs = np.asarray([])
    ys = np.asarray([])
    zs = np.asarray([])
    with open(filename, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            row = list(map(float, row))
            xs = np.append(xs, row[0])
            ys = np.append(ys, row[1])
            zs = np.append(zs, row[2])
            
    return xs, ys, zs

#__ transform three sequences to a 3-dimensional sequence
def from_1d_to_3d(xs, ys, zs):
    data3d = np.asarray([])
    for x, y, z in zip(xs, ys, zs):
        data3d = np.append(data3d, [x, y, z])
    data3d = data3d.reshape(len(xs),3)

    return data3d

#__ transform a 3-dimensional sequence to three sequences
def from_3d_to_1d(data3d):
    xs = np.asarray([])
    ys = np.asarray([])
    zs = np.asarray([])

    for i in range(data3d.shape[0]):
        xs = np.append(xs, data3d[i,0])
        ys = np.append(ys, data3d[i,1])
        zs = np.append(zs, data3d[i,2])

    return xs, ys, zs

#__ plot a 3-dimensional data
def plot_3Dgraph(xs, ys, zs, c):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(xs, ys, zs, "x",
            label='sphere lattice', color=c)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    #.. dimension of input data space
    input_dim = 3
    #.. size of our encoded representations
    encoding_dim = 3

    #.. get autoencoder model
    encoder, decoder, autoencoder = \
    model_autoencoder(input_dim, encoding_dim)

    #.. get input train data
    xs_train1, ys_train1, zs_train1 = \
    get_data_file('data/sphere_plot_r1.dat')
    train_data1 = \
    from_1d_to_3d(xs_train1, ys_train1, zs_train1)

    # get target train data
    xs_train2, ys_train2, zs_train2 = \
    get_data_file('data/sphere_plot_r2.dat')
    train_data2 = \
    from_1d_to_3d(xs_train2, ys_train2, zs_train2)

    # get test data
    xs_test, ys_test, zs_test = \
    get_data_file('data/sphere_plot_r100.dat')
    test_data = \
    from_1d_to_3d(xs_test, ys_test, zs_test)

    print("train1 shape", train_data1.shape)
    print("train2 shape", train_data2.shape)
    print("test shape", test_data.shape)

    # fit autoencoder
    autoencoder.fit(train_data1, train_data2,
                    epochs=40,
                    batch_size=None,
                    shuffle=False)

    #.. choose a basis of 3-dimensional Euclidean space
    basis = np.asarray([])
    basis = np.append(basis, [1.0, 0.0, 0.0])
    basis = np.append(basis, [0.0, 1.0, 0.0])
    basis = np.append(basis, [0.0, 0.0, 1.0])
    basis = basis.reshape(3,3)
    print("A basis:")
    print(basis)

    #.. encoded test data
    encoded_data = encoder.predict(test_data)
    encoded_basis = encoder.predict(basis)

    #.. reconstructed test data
    decoded_data = decoder.predict(encoded_data)
    decoded_basis = decoder.predict(encoded_basis)

    print("This basis is mapped to:")
    print(decoded_basis)

    #.. plot reconstructed data
    xs, ys, zs = from_3d_to_1d(decoded_data)
    plot_3Dgraph(xs, ys, zs, 'blue')
