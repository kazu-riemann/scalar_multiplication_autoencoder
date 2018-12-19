import numpy as np

"""
(x,y,z) = (r cos(phi) cos(theta), r cos(phi) sin(phi), r sin(phi))
r: fixed
theta in [0, 2 pi) <-- [0, 2)
phi in [-0.5 pi, 0.5 pi] <-- [-0.5, 0.5]
"""

pi=3.14159265358979

def make_sequence(init, div_num, delta):
    num = 0
    sequence = np.asarray([])
    value = init + delta
    sequence = np.append(sequence, value)
    while(num < div_num):
        num += 1
        value += 2.0 * delta
        sequence = np.append(sequence, value)

    return sequence

def make_lattice(lower_theta, upper_theta, divide_num_theta, \
                 lower_phi, upper_phi, divide_num_phi):
    delta_theta = (upper_theta - lower_theta) / (2.0 * divide_num_theta)
    sequence_theta = make_sequence(lower_theta, divide_num_theta, delta_theta)
    delta_phi = (upper_phi - lower_phi) / (2.0 * divide_num_phi)
    sequence_phi = make_sequence(lower_phi, divide_num_phi, delta_phi)

    return sequence_theta, sequence_phi

def parametrize_sphere(r, arg_theta, arg_phi):
    theta = arg_theta * pi
    phi = arg_phi * pi
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)

    return x, y, z

def plot_3Dgraph(xs, ys, zs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(xs, ys, zs, label='sphere lattice')
    ax = Axes3D(fig)
    ax.plot(xs, ys, zs, "x", label='sphere lattice')
    ax.legend()
    plt.show()
    
def write_data_file(xs, ys, zs):
    import csv

    with open("sphere_plot.dat", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for x, y, z, in zip(xs, ys, zs):
            writer.writerow([x, y, z])

if __name__ == "__main__":
    r = 100.0
    lower_theta = 0.0
    upper_theta = 2.0
    divide_num_theta = 50
    lower_phi = -0.5
    upper_phi = 0.5
    divide_num_phi = 25
    
    thetaseq, phiseq = \
    make_lattice(lower_theta, upper_theta, divide_num_theta, \
                 lower_phi, upper_phi, divide_num_phi)

    xseq = np.asarray([])
    yseq = np.asarray([])
    zseq = np.asarray([])

    x = 0.0
    y = 0.0
    z = 0.0

    for theta in thetaseq:
        for phi in phiseq:
            x, y, z = parametrize_sphere(r, theta, phi)
            xseq = np.append(xseq, x)
            yseq = np.append(yseq, y)
            zseq = np.append(zseq, z)

    # plot_3Dgraph(xseq, yseq, zseq)
    write_data_file(xseq, yseq, zseq)
