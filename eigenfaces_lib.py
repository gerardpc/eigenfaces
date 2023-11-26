import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def img_to_array(filepath: str, img_dim: tuple) -> np.ndarray:
    """Function to read and process images.

    Args:
        filepath: path to the image file
        img_dim: dimensions of the image in pixels

    Returns:
        numpy array with the face (in greyscale)
    """
    img = Image.open(filepath)
    # image is converted to greyscale
    img = img.convert('L') if img.mode != 'L' else img
    img_data = np.asarray(img, dtype=np.float64).reshape(img_dim[0]*img_dim[1], 1)
    return img_data


def project_image(
        filepath: str,
        Phi: np.ndarray,
        num_dim: int,
        start_ef: int,
        img_dim: tuple
) -> np.ndarray:
    """Project pics not in train dataset to first n eigenfaces.

    Args:
        filepath: path to image file
        Phi: eigenfaces matrix
        num_dim: 2 or 3 (eigenfaces)
        start_ef: integer to specify the first eigenface
            that gets projected. Eigenfaces used will be
            [start_ef:start_ef + num_dim]
        img_dim: dimensions of the image in pixels
    """
    img_data = img_to_array(filepath, img_dim)
    img_data = img_data - img_data.mean()
    img_data = img_data / img_data.std()
    return (img_data.T @ Phi[:, start_ef:start_ef + num_dim]).T


def images_to_A(
        img_dim: tuple,
        people: dict[str, str],
        N: int,
        show_avg_face: bool = False,
) -> np.ndarray:
    """Load images, process them and stack in matrix as
    column vectors.

    Args:
        img_dim: dimensions of the image in pixels
        people: dictionary of person's names
        N: number of faces per person
        show_avg_face: if True, plot avg face.
    """
    # the average face
    avg = np.zeros((img_dim[0] * img_dim[1], 1))
    A = np.empty((img_dim[0] * img_dim[1], 0))
    # Load images and calculate average
    for person, person_name in people.items():
        for j in range(1, N + 1):
            file_name = f'faces/{person_name}{str(j).zfill(2)}.jpg'
            if os.path.exists(file_name):
                R = img_to_array(file_name, img_dim=img_dim)
                R = R - R.mean()
                R = R / R.std()
                A = np.append(A, R, axis=1)
                avg += R

    # avg := avg of all N faces of 2 people (spaggetified)
    avg = avg / (2 * N)

    # Calculate the "averaged" face (as m x n matrix)
    avgTS = np.reshape(avg, img_dim)
    if show_avg_face:
        plt.imshow(avgTS, cmap='gray')
    # Center the sample pictures at the "origin"
    A_centered = A - avg
    return A_centered


def plot_eigenfaces(
        Phi: np.ndarray,
        img_dim: tuple,
):
    """Plot first 9 eigenfaces.

    Args:
        Phi: matrix of eigenfaces
        img_dim: dimensions of the image in pixels
    """
    fig, ax = plt.subplots(3, 3)
    fig.suptitle("First 9 eigenfaces")
    count = 1
    for i in range(3):
        for j in range(3):
            eigenface = np.reshape(Phi[:, count - 1], img_dim)
            # to give more contrast on plot we multiply by m x n
            eigenface = eigenface * img_dim[0] * img_dim[1]
            ax[i, j].imshow(eigenface, cmap='gray')
            count += 1


def plot_projections(
        PER_1: np.ndarray,
        PER_2: np.ndarray,
        people: dict[str, str],
        num_dim: int,
        start_ef: int
) -> plt.axes:
    """Plot the first num_dim components of a face.

    Args:
        PER_1: numpy vectors of person_1
        PER_2: numpy vectors of person_2
        num_dim: 2 or 3 (eigenfaces)
        start_ef: integer to specify the first eigenface
            that gets projected. Eigenfaces used will be
            [start_ef:start_ef + num_dim]
    """
    if num_dim == 2:
        # 2 components plot
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(PER_1[0, :], PER_1[1, :], c='r', label=people["person_1"])
        ax.scatter(PER_2[0, :], PER_2[1, :], c='b', label=people["person_2"])
        ax.set_xlabel(f'eigenface {start_ef + 1}')
        ax.set_ylabel(f'eigenface {start_ef + 2}')
        ax.set_title("2D projection on 2 eigenfaces")
        ax.legend()

    elif num_dim == 3:
        # 3 components plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(PER_1[0, :], PER_1[1, :], PER_1[2, :], c='r', label=people["person_1"])
        ax.scatter(PER_2[0, :], PER_2[1, :], PER_2[2, :], c='b', label=people["person_2"])
        ax.set_xlabel(f'eigenface {start_ef + 1}')
        ax.set_ylabel(f'eigenface {start_ef + 2}')
        ax.set_zlabel(f'eigenface {start_ef + 3}')
        ax.set_title("3D projection on 3 eigenfaces")
        ax.legend()
    else:
        print("The number of dimensions must be 2 or 3.")
        return
    return ax
