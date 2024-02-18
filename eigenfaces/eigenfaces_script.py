"""This script is used to run the eigenfaces algorithm."""
import matplotlib.pyplot as plt
import numpy as np
from eigenfaces.eigenfaces_lib import images_to_A, plot_eigenfaces, plot_projections, project_image

# People that will be used
people = {"person_1": "stallone", "person_2": "taylor"}
test_imgs = {
    "person_1": "faces/teststallone1.jpg",
    "person_2": "faces/testtaylor1.jpg",
}

# people = {"person_1": "stallone", "person_2": "arnold"}
# test_imgs = {
#     "person_1": "faces/teststallone1.jpg",
#     "person_2": "faces/testterminator2.jpg",
# }

# people = {"person_1": "arnold", "person_2": "taylor"}
# test_imgs = {
#     "person_1": "faces/teststallone1.jpg",
#     "person_2": "faces/testtaylor1.jpg",
# }
# Images pixel dimensions
m = 200
n = 175
img_dim = (m, n)

# Number of sample pictures
N = 20

# First eigenface to use for projection
start_ef = 0

# taylor arnold start_ef=4
# stallone arnold start_ef=1
# taylor stallone start_ef=1


# Print execution details on stdout
print(f"Using: {people['person_1']} and {people['person_2']}")
print(f"Image dimensions: {img_dim}")
print(f"Number of images per person: {N}")
print(f"First eigenface to use: {start_ef}")


A_centered = images_to_A(show_avg_face=False, N=N, img_dim=img_dim, people=people)
print("Images matrix shape:")
print(A_centered.shape)

# Computing the SVD
U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
Phi = U[:, : 2 * N]

# Plot Eigenfaces
plot_eigenfaces(Phi=Phi, img_dim=img_dim)
plt.show()

# Project faces of person_1 and person_2 to 3 first eigenfaces,
# starting at start_ef
PER_1 = (A_centered[:, :N].T @ Phi[:, start_ef : start_ef + 3]).T
PER_2 = (A_centered[:, N : 2 * N].T @ Phi[:, start_ef : start_ef + 3]).T

# plot in 1D
plot_projections(num_dim=1, start_ef=start_ef, PER_1=PER_1, PER_2=PER_2, people=people)
plt.show()

# plot in 2D
plot_projections(num_dim=2, start_ef=start_ef, PER_1=PER_1, PER_2=PER_2, people=people)
plt.show()

# plot in with one more eigenface, 3D
plot_projections(num_dim=3, start_ef=start_ef, PER_1=PER_1, PER_2=PER_2, people=people)
plt.show()

# Project test pictures not seen in SVD
per1_points = project_image(
    filepath=test_imgs["person_1"],
    num_dim=3,
    start_ef=start_ef,
    Phi=Phi,
    img_dim=img_dim,
)
per2_points = project_image(
    filepath=test_imgs["person_2"],
    num_dim=3,
    start_ef=start_ef,
    Phi=Phi,
    img_dim=img_dim,
)

# Plotting the test points
ax = plot_projections(
    num_dim=3, start_ef=start_ef, PER_1=PER_1, PER_2=PER_2, people=people
)
ax.scatter(
    per1_points[0], per1_points[1], per1_points[2], c="brown", label=people["person_1"]
)
ax.scatter(
    per2_points[0], per2_points[1], per2_points[2], c="aqua", label=people["person_2"]
)
ax.legend()
plt.show()
