# computer_graphics_code
import matplotlib.pyplot as plt
import numpy as np

def oblique_projection(vertices, angle=45, scale=1.0):
    """
    Apply an oblique (cavalier) projection to a set of 3D vertices.
    
    :param vertices: A numpy array of shape (n, 3) representing the vertices of the 3D object.
    :param angle: The angle in degrees for the projection lines relative to the horizontal plane.
    :param scale: The scale factor for the depth (typically 1.0 for cavalier projection).
    :return: A numpy array of shape (n, 2) representing the 2D projected vertices.
    """
    # Convert the angle to radians
    angle_rad = np.radians(angle)
    
    # Define the oblique projection matrix
    projection_matrix = np.array([
        [1, 0, scale * np.cos(angle_rad)],
        [0, 1, scale * np.sin(angle_rad)]
    ])
    
    # Apply the projection matrix to the vertices
    projected_vertices = vertices @ projection_matrix.T
    
    return projected_vertices

# Define the vertices of a 3D object (e.g., a cube)
cube_vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

# Apply the oblique projection to the vertices
projected_vertices = oblique_projection(cube_vertices)

# Print the original 3D vertices and their corresponding projected 2D vertices
print("Original 3D vertices and projected 2D vertices:")
for i, (v3d, v2d) in enumerate(zip(cube_vertices, projected_vertices)):
    print(f'P{i}: 3D ({v3d[0]:.2f}, {v3d[1]:.2f}, {v3d[2]:.2f}) -> 2D ({v2d[0]:.2f}, {v2d[1]:.2f})')

# Define the edges of the cube
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
]

# Plot the projected vertices and edges
plt.figure(figsize=(8, 8))
for edge in edges:
    start, end = edge
    plt.plot(
        [projected_vertices[start, 0], projected_vertices[end, 0]],
        [projected_vertices[start, 1], projected_vertices[end, 1]],
        'bo-'  # blue color, circle marker, solid line
    )

# Annotate the projected points
for i, (x, y) in enumerate(projected_vertices):
    plt.text(x, y, f'P{i}', fontsize=12, ha='right')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Oblique Projection (Cavalier Case) of a Cube')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
