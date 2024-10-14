import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy.linalg as LA
import math

class DrawingApp:
    """Application to draw a closed curve and simulate Curve Shortening Flow."""
    def __init__(self, root):
        """Initialize the drawing application."""
        self.root = root
        self.root.title("Drawing App")

        self.canvas = tk.Canvas(root, bg="white", width=200, height=200)
        self.canvas.pack()

        self.curve = []
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.reset_button = tk.Button(root, text="Reset", command=self.reset_canvas)
        self.reset_button.pack()

    def paint(self, event):
        """Draw on the canvas when the left mouse button is held down."""
        if 0 <= event.x <= 200 and 0 <= event.y <= 200:
            x, y = event.x, event.y
            self.curve.append((x, y))
            self.canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill="black", width=2)

    def stop_drawing(self, event):
        """Stop drawing and start the Curve Shortening Flow simulation."""
        if len(self.curve) > 2:
            self.curve.append(self.curve[0])  # Close the curve
            self.draw_closed_curve()
            self.start_csf_simulation()

    def reset_canvas(self):
        """Reset the canvas for a new drawing."""
        self.canvas.delete("all")
        self.curve = []

    def draw_closed_curve(self):
        """Draw the closed curve on the canvas."""
        self.canvas.create_line(self.curve, fill="black", width=2)

    def start_csf_simulation(self):
        """Start the Curve Shortening Flow simulation."""
        self.root.destroy()  # Close the drawing app window
        points = self.curve
        points = chaikin_smooth(points)  # Smooth the points
        points = np.array(points)
        contract_points(points)
        num_steps = 2000
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(0, 200)
        ax.set_ylim(200, 0)  # Invert y-axis to match tkinter's coordinate system
        line, = ax.plot(points[:, 0], points[:, 1])

        def update(frame):
            nonlocal points
            points = contract_points(points, scale_factor=1.0)
            line.set_data(points[:, 0], points[:, 1])
            if len(points) <= 5:
                ani.event_source.stop()
            return line,

        ani = FuncAnimation(fig, update, frames=num_steps, blit=True, repeat=False, interval=10)
        plt.show()
# Functions for CSF simulation

def generate_circle_points(center, radius, num_points):
    """Generate points forming a circle."""
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = [(cx + radius * np.cos(angle), cy + radius * np.sin(angle)) for angle in angles]
    points.extend(points[:1])
    return points

def curvature(x, y):
    """Calculate the curvature and normal vector at a given point."""
    TA = LA.norm([x[1] - x[0], y[1] - y[0]])
    TB = LA.norm([x[2] - x[1], y[2] - y[1]])
    matrix = np.array([
        [1, TA, np.square(TA)],
        [1, 0, 0],
        [1, -TB, np.square(TB)]
    ])
    a = np.matmul(LA.inv(matrix), x)
    b = np.matmul(LA.inv(matrix), y)
    curvature_value = (2 * a[2] * b[1] - 2 * b[2] * a[1]) / (a[1] ** 2 + b[1] ** 2) ** 1.5
    curvature_value = round(curvature_value, 2)
    curvature_value = max(min(curvature_value, 2), -2)
    normal = (b[1], -a[1]) / np.sqrt(a[1] ** 2 + b[1] ** 2)
    return curvature_value, normal

def move_point_along_normal(point, normal, curvature):
    """Move a point along the normal by a distance proportional to the curvature."""
    point = np.array(point)
    normal = np.array(normal)
    normal_unit = normal / np.linalg.norm(normal)
    new_point = point + curvature * normal_unit
    return tuple(new_point)

def contract_points(coordinates, scale_factor=0.5):
    """Contract points according to their curvature and normal vectors."""
    threshold = 1
    points = remove_close_coordinates(coordinates, threshold)
    n = len(points)
    curvatures = []
    normals = []
    new_points = np.zeros_like(points)

    for i in range(n):
        if i == 0:
            x = [points[n - 2][0], points[0][0], points[1][0]]
            y = [points[n - 2][1], points[0][1], points[1][1]]
        elif i == n - 1:
            x = [points[n - 2][0], points[0][0], points[1][0]]
            y = [points[n - 2][1], points[0][1], points[1][1]]
        elif i == n - 2:
            x = [points[n - 3][0], points[n - 2][0], points[n - 1][0]]
            y = [points[n - 3][1], points[n - 2][1], points[n - 1][1]]
        else:
            x = [points[i - 1][0], points[i][0], points[i + 1][0]]
            y = [points[i - 1][1], points[i][1], points[i + 1][1]]
        k, normal = curvature(x, y)
        curvatures.append(k)
        normals.append(normal)

    max_curvature = np.max(np.abs(curvatures))
    for j in range(n):
        new_points[j] = points[j] + scale_factor * (curvatures[j] / max_curvature * normals[j])
    return new_points

def euclidean_distance(coord1, coord2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def remove_close_coordinates(coordinates, threshold):
    """Remove points that are too close to each other."""
    if len(coordinates) < 3:
        return coordinates
    filtered_coords = [coordinates[0]]
    for coord in coordinates[1:-1]:
        if all(euclidean_distance(coord, existing_coord) >= threshold for existing_coord in filtered_coords):
            filtered_coords.append(coord)
    filtered_coords.append(coordinates[-1])
    return filtered_coords

def chaikin_smooth(points, num_iterations=2):
    """Apply Chaikin's smoothing algorithm to a set of points."""
    def chaikin_subdivide(points):
        new_points = []
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_points.extend([q, r])
        new_points.append(points[-1])
        return new_points

    for _ in range(num_iterations):
        points = chaikin_subdivide(points)

    return points

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()


