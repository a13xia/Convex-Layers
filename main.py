import matplotlib.pyplot as plt     # library used for plotting the convex hulls and the convex layers
import numpy as np  # for the random points
from tkinter import Tk, Frame, ttk, Label  #gui
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg     # gui
import time     #to measure the time that it takes for each algorithm to run
import psutil # to determine how much memory is used by each alg
import multiprocessing
import tracemalloc






#                                           QUICKHULL


def quickhull(points):

    def find_hull(points, A, B):

        if not points:
            return []
        #when no points remain to be checked and added to the convex hull


        #find the farthest point from the segment [point A, point B]
        farthest_point = max(points, key=lambda point: distance(A, B, point))

        points.remove(farthest_point) # remove the farthest point from the set

        # here, we check the points that are on the left of the lines [A, farthestP] and [farthestP, B]
        leftLine1 = [p for p in points if is_left_of_line(A, farthest_point, p)]
        leftLine2 = [p for p in points if is_left_of_line(farthest_point, B, p)]


        # computes the hull on each side
        return (
                find_hull(leftLine1, A, farthest_point)
                + [farthest_point]
                + find_hull(leftLine2, farthest_point, B)
        )



    def distance(pointA, pointB, point):
        return abs((pointB[0] - pointA[0]) * (pointA[1] - point[1]) - (pointA[0] - point[0]) * (pointB[1] - pointA[1]))

    #this function determines if a point  is or it's not on the left of [A,B]
    def is_left_of_line(pointA, pointB, point):

        return (pointB[0] - pointA[0]) * (point[1] - pointA[1]) - (point[0] - pointA[0]) * (pointB[1] - pointA[1]) > 0

    leftmost = min(points, key=lambda p: p[0])
    rightmost = max(points, key=lambda p: p[0])


    # here we divide the points into 2 groups: those that are above the line [leftmost, rightmost],
    # and those that are below
    upper = [p for p in points if is_left_of_line(leftmost, rightmost, p)]
    lower = [p for p in points if is_left_of_line(rightmost, leftmost, p)]
    upperHull = find_hull(upper, leftmost, rightmost)
    lowerHull = find_hull(lower, rightmost, leftmost)



    return [leftmost] + upperHull + [rightmost] + lowerHull






#                                               GRAHAM'S SCAN


def graham_scan(points):

    # sort the points by the x-coordinate, then by the y coordinate
    points = sorted(points, key=lambda p: (p[0], p[1]))


# this function returns the orientation of the triplet (p,q,r). it returns 1 if it's clockwise, -1 if it's counterclockwise and 0 if it's collinear
    def orientation(p, q, r):
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    lowerHull = []
    for p in points:
        while len(lowerHull) >= 2 and orientation(lowerHull[-2], lowerHull[-1], p) <= 0:
            lowerHull.pop() #here, we remove the last point  point if it makes a right turn/a non-left turn
        lowerHull.append(p)



    upperHull = []
    for p in reversed(points):
        while len(upperHull) >= 2 and orientation(upperHull[-2], upperHull[-1], p) <= 0:
            upperHull.pop()
        upperHull.append(p)

# combining the hulls, excluding the upper duplicate end points
    return lowerHull[:-1] + upperHull[:-1]







                                        # JARVIS MARCH

def jarvis_march(points):
    Hull = []
    # we start with the leftmost point
    startPoint = min(points, key=lambda p: (p[0], p[1]))


    Point = startPoint
    while True:
        Hull.append(Point) # adding the current point to the hull
        next_point = points[0] # initializing the next point
        for p in points:
            if p == Point: # we will ignore the startpoint
                continue
                # here, we det the orientation
            cross_product = ((next_point[0] - Point[0]) * (p[1] - Point[1]) -
                             (p[0] - Point[0]) * (next_point[1] - Point[1]))
            # update the next point if a better one exists
            if cross_product < 0 or (cross_product == 0 and
                                     np.linalg.norm(np.array(Point) - np.array(p)) >
                                     np.linalg.norm(np.array(Point) - np.array(next_point))):
                next_point = p
        Point = next_point  # point is a placeholder, like an i. now it will go to the next point
        if Point == startPoint: # in this case, the hull is closed.
            break
    return Hull




def compute_convex_layers(points, algorithm):
    all_Layers = []

    #a copy of points
    remainingPoints = points.copy()


    while remainingPoints:
        hull = algorithm(remainingPoints)
        all_Layers.append(hull)
        remainingPoints = [p for p in remainingPoints if p not in hull]

    return all_Layers




def compute_layers_with_memory(points, algorithm):
    """Run the convex hull algorithm and measure memory usage."""
    tracemalloc.start()

    # Snapshot before the computation
    start_snapshot = tracemalloc.take_snapshot()

    # Run the algorithm
    layers = compute_convex_layers(points, algorithm)

    # Snapshot after the computation
    end_snapshot = tracemalloc.take_snapshot()

    tracemalloc.stop()

    # Calculate memory usage
    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    peak_memory = tracemalloc.get_traced_memory()[1]  # Peak memory in bytes
    return layers, stats, peak_memory

def run_algorithm_in_process(points, algorithm):
    """Function to be executed in a separate process."""
    layers, stats, peak_memory = compute_layers_with_memory(points, algorithm)
    print("\n[ Top Memory Usage Stats ]")
    for stat in stats[:10]:
        print(stat)
    return len(layers), peak_memory  # Return the result and peak memory usage


class ConvexHullApp:

    """The Tkinter-based app to visualize convex hull algorithms and the layers they produce."""



    def __init__(self, root):
        self.root = root
        self.root.title("Convex Hull Algorithms with Layers")
        # generating the random points
        # based on different input
        np.random.seed(0)
        self.points = np.random.rand(100, 2) * 100  # *100 means there is a 100x100 grid
        self.points = [tuple(p) for p in self.points]

        # regenerate duplicate points
        grid_size = 100
        unique_points = self.generate_unique_points(self.points, grid_size)

        # a tab for each alg
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both")

        # coloring the layers for each algorithm in a different colour, for better distinction.

        algorithms = [("QuickHull", quickhull, "red"),
                     ("Graham's Scan", graham_scan, "green"),
                     ("Jarvis March", jarvis_march, "blue")]

        # algorithms = [("Jarvis March", jarvis_march, "blue")]
# adiing a tab for each algorithm
        for name, algorithm, color in algorithms:
            self.addTab(name, algorithm, color)

    @staticmethod
    def generate_unique_points(points, grid_size):
        unique_points = set()
        regenerated_points = []

        for x, y in points:
            while (x, y) in unique_points:
                # regenerate the point if it already exists
                x, y = np.random.rand(2) * grid_size
            unique_points.add((x, y))
            regenerated_points.append((x, y))

        return regenerated_points


    def addTab(self, name, algorithm, color):
        frame = Frame(self.notebook)
        self.notebook.add(frame, text=name)
        self.displayAlgorithm(frame, name, algorithm, color)

    def get_memory_usage(self):
        process = psutil.Process()
        return process.memory_info().rss

    def displayAlgorithm(self, frame, name, algorithm, color):


        # storing the starting time
        starting_time = time.time()

        # the starting usage
        start_memory = self.get_memory_usage()

        layers = compute_convex_layers(self.points, algorithm)

        end_memory = self.get_memory_usage()
        mem_used = end_memory - start_memory # memory, in bytes

        # storing the end time
        end_time = time.time()
        runtime = end_time - starting_time

        # convert memory from bytes to MB
        memory_used_mb = mem_used / (1024 * 1024)

        memory_label = Label(
            frame,
            text=f"Memory Used: {memory_used_mb:.2f} MB",
            font=("Arial", 16),
            fg="blue"
        )
        memory_label.pack(pady=10)

        fig, ax = plt.subplots(figsize=(6, 4))
        points = np.array(self.points)
        ax.scatter(points[:, 0], points[:, 1], label="Points", alpha=0.5)

        for i, layer in enumerate(layers):
            layer_points = np.array(layer + [layer[0]])  # Close the hull- adding the start point
            ax.plot(layer_points[:, 0], layer_points[:, 1], label=f"Layer {i + 1}", color=color, alpha=0.7)
            ax.scatter(layer_points[:, 0], layer_points[:, 1], color=color, alpha=0.7)


        ax.legend()
        ax.set_title(f"{name}: Convex Layers (Runtime: {runtime:.4f} seconds)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()




if __name__ == "__main__":
    root = Tk()
    app = ConvexHullApp(root)
    root.mainloop()
