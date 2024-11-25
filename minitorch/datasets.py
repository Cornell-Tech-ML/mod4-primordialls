import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate list of N random points on the cartesian plane"""
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a graph with vertices and labels based on a simple rule.

    This function creates a set of points and assigns binary labels to them based on their x-coordinate.
    Specifically, if the x-coordinate of a point is less than 0.5, it is assigned a label of 1; otherwise, it is assigned 0.

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: An instance of the `Graph` class containing the generated points and their corresponding labels.
        The `Graph` object is initialized with:
        - `N`: The number of points.
        - `X`: A list of tuples representing the generated points.
        - `y`: A list of binary labels corresponding to each point.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a graph with vertices and labels based on a diagonal rule.

    This function creates a set of points and assigns binary labels to them based on the sum of their coordinates.
    Specifically, if the sum of the x and y coordinates of a point is less than 0.5, it is assigned a label of 1; otherwise, it is assigned 0.

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: An instance of the `Graph` class containing the generated points and their corresponding labels.
        The `Graph` object is initialized with:
        - `N`: The number of points.
        - `X`: A list of tuples representing the generated points.
        - `y`: A list of binary labels corresponding to each point.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a graph with vertices and labels based on a splitting rule.

    This function creates a set of points and assigns binary labels to them based on the x-coordinate.
    Specifically, if the x-coordinate of a point is less than 0.2 or greater than 0.8, it is assigned a label of 1; otherwise, it is assigned 0.

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: An instance of the `Graph` class containing the generated points and their corresponding labels.
        The `Graph` object is initialized with:
        - `N`: The number of points.
        - `X`: A list of tuples representing the generated points.
        - `y`: A list of binary labels corresponding to each point.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a graph with vertices and labels based on an XOR rule.

    This function creates a set of points and assigns binary labels to them based on their coordinates.
    Specifically, if a point's coordinates satisfy one of the following conditions:
    - x-coordinate is less than 0.5 and y-coordinate is greater than 0.5, or
    - x-coordinate is greater than 0.5 and y-coordinate is less than 0.5,
    it is assigned a label of 1; otherwise, it is assigned 0.

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: An instance of the `Graph` class containing the generated points and their corresponding labels.
        The `Graph` object is initialized with:
        - `N`: The number of points.
        - `X`: A list of tuples representing the generated points.
        - `y`: A list of binary labels corresponding to each point.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a graph with vertices and labels based on a circular rule.

    This function creates a set of points and assigns binary labels to them based on their distance from the origin
    of a unit circle centered at (0.5, 0.5). Specifically, if the squared distance of a point from the center is greater
    than 0.1, it is assigned a label of 1; otherwise, it is assigned 0.

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: An instance of the `Graph` class containing the generated points and their corresponding labels.
        The `Graph` object is initialized with:
        - `N`: The number of points.
        - `X`: A list of tuples representing the generated points.
        - `y`: A list of binary labels corresponding to each point based on their distance from the center.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a graph with vertices and labels based on a spiral pattern.

    This function creates a set of points that follow a spiral trajectory and assigns binary labels to them.
    The points are divided into two segments:
    - The first segment follows a spiral outward from the origin in one direction.
    - The second segment follows a spiral outward from the origin in the opposite direction.
    Points in the first segment are assigned a label of 0, and points in the second segment are assigned a label of 1.

    Args:
    ----
    N (int): The number of points to generate. The function will generate `N` points, with half following the spiral in one direction and half in the opposite direction.

    Returns:
    -------
    Graph: An instance of the `Graph` class containing the generated points and their corresponding labels.
        The `Graph` object is initialized with:
        - `N`: The total number of points.
        - `X`: A list of tuples representing the generated points, with a spiral pattern.
        - `y`: A list of binary labels where the first half of the points are labeled 0 and the second half are labeled 1.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
