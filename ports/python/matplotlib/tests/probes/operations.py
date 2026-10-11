"""Render an Agg plot and exercise native triangulation without a display."""

import io

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from PIL import Image

figure = Figure(figsize=(2, 2), dpi=40)
canvas = FigureCanvasAgg(figure)
axes = figure.subplots()
axes.plot([0, 1, 2], [0, 1, 0])
axes.set_title("Port smoke")
buffer = io.BytesIO()
canvas.print_png(buffer)
image = Image.open(io.BytesIO(buffer.getvalue()))
assert image.size == (80, 80)
assert buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
tri = Triangulation([0, 1, 1, 0], [0, 0, 1, 1])
assert tri.triangles.shape == (2, 3)
print("matplotlib: Agg PNG and Qhull triangulation passed")
