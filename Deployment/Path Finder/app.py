from flask import Flask, request, jsonify
import numpy as np
import cv2
import heapq
from skimage.morphology import skeletonize
from PIL import Image
import io
import time

app = Flask(__name__)

# Load and process the image into a skeleton
def load_and_process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    rgb_img = np.array(image)
    gray_img = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, thr_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    skeleton = skeletonize(thr_img > 0)
    return ~skeleton

# BFS Pathfinding Algorithm
def find_path_bfs(mapT, start_point, end_point, box_radius=30):
    _mapt = np.copy(mapT)
    x1, y1 = end_point
    x0, y0 = start_point

    start_y, start_x = np.where(_mapt[y0-box_radius:y0+box_radius, 
                                    x0-box_radius:x0+box_radius] == 0)
    start_y += y0-box_radius
    start_x += x0-box_radius
    start_idx = np.argmin(np.sqrt((start_y - y0)**2 + (start_x - x0)**2))
    start_y, start_x = start_y[start_idx], start_x[start_idx]

    end_y, end_x = np.where(_mapt[y1-box_radius:y1+box_radius, 
                                x1-box_radius:x1+box_radius] == 0)
    end_y += y1-box_radius
    end_x += x1-box_radius
    end_idx = np.argmin(np.sqrt((end_y - y1)**2 + (end_x - x1)**2))
    end_y, end_x = end_y[end_idx], end_x[end_idx]

    pts_x, pts_y, pts_c = [start_x], [start_y], [0]
    xmesh, ymesh = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    ymesh, xmesh = ymesh.ravel(), xmesh.ravel()
    dst = np.zeros(mapT.shape)

    while True:
        if not pts_x:
            return [], []

        idc = np.argmin(pts_c)
        ct = pts_c.pop(idc)
        x = pts_x.pop(idc)
        y = pts_y.pop(idc)

        ys, xs = np.where(_mapt[y-1:y+2, x-1:x+2] == 0)
        _mapt[ys+y-1, xs+x-1] = ct
        _mapt[y, x] = 9999999
        dst[ys+y-1, xs+x-1] = ct + 1

        for new_x, new_y in zip(xs + x - 1, ys + y - 1):
            if 0 <= new_x < 1000 and 0 <= new_y < 1000:
                pts_x.append(new_x)
                pts_y.append(new_y)
                pts_c.append(ct + 1)

        if np.sqrt((x - end_x)**2 + (y - end_y)**2) < 2:
            break

    path_x, path_y = [end_x], [end_y]
    y, x = end_y, end_x

    while True:
        nbh = dst[y-1:y+2, x-1:x+2]
        nbh[1, 1] = 9999999
        nbh[nbh == 0] = 9999999

        if np.min(nbh) == 9999999:
            break

        idx = np.argmin(nbh)
        y += ymesh[idx]
        x += xmesh[idx]

        if x < 0 or x > 1000 or y < 0 or y > 1000:
            continue

        path_y.append(y)
        path_x.append(x)

        if np.sqrt((x - start_x)**2 + (y - start_y)**2) < 2:
            break

    return path_x, path_y, (start_x, start_y), (end_x, end_y)

# Dijkstra's Pathfinding Algorithm
def find_path_dijkstra(mapT, start_point, end_point, box_radius=30):
    _mapt = np.copy(mapT)
    x1, y1 = end_point
    x0, y0 = start_point

    start_y, start_x = np.where(_mapt[y0-box_radius:y0+box_radius, 
                                    x0-box_radius:x0+box_radius] == 0)
    start_y += y0-box_radius
    start_x += x0-box_radius
    start_idx = np.argmin(np.sqrt((start_y - y0)**2 + (start_x - x0)**2))
    start_y, start_x = start_y[start_idx], start_x[start_idx]

    end_y, end_x = np.where(_mapt[y1-box_radius:y1+box_radius, 
                                x1-box_radius:x1+box_radius] == 0)
    end_y += y1-box_radius
    end_x += x1-box_radius
    end_idx = np.argmin(np.sqrt((end_y - y1)**2 + (end_x - x1)**2))
    end_y, end_x = end_y[end_idx], end_x[end_idx]

    pq = [(0, start_x, start_y)]
    dst = np.full(mapT.shape, np.inf)
    dst[start_y, start_x] = 0
    xmesh, ymesh = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    ymesh, xmesh = ymesh.ravel(), xmesh.ravel()

    while pq:
        ct, x, y = heapq.heappop(pq)

        if np.sqrt((x - end_x)**2 + (y - end_y)**2) < 2:
            break

        for dx, dy in zip(xmesh, ymesh):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 1000 and 0 <= new_y < 1000 and _mapt[new_y, new_x] == 0:
                new_cost = ct + 1
                if new_cost < dst[new_y, new_x]:
                    dst[new_y, new_x] = new_cost
                    heapq.heappush(pq, (new_cost, new_x, new_y))

    path_x, path_y = [end_x], [end_y]
    y, x = end_y, end_x

    while True:
        nbh = dst[y-1:y+2, x-1:x+2]
        nbh[1, 1] = np.inf
        nbh[nbh == 0] = np.inf

        if np.min(nbh) == np.inf:
            break

        idx = np.argmin(nbh)
        y += ymesh[idx]
        x += xmesh[idx]

        if x < 0 or x > 1000 or y < 0 or y > 1000:
            continue

        path_y.append(y)
        path_x.append(x)

        if np.sqrt((x - start_x)**2 + (y - start_y)**2) < 2:
            break

    return path_x, path_y, (start_x, start_y), (end_x, end_y)

# A* Pathfinding Algorithm
def find_path_astar(mapT, start_point, end_point, box_radius=30):
    _mapt = np.copy(mapT)
    x1, y1 = end_point
    x0, y0 = start_point

    start_y, start_x = np.where(_mapt[y0-box_radius:y0+box_radius, 
                                    x0-box_radius:x0+box_radius] == 0)
    start_y += y0-box_radius
    start_x += x0-box_radius
    start_idx = np.argmin(np.sqrt((start_y - y0)**2 + (start_x - x0)**2))
    start_y, start_x = start_y[start_idx], start_x[start_idx]

    end_y, end_x = np.where(_mapt[y1-box_radius:y1+box_radius, 
                                x1-box_radius:x1+box_radius] == 0)
    end_y += y1-box_radius
    end_x += x1-box_radius
    end_idx = np.argmin(np.sqrt((end_y - y1)**2 + (end_x - x1)**2))
    end_y, end_x = end_y[end_idx], end_x[end_idx]

    pq = [(0, start_x, start_y)]
    dst = np.full(mapT.shape, np.inf)
    dst[start_y, start_x] = 0
    xmesh, ymesh = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    ymesh, xmesh = ymesh.ravel(), xmesh.ravel()

    while pq:
        ct, x, y = heapq.heappop(pq)

        if np.sqrt((x - end_x)**2 + (y - end_y)**2) < 2:
            break

        for dx, dy in zip(xmesh, ymesh):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 1000 and 0 <= new_y < 1000 and _mapt[new_y, new_x] == 0:
                new_cost = ct + 1 + np.sqrt((new_x - end_x)**2 + (new_y - end_y)**2)
                if new_cost < dst[new_y, new_x]:
                    dst[new_y, new_x] = new_cost
                    heapq.heappush(pq, (new_cost, new_x, new_y))

    path_x, path_y = [end_x], [end_y]
    y, x = end_y, end_x

    while True:
        nbh = dst[y-1:y+2, x-1:x+2]
        nbh[1, 1] = np.inf
        nbh[nbh == 0] = np.inf

        if np.min(nbh) == np.inf:
            break

        idx = np.argmin(nbh)
        y += ymesh[idx]
        x += xmesh[idx]

        if x < 0 or x > 1000 or y < 0 or y > 1000:
            continue

        path_y.append(y)
        path_x.append(x)

        if np.sqrt((x - start_x)**2 + (y - start_y)**2) < 2:
            break

    return path_x, path_y, (start_x, start_y), (end_x, end_y)

# Define algorithms
algorithms = {
    "BFS": find_path_bfs,
    "Dijkstra": find_path_dijkstra,
    "A*": find_path_astar
}

@app.route('/solve-maze', methods=['POST'])
def solve_maze():
    try:
        # Get the algorithm and maze image
        algorithm = request.form.get('algorithm')
        maze_image = request.files['maze_image'].read()

        # Load and process the image
        mapT = load_and_process_image(maze_image)

        # Start and end points
        start_point = (855, 1000)
        end_point = (810, 1000)

        if algorithm not in algorithms:
            return jsonify({'error': f"Unsupported algorithm: {algorithm}"}), 400

        start_time = time.time()
        path_x, path_y, start, end = algorithms[algorithm](mapT, start_point, end_point)
        end_time = time.time()

        return jsonify({
            'path_x': list(map(int, path_x)),
            'path_y': list(map(int, path_y)),
            'start': tuple(map(int, start)),
            'end': tuple(map(int, end)),
            'computation_time': end_time - start_time
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
