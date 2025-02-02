import argparse
import os
import numpy as np
import cv2
import json
from tqdm import tqdm
from numba import njit, prange
from enum import Enum


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_progress(frame_index, progress_file):
    with open(progress_file, 'w') as f:
        json.dump({'last_frame': frame_index}, f)


def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
        return data.get('last_frame', 0)
    return 0


class RenderMode(Enum):
    JULIA_ANIMATION = "julia_animaiton"
    JULIA_STATIC = "julia_static"
    MANDELBROT_STATIC = "mandelbrot_static"
    MANDELBROT_ZOOM = "mandelbrot_zoom"


def compute_mandelbrot_CPU(w, h, max_iter, center_x, center_y, zoom, palette):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in prange(w):
        for y in range(h):
            zx, zy = 0.0, 0.0
            c_x = (x - w / 2) / (0.5 * zoom * w) + center_x
            c_y = (y - h / 2) / (0.5 * zoom * h) + center_y
            i = max_iter
            while zx * zx + zy * zy < 4 and i > 1:
                tmp = zx * zx - zy * zy + c_x
                zy, zx = 2.0 * zx * zy + c_y, tmp
                i -= 1
            index = (i * len(palette)) // max_iter
            img[y, x] = palette[index]
    return img


def compute_julia_CPU(w, h, max_iter, c_x, c_y, zoom, move_x, move_y, palette):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in prange(w):
        for y in range(h):
            zx = 1.5 * (x - w / 2) / (0.5 * zoom * w) + move_x
            zy = 1.0 * (y - h / 2) / (0.5 * zoom * h) + move_y
            i = max_iter
            while zx * zx + zy * zy < 20 and i > 1:
                tmp = zx * zx - zy * zy + c_x
                zy, zx = 2.0 * zx * zy + c_y, tmp
                i -= 1
            index = (i * len(palette)) // max_iter
            img[y, x] = palette[index]
    return img


def generate_colors(n):
    colors = [(0, 0, 0), (0, 0, 255), (255, 255, 255), (255, 200, 0), (128, 0, 0), (0, 0, 0)]
    palette = [tuple(np.interp(i / (n - 1), np.linspace(0, 1, len(colors)), np.array(colors)[:, j])) for i in range(n)
               for j in range(3)]
    return np.array(palette, dtype=np.uint8).reshape(n, 3)


def interpolate_zoom(zoom_start, zoom_end, steps):
    return [zoom_start * (zoom_end / zoom_start) ** (i / steps) for i in range(steps)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default="output.mp4", help='Output video filename')
    parser.add_argument('-i', '--iterations', default=150, type=int, help='Number of iterations')
    parser.add_argument('-m', '--mode', type=str, default=RenderMode.MANDELBROT_ZOOM.value,
                        choices=[e.value for e in RenderMode], help='Render mode')
    args = parser.parse_args()

    width, height, frames, fps = 1920, 1080, 10000, 60
    output_dir = os.path.splitext(args.output)[0]
    progress_file = os.path.join(output_dir, "progress.json")
    ensure_directory_exists(output_dir)

    palette = generate_colors(100000)
    last_frame = load_progress(progress_file)

    if args.mode == RenderMode.MANDELBROT_ZOOM.value:
        center_x, center_y = -0.74364388703, 0.13182590421
        zooms = interpolate_zoom(1.2, 1e15, frames)
        for i in tqdm(range(last_frame, frames)):
            frame_path = os.path.join(output_dir, f"frame_{i:05d}.png")
            if not os.path.exists(frame_path):
                img = compute_mandelbrot_CPU(width, height, args.iterations, center_x, center_y, zooms[i], palette)
                cv2.imwrite(frame_path, img)
            save_progress(i, progress_file)

    elif args.mode == RenderMode.JULIA_ANIMATION.value:
        c_x, c_y = -0.7, 0.27015
        zoom, move_x, move_y = 1.0, 0.0, 0.0
        for i in tqdm(range(last_frame, frames)):
            frame_path = os.path.join(output_dir, f"frame_{i:05d}.png")
            if not os.path.exists(frame_path):
                img = compute_julia_CPU(width, height, args.iterations, c_x, c_y, zoom, move_x, move_y, palette)
                cv2.imwrite(frame_path, img)
            save_progress(i, progress_file)

    frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")])
    frame_sample = cv2.imread(frame_files[0])
    video = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                            (frame_sample.shape[1], frame_sample.shape[0]))
    for frame_file in tqdm(frame_files):
        video.write(cv2.imread(frame_file))
    video.release()
