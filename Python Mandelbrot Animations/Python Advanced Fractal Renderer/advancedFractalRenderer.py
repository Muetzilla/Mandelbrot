import argparse
import numpy as np
import cv2
from tqdm import tqdm
from numba import njit, prange
from enum import Enum
from decimal import Decimal, getcontext
from mpmath import mp


class RenderMode(Enum):
    JULIA_ANIMATION = "julia_animaiton"
    JULIA_STATIC = "julia_static"
    MANDELBROT_STATIC = "mandelbrot_static"
    MANDELBROT_ZOOM = "mandelbrot_zoom"

    def __str__(self):
        return '%s' % self.value


def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier


def generate_colors(n):
    colors = [
        (0, 0, 0),  # Schwarz (Innere Mandelbrot-Menge)
        (0, 0, 128),  # Dunkelblau
        (0, 0, 255),  # Blau
        (255, 255, 255),  # Weiß (starke Kanten)
        (255, 200, 0),  # Gelb
        (128, 0, 0),  # Dunkelrot
        (0, 0, 0)  # Wieder Schwarz für tiefe Bereiche
    ]

    palette = []
    num_colors = len(colors) - 1

    for i in range(n):
        t = i / (n - 1)  # Normalisierte Position zwischen 0 und 1
        index = int(t * num_colors)  # Finde die zwei angrenzenden Farben
        next_index = min(index + 1, num_colors)

        c1 = np.array(colors[index])
        c2 = np.array(colors[next_index])

        mix = c1 + (c2 - c1) * (t * num_colors - index)  # Interpolation zwischen den Farben
        palette.append(tuple(mix.astype(int)))

    return palette


def generate_colors_dynamic(n, zoom_level):
    """Generiert eine dynamische Farbpalette mit mehr Kontrast und dunklerem Hintergrund."""

    if isinstance(zoom_level, (list, np.ndarray)):
        zoom_level = zoom_level[0]

    hue_shift = int(10 * np.cos(float(zoom_level) * 0.3))  # Reduzierte Helligkeit

    colors = [
        (0, 0, 0),  # Schwarz (Innere Mandelbrot-Menge)
        (0, 0, 128),  # Dunkelblau
        (0, 0, 255),  # Blau
        (255, 255, 255),  # Weiß (starke Kanten)
        (255, 200, 0),  # Gelb
        (128, 0, 0),  # Dunkelrot
        (0, 0, 0)  # Wieder Schwarz für tiefe Bereiche
    ]

    palette = []
    num_colors = len(colors) - 1

    for i in range(n):
        t = i / (n - 1)
        index = int(t * num_colors)
        next_index = min(index + 1, num_colors)

        c1 = np.array(colors[index])
        c2 = np.array(colors[next_index])

        mix = c1 + (c2 - c1) * ((t * num_colors - index) ** 1.5)  # Quadratische Interpolation
        mix = np.clip(mix + hue_shift, 0, 255)  # Keine übermäßige Helligkeit

        palette.append(tuple(mix.astype(int)))

    return palette


# Gamma-Korrektur für besseren Kontrast
def adjust_gamma(image, gamma=0.3):  # <1 = dunkler
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def interpolate(c1, c2, steps):
    delta1 = (c2[0] - c1[0]) / steps
    delta2 = (c2[1] - c1[1]) / steps
    res = []
    cc1, cc2 = c1

    for i in range(steps):
        res.append((cc1, cc2))
        cc1 += delta1
        cc2 += delta2

    return res


@njit(parallel=True, fastmath=True)
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


def process_julia(max_iter, c, palette, width, height):
    w, h, zoom = width, height, 0.7
    move_x, move_y = 0.0, 0.0
    c_x, c_y = c
    img = compute_julia_CPU(w, h, max_iter, c_x, c_y, zoom, move_x, move_y, np.array(palette, dtype=np.uint8))

    return img


@njit(parallel=True, fastmath=True)
def compute_mandelbrot_CPU(w, h, max_iter, center_x, center_y, zoom, palette_np):
    img = np.zeros((h, w, 3), dtype=np.uint8)

    for x in prange(w):
        for y in range(h):
            zx, zy = 0.0, 0.0
            zoom_factor = min(w, h)  # Nutzt die kleinere Dimension für ein einheitliches Zoom
            c_x = (x - w / 2) / (0.5 * zoom * zoom_factor) + center_x
            c_y = (y - h / 2) / (0.5 * zoom * zoom_factor) + center_y

            i = max_iter
            while zx * zx + zy * zy < 4 and i > 1:
                tmp = zx * zx - zy * zy + c_x
                zy, zx = 2.0 * zx * zy + c_y, tmp
                i -= 1

            index = (i * len(palette)) // max_iter
            img[y, x] = palette[index]

    return img


def interpolate_zoom(zoom_start, zoom_end, steps):
    zooms = []
    zoom_factor = (zoom_end / zoom_start) ** (1 / steps)

    zoom = zoom_start * (height / width)
    for _ in range(steps):
        zooms.append(zoom)
        zoom *= zoom_factor  # Sanftes Zoomen

    return zooms


if __name__ == '__main__':
    width = 960
    height = 540
    mandelbrot_size = 2160  # Berechnung in quadratischem Format
    getcontext().prec = 200
    mp.dps = 200  # Setze die Anzahl der Dezimalstellen auf 100
    # Seepferdchen-Tal    center_x, center_y = -0.74364388703, 0.13182590421
    center_y = mp.mpf(
        "-0.0000000000000016571246929541869232581096198127918902650429012737576040533449811085095604736830870705073596032339738954703823119487248269034036992175051414692240092855401199612311290200085666684708878815843399535840677925940422190475")
    center_x = mp.mpf(
        "-1.74999841099374081749002483162428393452822172335808534616943930976364725846655540417646727085571962736578151132907961927190726789896685696750162524460775546580822744596887978637416593715319388030232414667046419863755743802804780843375")

    fps = 60
    iteration_manipulator = 8
    frames = 15_0  # Dauer der Animation

    render_mode = RenderMode.MANDELBROT_ZOOM

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        default='interpolate_' + render_mode.value + '_' + str(
                            width) + '_' + str(
                            height) + '_iter-' + str(iteration_manipulator) + "_" + str(fps) + '_fps_length' + str(
                            frames) + 'cord_x-' + str(truncate_float(center_x, 20)) + '_cord_y-' + str(
                            truncate_float(center_y, 20)) + '.mp4', type=str,
                        help='Resulting video')
    parser.add_argument('-i', '--iterations', default=150, type=int, help='Number of iterations')
    args = parser.parse_args()
    print(args)

    if render_mode == RenderMode.JULIA_ANIMATION:
        palette = np.array(generate_colors(100_000), dtype=np.uint8)
        interps = []
        num_iter = args.iterations * iteration_manipulator

        interps.append(interpolate((-0.16, 1.0405), (-0.722, 0.246), num_iter))
        interps.append(interpolate((-0.722, 0.246), (-0.235125, 0.827215), num_iter))
        interps.append(interpolate((-0.235125, 0.827215), (-1.25066, 0.02012), num_iter))
        # interps.append(interpolate((-1.25066, 0.02012), (-0.748, 0.1), num_iter))
        # interps.append(interpolate((-0.748, 0.1), (0.9927, -0.181), num_iter))

        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), fps, (width, height))
        pbar = tqdm(total=(num_iter * len(interps)))

        for interp in interps:
            for p in interp:
                r = process_julia(num_iter, p, palette, width, height)
                out.write(r)
                pbar.update(1)

        out.release()

    elif render_mode == RenderMode.MANDELBROT_ZOOM:

        render_width, render_height = mandelbrot_size, mandelbrot_size

        zooms = interpolate_zoom(1.2, 1_000_000_000_000_000_000_000_000_000_000_000.0, frames)
        palette = np.array(generate_colors_dynamic(100_000, zooms), dtype=np.uint8)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), fps, (width, height))
        pbar = tqdm(total=len(zooms))

        # center_x, center_y = -0.74364388703, 0.13182590421  # Seepferdchen-Tal
        num_iter = args.iterations * iteration_manipulator

        for zoom in zooms:
            img = compute_mandelbrot_CPU(width, height, num_iter,np.longdouble(center_x), np.longdouble(center_y),
                                         zoom, palette)

            # img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            out.write(img)
            pbar.update(1)

        out.release()
