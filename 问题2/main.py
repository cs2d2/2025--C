# -*- coding: utf-8 -*-
"""
问题2：正弦状裂隙拟合第一版

主流程：
    读取附件2图像
    -> 中值滤波 + Otsu 阈值 + 形态学后处理，得到裂隙掩膜
    -> 骨架化提取中心线
    -> 像素坐标转换为毫米坐标
    -> RANSAC + 非线性最小二乘拟合多条正弦曲线
    -> 输出参数表与拟合叠加图
"""

from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.morphology import skeletonize

CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
ATTACH2_DIR = BASE_DIR / "2025年研究生数学建模竞赛赛题" / "C题" / "C题" / "附件2"
OUTPUT_DIR = CODE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

BOREHOLE_PERIMETER = 94.25
IMAGE_DEPTH = 500.0
IMAGE_WIDTH = 244
IMAGE_HEIGHT = 1350
MIN_POINTS_FIT = 60
INLIER_TOLERANCE = 5.0
RANSAC_ITERATIONS = 100
MAX_FRACTURES = 5
SMALL_COMPONENT_AREA = 40


def read_gray(image_path):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return image


def write_image(image_path, image):
    suffix = image_path.suffix or ".png"
    success, buffer = cv2.imencode(suffix, image)
    if not success:
        raise IOError(f"无法写入图像: {image_path}")
    buffer.tofile(str(image_path))


def preprocess_mask(image):
    """中值滤波 + Otsu 阈值 + 形态学闭运算 + 删除小连通域。"""
    blurred = cv2.medianBlur(image, 3)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= SMALL_COMPONENT_AREA:
            mask[labels == i] = 1
    return mask


def extract_centerline_points(mask):
    """提取骨架点，并转换为毫米坐标。"""
    skeleton = skeletonize(mask.astype(bool))
    ys, xs = np.nonzero(skeleton)
    x_mm = xs.astype(np.float64) * BOREHOLE_PERIMETER / IMAGE_WIDTH
    y_mm = (IMAGE_HEIGHT - ys.astype(np.float64)) * IMAGE_DEPTH / IMAGE_HEIGHT
    return x_mm, y_mm


def sine_func(x, R, P, beta, C):
    return R * np.sin(2.0 * np.pi * x / P + beta) + C


def linear_initial_guess(x, y):
    """先固定周期为钻孔周长，用线性最小二乘得到 R、beta、C 的初值。"""
    k = 2.0 * np.pi / BOREHOLE_PERIMETER
    design = np.column_stack([np.sin(k * x), np.cos(k * x), np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    a, b, c = coef
    R0 = float(np.hypot(a, b))
    beta0 = float(np.arctan2(b, a))
    C0 = float(c)
    return [R0, BOREHOLE_PERIMETER, beta0, C0]


def fit_single_sine(x, y, p0=None):
    """用非线性最小二乘拟合一条正弦曲线。"""
    if p0 is None:
        p0 = linear_initial_guess(x, y)
    bounds = (
        [0.0, 30.0, -2.0 * np.pi, 0.0],
        [100.0, 150.0, 2.0 * np.pi, IMAGE_DEPTH],
    )
    popt, _ = curve_fit(sine_func, x, y, p0=p0, bounds=bounds, maxfev=10000)
    return popt


def ransac_sine_fit(x, y):
    """顺序 RANSAC：每次拟合一条曲线并移除其内点，直到点数不足。"""
    rng = np.random.default_rng(42)
    remaining = np.arange(len(x))
    results = []

    for _ in range(MAX_FRACTURES):
        if len(remaining) < MIN_POINTS_FIT:
            break

        best = None
        sample_size = max(MIN_POINTS_FIT, 8)
        for _ in range(RANSAC_ITERATIONS):
            if len(remaining) < sample_size:
                sample_size = len(remaining)
            sample_idx = rng.choice(remaining, size=sample_size, replace=False)

            try:
                p0 = linear_initial_guess(x[sample_idx], y[sample_idx])
                popt = fit_single_sine(x[sample_idx], y[sample_idx], p0=p0)
            except Exception:
                continue

            residual = np.abs(y[remaining] - sine_func(x[remaining], *popt))
            inlier_idx = remaining[residual < INLIER_TOLERANCE]
            if best is None or len(inlier_idx) > len(best["idx"]):
                best = {"popt": popt, "idx": inlier_idx}

        if best is None or len(best["idx"]) < MIN_POINTS_FIT:
            break

        results.append(best["popt"])
        remaining = np.setdiff1d(remaining, best["idx"], assume_unique=True)

    return results


def save_fit_plot(image, x_mm, y_mm, results, output_path):
    """左侧原图、右侧笛卡尔坐标拟合图，并排对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].scatter(x_mm, y_mm, s=4, c="gray", alpha=0.4, label="Extracted points")
    x_plot = np.linspace(0.0, BOREHOLE_PERIMETER, 500)
    for idx, popt in enumerate(results, 1):
        y_plot = sine_func(x_plot, *popt)
        axes[1].plot(x_plot, y_plot, linewidth=1.5, label=f"Curve {idx}")

    axes[1].set_xlim(0.0, BOREHOLE_PERIMETER)
    axes[1].set_ylim(0.0, IMAGE_DEPTH)
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    axes[1].set_title("Fitted sine curves in Cartesian coordinates")
    axes[1].legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    image_paths = sorted(ATTACH2_DIR.glob("*.jpg"), key=lambda p: int(p.stem.split("-")[1]))
    if not image_paths:
        raise FileNotFoundError(f"未找到附件2图像: {ATTACH2_DIR}")

    print(f"发现 {len(image_paths)} 张附件2图像")
    all_rows = []

    for path in image_paths:
        image = read_gray(path)
        mask = preprocess_mask(image)
        x_mm, y_mm = extract_centerline_points(mask)

        if len(x_mm) < MIN_POINTS_FIT:
            print(f"{path.name}: 有效中心线点过少，跳过")
            continue

        results = ransac_sine_fit(x_mm, y_mm)
        print(f"{path.name}: 拟合到 {len(results)} 条正弦曲线")

        for i, popt in enumerate(results, 1):
            R, P, beta, C = popt
            all_rows.append([path.name, i, float(R), float(P), float(beta), float(C)])

        save_fit_plot(image, x_mm, y_mm, results, OUTPUT_DIR / f"{path.stem}_fit.png")

    result_csv = OUTPUT_DIR / "fit_results.csv"
    with open(result_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["图像编号", "裂隙编号", "R(mm)", "P(mm)", "beta(rad)", "C(mm)"])
        writer.writerows(all_rows)

    print(f"拟合结果已保存到: {result_csv}")
    print(f"叠加图已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
