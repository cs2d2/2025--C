# -*- coding: utf-8 -*-
"""
问题4 三维空间生成演示：使用问题2拟合出的正弦参数 R、P、beta、C。
此处忽略问题3的 JRC 参数。

说明：
  本脚本读取 问题2/output/fit_results.csv 中的参数，并为了演示把
  “图2-1.jpg ~ 图2-6.jpg”分别对应到附件4的 1# ~ 6# 钻孔，
  然后根据正弦参数重建三维裂隙圆盘。该映射仅用于展示三维空间生成效果，
  并非附件4中裂隙的真实归属。
"""
from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
ATTACH4_DIR = BASE_DIR / "2025年研究生数学建模竞赛赛题" / "C题" / "C题" / "附件4"
PROBLEM2_CSV = BASE_DIR / "问题2" / "output" / "fit_results.csv"
OUTPUT_DIR = CODE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

BOREHOLE_RADIUS = 15.0
DISK_RADIUS = 400.0

BOREHOLES = {
    "1#": ((500.0, 2000.0, 0.0), (500.0, 2000.0, 7000.0)),
    "2#": ((1500.0, 2000.0, 0.0), (1500.0, 2000.0, 7000.0)),
    "3#": ((2500.0, 2000.0, 0.0), (2500.0, 2000.0, 7000.0)),
    "4#": ((500.0, 1000.0, 0.0), (500.0, 1000.0, 5000.0)),
    "5#": ((1500.0, 1000.0, 0.0), (1500.0, 1000.0, 7000.0)),
    "6#": ((2500.0, 1000.0, 0.0), (2500.0, 1000.0, 7000.0)),
}

COLORS = ["red", "green", "blue", "orange", "purple", "brown"]


def load_problem2_params():
    rows_by_image = {}
    with open(PROBLEM2_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            image = row[0]
            R = float(row[2])
            P = float(row[3])
            beta = float(row[4])
            C = float(row[5])
            rows_by_image.setdefault(image, []).append((R, P, beta, C))
    return rows_by_image


def draw_fracture_disk(ax, mouth, R, P, beta, C, color, disk_radius=DISK_RADIUS, resolution=36):
    x0, y0 = mouth[0], mouth[1]
    r = BOREHOLE_RADIUS
    a = R * np.sin(beta) / r
    b = R * np.cos(beta) / r

    normal = np.array([-a, -b, 1.0])
    normal = normal / np.linalg.norm(normal)

    if abs(normal[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])

    u1 = np.cross(normal, helper)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(normal, u1)

    center = np.array([x0, y0, C])
    phi = np.linspace(0.0, 2.0 * np.pi, resolution)
    radius = np.linspace(0.0, disk_radius, 2)
    PHI, RR = np.meshgrid(phi, radius)

    X = center[0] + RR * (np.cos(PHI) * u1[0] + np.sin(PHI) * u2[0])
    Y = center[1] + RR * (np.cos(PHI) * u1[1] + np.sin(PHI) * u2[1])
    Z = center[2] + RR * (np.cos(PHI) * u1[2] + np.sin(PHI) * u2[2])

    ax.plot_surface(X, Y, Z, color=color, alpha=0.25, edgecolor=color, linewidth=0.2)


def main():
    rows_by_image = load_problem2_params()

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for i, hole_name in enumerate(BOREHOLES.keys()):
        color = COLORS[i]
        mouth, bottom = BOREHOLES[hole_name]
        ax.plot(
            [mouth[0], bottom[0]],
            [mouth[1], bottom[1]],
            [mouth[2], bottom[2]],
            color=color,
            linewidth=2.0,
            label=hole_name,
        )

        image_name = f"图2-{i + 1}.jpg"
        for R, P, beta, C in rows_by_image.get(image_name, [])[:2]:
            draw_fracture_disk(ax, mouth, R, P, beta, C, color)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("3D reconstruction using problem2 sine parameters")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 3000)
    ax.set_zlim(0, 7000)

    output_path = OUTPUT_DIR / "attachment4_3d_planes.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=160)
    plt.close(fig)
    print(f"三维图已保存: {output_path}")


if __name__ == "__main__":
    main()
