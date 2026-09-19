# -*- coding: utf-8 -*-
"""
2025 华为杯 C 题 问题1：附件1 图像预处理 + K-means 裂隙候选提取

运行环境：
    conda activate huawei_c1

主流程：
    读取附件1图像
    -> 中值滤波
    -> 小波分解与去噪
    -> K-means 像素聚类
    -> 选择暗色线状类作为裂隙候选
    -> 输出二值掩膜
"""

from pathlib import Path

import cv2
import numpy as np
import pywt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 路径配置：main.py 位于 D:/华为杯数学建模/问题1代码/
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
ATTACH1_DIR = (
    BASE_DIR
    / "2025年研究生数学建模竞赛赛题"
    / "C题"
    / "C题"
    / "附件1"
)
OUTPUT_DIR = CODE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# K-means 聚类数：通常取 3~5，便于区分暗裂隙、岩石背景、亮伪影等
N_CLUSTERS = 3


def read_gray(image_path: Path) -> np.ndarray:
    """读取单通道灰度图像。"""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return image


def write_image(image_path: Path, image: np.ndarray) -> None:
    # Support Chinese path on Windows.
    suffix = image_path.suffix or ".png"
    success, buffer = cv2.imencode(suffix, image)
    if not success:
        raise IOError(f"无法写入图像: {image_path}")
    buffer.tofile(str(image_path))


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """中值滤波：去除孤立点噪声，同时尽量保留线状结构。"""
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def wavelet_denoise(
    image: np.ndarray,
    wavelet: str = "db2",
    level: int = 2,
    mode: str = "soft",
) -> np.ndarray:
    """
    小波分解去噪：对高频细节系数做软阈值收缩，压制纹理噪声。

    Parameters
    ----------
    image : 二维灰度图
    wavelet : 小波基
    level : 分解层数
    mode : 阈值模式，'soft' 或 'hard'
    """
    coeffs = pywt.wavedec2(image.astype(np.float32), wavelet, level=level)
    approx = coeffs[0]

    # 用最细尺度的高频对角子带估计噪声标准差
    finest_hh = coeffs[-1][2]
    sigma = np.median(np.abs(finest_hh - np.median(finest_hh))) / 0.6745
    sigma = max(sigma, 1e-6)

    # 通用阈值
    threshold = sigma * np.sqrt(2 * np.log(max(image.size, 2)))

    new_coeffs = [approx]
    for cH, cV, cD in coeffs[1:]:
        new_coeffs.append(
            (
                pywt.threshold(cH, threshold, mode=mode),
                pywt.threshold(cV, threshold, mode=mode),
                pywt.threshold(cD, threshold, mode=mode),
            )
        )

    reconstructed = pywt.waverec2(new_coeffs, wavelet)
    # 逆变换尺寸可能与原图不同，裁剪回原始尺寸
    return reconstructed[: image.shape[0], : image.shape[1]]


def build_pixel_features(image: np.ndarray) -> np.ndarray:
    """
    为每个像素构造聚类特征：
    [灰度值, 局部均值, 局部方差]
    """
    image_f = image.astype(np.float32)
    local_mean = cv2.blur(image_f, (5, 5))
    local_sq_mean = cv2.blur(image_f * image_f, (5, 5))
    local_var = local_sq_mean - local_mean * local_mean

    features = np.stack(
        [image_f, local_mean, local_var],
        axis=-1,
    ).reshape(-1, 3)
    return features


def kmeans_segment(image: np.ndarray, n_clusters: int = N_CLUSTERS):
    """
    对预处理后的图像做 K-means 聚类，并返回裂隙候选掩膜。

    选择平均灰度最低的类作为“裂隙候选”，因为裂隙在钻孔图中通常偏暗。
    """
    features = build_pixel_features(image)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    label_image = labels.reshape(image.shape)

    # 计算每个聚类的平均灰度，暗类作为裂隙候选
    cluster_mean_gray = []
    for k in range(n_clusters):
        cluster_pixels = image[label_image == k]
        if cluster_pixels.size > 0:
            cluster_mean_gray.append(float(cluster_pixels.mean()))
        else:
            cluster_mean_gray.append(255.0)

    dark_cluster = int(np.argmin(cluster_mean_gray))
    mask = (label_image == dark_cluster).astype(np.uint8)
    return mask, label_image, kmeans


def process_one(image_path: Path) -> Path:
    """处理单张附件1图像，返回输出文件路径。"""
    image = read_gray(image_path)

    # 1) 中值滤波
    image_median = median_filter(image, kernel_size=3)

    # 2) 小波分解去噪
    image_wavelet = wavelet_denoise(image_median, wavelet="db2", level=2)
    image_preprocessed = np.clip(image_wavelet, 0, 255).astype(np.uint8)

    # 3) K-means 聚类，选出暗色候选
    candidate_mask, _, _ = kmeans_segment(image_preprocessed)

    # 4) 输出二值图：裂隙像素=0（黑），背景像素=255（白）
    binary_result = np.where(candidate_mask > 0, 0, 255).astype(np.uint8)
    output_path = OUTPUT_DIR / f"{image_path.stem}_mask.png"
    write_image(output_path, binary_result)
    return output_path


def main() -> None:
    image_paths = sorted(ATTACH1_DIR.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"未找到附件1图像，请检查路径: {ATTACH1_DIR}")

    print(f"发现 {len(image_paths)} 张附件1图像")
    for image_path in image_paths:
        output_path = process_one(image_path)
        print(f"处理完成: {image_path.name} -> {output_path.name}")

    print(f"全部结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
