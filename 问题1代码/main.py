# -*- coding: utf-8 -*-
"""
2025 华为杯 C 题 问题1：附件1 图像预处理 + K-means 裂隙候选提取

运行环境：
    conda activate huawei_c1

流程：
    1. 中值滤波
    2. 小波分解去噪
    3. 计算 Frangi 暗线响应
    4. 用前3张图拟合 K-means 模型
    5. 在所有图上预测并选择裂隙类
    6. 形态学闭运算 + 删除小连通域
    7. 输出二值图
"""

from pathlib import Path

import cv2
import numpy as np
import pywt
from skimage import measure
from skimage.filters import frangi
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
ATTACH1_DIR = BASE_DIR / "2025年研究生数学建模竞赛赛题" / "C题" / "C题" / "附件1"
OUTPUT_DIR = CODE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

N_CLUSTERS = 4
MIN_COMPONENT_AREA = 50
MAX_FIT_SAMPLES = 200000
MIN_LINE_LENGTH = 40
MAX_LINE_WIDTH = 18
CLOSE_KERNEL_SIZE = (3, 3)


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


def median_filter(image, kernel_size=3):
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def wavelet_denoise(image, wavelet="db2", level=2, mode="soft"):
    coeffs = pywt.wavedec2(image.astype(np.float32), wavelet, level=level)
    approx = coeffs[0]
    finest_hh = coeffs[-1][2]
    sigma = np.median(np.abs(finest_hh - np.median(finest_hh))) / 0.6745
    sigma = max(sigma, 1e-6)
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
    return reconstructed[: image.shape[0], : image.shape[1]]


def frangi_response(image):
    response = frangi(image, sigmas=range(1, 6, 2), black_ridges=True)
    min_val = float(response.min())
    max_val = float(response.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(response, dtype=np.float32)
    return ((response - min_val) / (max_val - min_val)).astype(np.float32)


def preprocess(image):
    image_median = median_filter(image, kernel_size=3)
    image_wavelet = wavelet_denoise(image_median, wavelet="db2", level=2)
    image_preprocessed = np.clip(image_wavelet, 0, 255).astype(np.uint8)
    image_frangi = frangi_response(image_preprocessed)
    return image_preprocessed, image_frangi


def build_pixel_features(image, frangi_norm):
    image_f = image.astype(np.float32)
    local_mean = cv2.blur(image_f, (5, 5))
    local_sq_mean = cv2.blur(image_f * image_f, (5, 5))
    local_var = local_sq_mean - local_mean * local_mean
    features = np.stack([image_f, local_mean, local_var, frangi_norm], axis=-1).reshape(-1, 4)
    return features


def postprocess_mask(mask):
    mask = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL_SIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            cleaned[labels == i] = 1
    return cleaned


def line_score(mask):
    cleaned = postprocess_mask(mask)
    if cleaned.sum() == 0:
        return 0.0
    label_image = measure.label(cleaned, connectivity=2)
    props = measure.regionprops(label_image)
    if not props:
        return 0.0
    long_pixels = 0
    for prop in props:
        if prop.axis_major_length >= MIN_LINE_LENGTH and prop.axis_minor_length <= MAX_LINE_WIDTH:
            long_pixels += prop.area
    return long_pixels / max(cleaned.sum(), 1)


def cluster_score(image, frangi_norm, mask):
    if mask.sum() == 0:
        return -1e9
    darkness = (255.0 - float(image[mask > 0].mean())) / 255.0
    frangi_mean = float(frangi_norm[mask > 0].mean())
    line = line_score(mask)
    return darkness + 2.0 * frangi_mean + 1.0 * line


def fit_kmeans(train_preprocessed, train_frangi, n_clusters=N_CLUSTERS):
    all_features = []
    for image, frangi_norm in zip(train_preprocessed, train_frangi):
        all_features.append(build_pixel_features(image, frangi_norm))
    features = np.vstack(all_features)

    rng = np.random.default_rng(42)
    if features.shape[0] > MAX_FIT_SAMPLES:
        index = rng.choice(features.shape[0], size=MAX_FIT_SAMPLES, replace=False)
        features = features[index]

    scaler = StandardScaler().fit(features)
    features_scaled = scaler.transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=200)
    kmeans.fit(features_scaled)
    return scaler, kmeans


def select_fracture_cluster(scaler, kmeans, train_preprocessed, train_frangi, n_clusters=N_CLUSTERS):
    scores = np.zeros(n_clusters, dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int32)
    for image, frangi_norm in zip(train_preprocessed, train_frangi):
        features = build_pixel_features(image, frangi_norm)
        labels = kmeans.predict(scaler.transform(features))
        label_image = labels.reshape(image.shape)
        for k in range(n_clusters):
            mask = (label_image == k).astype(np.uint8)
            if mask.sum() == 0:
                continue
            scores[k] += cluster_score(image, frangi_norm, mask)
            counts[k] += 1
    valid = counts > 0
    scores[valid] /= counts[valid]
    return int(np.argmax(scores))


def segment_image(image, frangi_norm, scaler, kmeans, fracture_cluster):
    features = build_pixel_features(image, frangi_norm)
    labels = kmeans.predict(scaler.transform(features))
    label_image = labels.reshape(image.shape)
    mask = (label_image == fracture_cluster).astype(np.uint8)
    mask = postprocess_mask(mask)
    return mask


def main():
    image_paths = sorted(ATTACH1_DIR.glob("*.jpg"), key=lambda p: int(p.stem.split("-")[1]))
    if not image_paths:
        raise FileNotFoundError(f"未找到附件1图像: {ATTACH1_DIR}")

    print(f"发现 {len(image_paths)} 张附件1图像，逐张直接进行 K-means 聚类")

    for path in image_paths:
        image, frangi_norm = preprocess(read_gray(path))

        # 每一张图单独拟合 K-means，不做训练/验证拆分
        scaler, kmeans = fit_kmeans([image], [frangi_norm])
        fracture_cluster = select_fracture_cluster(
            scaler, kmeans, [image], [frangi_norm]
        )
        mask = segment_image(image, frangi_norm, scaler, kmeans, fracture_cluster)

        result = np.where(mask > 0, 0, 255).astype(np.uint8)
        output_path = OUTPUT_DIR / f"{path.stem}_mask.png"
        write_image(output_path, result)
        print(f"处理完成: {path.name} -> 聚类编号 {fracture_cluster} -> {output_path.name}")

    print(f"结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
