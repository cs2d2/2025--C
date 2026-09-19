# 问题1 PyCharm 代码目录

本目录用于存放 2025 年华为杯 C 题问题1（基于像素分类的裂隙智能识别）的 Python 代码、结果图与中间数据。

## 已创建的 Conda 环境

- 环境名：`huawei_c1`
- Python：3.11
- 环境路径：`D:\anaconda\envs\huawei_c1`
- 主要包：numpy、scipy、scikit-image、scikit-learn、matplotlib、pandas、opencv-python、PyWavelets

激活环境：

```powershell
conda activate huawei_c1
```

## 在 PyCharm 中使用该环境

1. 打开 PyCharm，选择本目录作为项目目录。
2. `File -> Settings -> Project -> Python Interpreter -> Add Interpreter -> Conda Environment`。
3. 选择 `Existing environment`，解释器路径填：

```text
D:\anaconda\envs\huawei_c1\python.exe
```

4. 确认后即可在 PyCharm 中运行本目录下的脚本。

## 依赖说明

- `numpy`、`scipy`：数值计算与信号/图像处理。
- `scikit-image`：形态学、连通域、Frangi 滤波、骨架化等。
- `scikit-learn`：SVM、随机森林、K-means、PCA、分类指标等。
- `opencv-python`：图像读写、灰度化、滤波、边缘检测、阈值分割等。
- `PyWavelets`：小波去噪、多尺度纹理/结构分离。
- `matplotlib`、`pandas`：结果可视化与结果表整理。

## 建议目录结构

```text
问题1代码/
├─ environment.yml
├─ README.md
├─ src/                # 核心算法代码
├─ data/               # 附件1输入图像
├─ output/             # 二值化识别结果
└─ main.py             # 入口脚本
```
