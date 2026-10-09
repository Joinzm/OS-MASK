#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依据批量质量评分总表筛选高质量样本，并整理生成标签、OSM、原图及对比图。

输入总表仅需两列：
    sample_name,Q_mean

筛选规则：
    Q_mean > threshold

Q_mean 为空的样本表示质量分数不可计算，会被安全跳过，不视为 0 分。

输出：
    <output_dir>/selected_scores.csv
    <output_dir>/<sample_name>/
        image.<ext>
        08_final_mask.png
        osm_label.png
        overlay.png
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LOGGER = logging.getLogger("filter_quality_results_v1")


def read_scores(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"sample_name", "Q_mean"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"评分表缺少必要字段 {sorted(required)}，实际字段为 {reader.fieldnames}"
            )
        return list(reader)


def parse_score(value: str) -> Optional[float]:
    text = (value or "").strip()
    if not text or text.lower() in {"null", "none", "nan"}:
        return None
    try:
        score = float(text)
    except ValueError:
        return None
    return score if np.isfinite(score) else None


def resolve_named_file(directory: Path, sample_name: str) -> Optional[Path]:
    """优先使用完全同名文件；必要时按 stem 查找常见影像后缀。"""
    exact = directory / sample_name
    if exact.is_file():
        return exact

    stem = Path(sample_name).stem
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_mask(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (shape[1], shape[0]):
            image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        array = np.asarray(image)

    if array.ndim == 2:
        return array > 0
    if array.ndim == 3 and array.shape[2] == 4:
        return (array[..., :3] > 0).any(axis=2) & (array[..., 3] > 0)
    if array.ndim == 3:
        return (array[..., :3] > 0).any(axis=2)
    raise ValueError(f"不支持的 mask 维度 {array.shape}: {path}")


def overlay_image(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """以红色半透明覆盖前景，不改变背景。"""
    result = image.astype(np.float32).copy()
    overlay = np.zeros_like(result)
    overlay[..., 0] = 255.0
    result[mask] = (1.0 - alpha) * result[mask] + alpha * overlay[mask]
    return np.clip(result, 0, 255).astype(np.uint8)


def save_comparison_overlay(
    image_path: Path,
    generated_mask_path: Path,
    osm_mask_path: Path,
    output_path: Path,
    sample_name: str,
    q_mean: float,
    alpha: float,
) -> None:
    image = read_rgb(image_path)
    shape = image.shape[:2]
    generated = read_mask(generated_mask_path, shape)
    osm = read_mask(osm_mask_path, shape)

    generated_overlay = overlay_image(image, generated, alpha)
    osm_overlay = overlay_image(image, osm, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(generated_overlay)
    axes[0].set_title(f"Generated mask | Q_mean={q_mean:.4e}")
    axes[1].imshow(osm_overlay)
    axes[1].set_title("OSM raster")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(sample_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def copy_image(image_path: Path, target_dir: Path) -> Path:
    suffix = image_path.suffix.lower() or ".png"
    target = target_dir / f"image{suffix}"
    shutil.copy2(image_path, target)
    return target


def filter_results(
    score_csv: Path,
    left_threshold: float,
    right_threshold: float,
    generated_root: Path,
    osm_dir: Path,
    image_dir: Path,
    output_dir: Path,
    final_mask_name: str,
    alpha: float,
) -> Path:
    if not (
        0.0 <= left_threshold < right_threshold <= 1.0
    ):
        raise ValueError(
            "必须满足 0 <= LEFT_THRESHOLD < RIGHT_THRESHOLD <= 1"
    )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]")

    score_csv = score_csv.resolve()
    generated_root = generated_root.resolve()
    osm_dir = osm_dir.resolve()
    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()

    for path, name in (
        (score_csv, "评分 CSV"),
        (generated_root, "生成结果目录"),
        (osm_dir, "OSM 目录"),
        (image_dir, "原图目录"),
    ):
        if name == "评分 CSV" and not path.is_file():
            raise FileNotFoundError(f"{name}不存在: {path}")
        if name != "评分 CSV" and not path.is_dir():
            raise NotADirectoryError(f"{name}不存在: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_scores(score_csv)
    selected: List[Tuple[str, float]] = []
    skipped_null = 0

    for row in rows:
        sample_name = (row.get("sample_name") or "").strip()
        q_mean = parse_score(row.get("Q_mean", ""))
        if not sample_name:
            LOGGER.warning("跳过空 sample_name 行")
            continue
        if q_mean is None:
            skipped_null += 1
            continue
        if not (
            left_threshold < q_mean <= right_threshold
        ):
            continue

        generated_mask = generated_root / sample_name / final_mask_name
        osm_mask = resolve_named_file(osm_dir, sample_name)
        image_path = resolve_named_file(image_dir, sample_name)

        missing: List[str] = []
        if not generated_mask.is_file():
            missing.append(str(generated_mask))
        if osm_mask is None:
            missing.append(str(osm_dir / sample_name))
        if image_path is None:
            missing.append(str(image_dir / sample_name))
        if missing:
            LOGGER.warning("跳过 %s，缺少文件: %s", sample_name, "; ".join(missing))
            continue

        sample_output = output_dir / sample_name
        sample_output.mkdir(parents=True, exist_ok=True)

        copy_image(image_path, sample_output)
        shutil.copy2(generated_mask, sample_output / final_mask_name)
        shutil.copy2(osm_mask, sample_output / "osm_label.png")
        save_comparison_overlay(
            image_path=image_path,
            generated_mask_path=generated_mask,
            osm_mask_path=osm_mask,
            output_path=sample_output / "overlay.png",
            sample_name=sample_name,
            q_mean=q_mean,
            alpha=alpha,
        )
        selected.append((sample_name, q_mean))

    LOGGER.info(
        "筛选完成：%d 个样本，规则 %.6f < Q_mean <= %.6f",
        len(selected),
        left_threshold,
        right_threshold
    )
    if skipped_null:
        LOGGER.info("跳过 %d 个 Q_mean 为空的不可评分样本", skipped_null)
    LOGGER.info("输出目录：%s", output_dir)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 Q_mean 筛选高质量生成标签")
    parser.add_argument("--score-csv", type=Path, required=True, help="quality_scores.csv")
    parser.add_argument("--threshold", type=float, required=True, help="筛选阈值，严格使用 Q_mean > threshold")
    parser.add_argument(
        "--generated-root",
        type=Path,
        required=True,
        help="类别生成结果目录；样本目录下包含 08_final_mask.png",
    )
    parser.add_argument("--osm-dir", type=Path, required=True, help="OSM 栅格标签目录")
    parser.add_argument("--image-dir", type=Path, required=True, help="原始影像目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="筛选结果输出目录")
    parser.add_argument(
        "--final-mask-name",
        default="08_final_mask.png",
        help="最终标签文件名，默认 08_final_mask.png",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="overlay 掩膜透明度，默认 0.45")
    return parser.parse_args()

def main() -> None:
    """批量生成不同质量区间筛选结果。"""

    # ============================================================
    # 用户配置区
    # ============================================================
    # AREA = "Phoenix"
    AREA = "Brussels"
    # CATEGORY_NAME = "pond"
    CATEGORY_NAMEs = [ "tennis_court",
                        "basketball_court",
                        "helipad",
                        "swimming_pool",
                        "pond",]
    for CATEGORY_NAME in CATEGORY_NAMEs:
        
        START_THRESHOLD = 0.40
        END_THRESHOLD = 1.00
        STEP = 0.05

        SCORE_CSV = Path(
            f"/home/zhaoming/sam3-main/{AREA}_quality_scores/{CATEGORY_NAME}/quality_scores.csv"
        )

        GENERATED_ROOT = Path(
            f"/home/zhaoming/sam3-main/{AREA}_sam3_mask_check_v3/{CATEGORY_NAME}"
        )

        OSM_DIR = Path(
            f"/home/zhaoming/ohsome2label/ohsome2label-master/example_result_{AREA}_{CATEGORY_NAME}/labels"
        )

        IMAGE_DIR = Path(
            f"/home/zhaoming/ohsome2label/ohsome2label-master/example_result_{AREA}_{CATEGORY_NAME}/images"
        )

        OUTPUT_ROOT = Path(
            f"/home/zhaoming/sam3-main/{AREA}_quality_selected"
        )


        # ============================================================
        # 固定配置
        # ============================================================
        FINAL_MASK_NAME = "08_final_mask.png"
        OVERLAY_ALPHA = 0.45

        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s | %(message)s"
        )

        # ============================================================
        # Batch filtering
        # ============================================================
        thresholds = np.arange(
            START_THRESHOLD,
            END_THRESHOLD,
            STEP
        )

        for left_threshold in thresholds:

            right_threshold = min(
                left_threshold + STEP,
                END_THRESHOLD
            )

            # 避免浮点误差
            left_threshold = round(
                float(left_threshold),
                2
            )

            right_threshold = round(
                float(right_threshold),
                2
            )

            output_dir = OUTPUT_ROOT / (
                f"{CATEGORY_NAME}_"
                f"{left_threshold:.2f}_"
                f"{right_threshold:.2f}"
            )

            logging.info(
                "Processing Q_mean interval: %.2f - %.2f",
                left_threshold,
                right_threshold
            )

            filter_results(
                score_csv=SCORE_CSV,
                left_threshold=left_threshold,
                right_threshold=right_threshold,
                generated_root=GENERATED_ROOT,
                osm_dir=OSM_DIR,
                image_dir=IMAGE_DIR,
                output_dir=output_dir,
                final_mask_name=FINAL_MASK_NAME,
                alpha=OVERLAY_ALPHA,
            )

if __name__ == "__main__":
    main()