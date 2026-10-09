#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OS-MASK 单图测试

功能：
1. 从二值 label 的每个外部轮廓分别生成 1 个 box，并以轮廓中心扩大指定倍数；
2. 每个轮廓分别采样内部正点，使用 SAM3 官方 predict_inst 接口执行 point prompt；
3. 对比 text、box、point、box+text 联合提示；
4. box+text 对每个 box 独立推理，并通过强/弱文本语义证据过滤错误 box；
5. text-only 先按置信度阈值过滤，再仅允许可靠联合候选补全边界或按配置恢复漏检；
6. 保存每种方案的叠加图、合并 mask、实例 mask和运行信息。

依赖：numpy、opencv-python、Pillow、matplotlib、torch、官方 facebookresearch/sam3。
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

Box = Tuple[int, int, int, int]  # xyxy，右下角为开区间
Point = Tuple[int, int]


@dataclass
class PromptInstance:
    """由 label 的一个外部轮廓生成的一组几何提示。"""

    index: int
    mask: np.ndarray
    box: Box
    points: List[Point]
    contour_area: float


@dataclass
class Prediction:
    """统一保存不同 SAM3 接口返回的结果。"""

    masks: List[np.ndarray] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    boxes: List[Box] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.masks)


@dataclass(frozen=True)
class JointGateConfig:
    """box+text 联合结果的门控参数。"""

    min_joint_score: float = 0.40
    min_inside_box: float = 0.25
    min_strong_overlap: float = 0.30
    min_strong_candidate_coverage: float = 0.05
    min_weak_overlap: float = 0.35
    min_weak_candidate_coverage: float = 0.08
    min_new_instance_score: float = 0.55
    allow_new_instances: bool = False


# -----------------------------------------------------------------------------
# 输入与几何提示
# -----------------------------------------------------------------------------


def load_rgb_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"影像不存在: {path}")
    return Image.open(path).convert("RGB")


def load_binary_label(
    path: Path,
    target_size: Tuple[int, int],
    threshold: int = 0,
) -> np.ndarray:
    """读取灰度/RGB/RGBA label，输出 H×W bool mask。"""
    if not path.is_file():
        raise FileNotFoundError(f"label 不存在: {path}")

    label = Image.open(path)
    if label.size != target_size:
        label = label.resize(target_size, Image.Resampling.NEAREST)

    array = np.asarray(label)
    if array.ndim == 2:
        mask = array > threshold
    elif array.ndim == 3 and array.shape[2] == 4:
        mask = (array[..., :3] > threshold).any(axis=2) & (array[..., 3] > threshold)
    elif array.ndim == 3:
        mask = (array[..., :3] > threshold).any(axis=2)
    else:
        raise ValueError(f"不支持的 label 维度: {array.shape}")

    return mask.astype(bool)


def expand_box(box: Box, scale: float, image_shape: Tuple[int, int]) -> Box:
    """围绕 box 中心等比例放大，并裁剪到影像范围。"""
    if scale < 1.0:
        raise ValueError("box_scale 必须大于或等于 1.0")

    x0, y0, x1, y1 = box
    image_h, image_w = image_shape
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    new_width, new_height = width * scale, height * scale

    new_x0 = max(0, math.floor(center_x - new_width / 2.0))
    new_y0 = max(0, math.floor(center_y - new_height / 2.0))
    new_x1 = min(image_w, math.ceil(center_x + new_width / 2.0))
    new_y1 = min(image_h, math.ceil(center_y + new_height / 2.0))
    return new_x0, new_y0, max(new_x0 + 1, new_x1), max(new_y0 + 1, new_y1)


def sample_interior_points(mask: np.ndarray, max_points: int) -> List[Point]:
    """
    从单个轮廓内部采样稳定正点。

    使用距离变换优先选择远离边界的位置，并对已选位置做抑制，
    比按前景数组顺序均匀采样更不容易把点放到狭窄边缘或噪声处。
    """
    if max_points <= 0 or not mask.any():
        return []

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    work = distance.copy()
    points: List[Point] = []
    suppress_radius = max(2, int(round(math.sqrt(float(mask.sum()) / max_points) / 2.0)))

    for _ in range(max_points):
        flat_index = int(np.argmax(work))
        y, x = np.unravel_index(flat_index, work.shape)
        if work[y, x] <= 0:
            break
        points.append((int(x), int(y)))
        cv2.circle(work, (int(x), int(y)), suppress_radius, 0.0, thickness=-1)

    if not points:
        ys, xs = np.where(mask)
        points.append((int(xs[len(xs) // 2]), int(ys[len(ys) // 2])))
    return points


def extract_prompt_instances(
    label_mask: np.ndarray,
    box_scale: float = 1.5,
    points_per_instance: int = 3,
) -> List[PromptInstance]:
    """每个外部轮廓单独生成 component mask、扩大 box 和正点。"""
    contours, _ = cv2.findContours(
        label_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    # 按面积降序，保证输出顺序稳定且主要目标优先。
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    instances: List[PromptInstance] = []

    for index, contour in enumerate(contours):
        component_mask = np.zeros_like(label_mask, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 1, thickness=cv2.FILLED)
        component_mask = component_mask.astype(bool)

        x, y, width, height = cv2.boundingRect(contour)
        raw_box: Box = (x, y, x + width, y + height)
        box = expand_box(raw_box, box_scale, label_mask.shape)
        points = sample_interior_points(component_mask, points_per_instance)

        instances.append(
            PromptInstance(
                index=index,
                mask=component_mask,
                box=box,
                points=points,
                contour_area=float(cv2.contourArea(contour)),
            )
        )
    return instances


def xyxy_to_normalized_cxcywh(box: Box, image_size: Tuple[int, int]) -> List[float]:
    """转换为 Sam3Processor.add_geometric_prompt 所需的归一化 cxcywh。"""
    x0, y0, x1, y1 = box
    image_w, image_h = image_size
    return [
        ((x0 + x1) / 2.0) / image_w,
        ((y0 + y1) / 2.0) / image_h,
        (x1 - x0) / image_w,
        (y1 - y0) / image_h,
    ]


# -----------------------------------------------------------------------------
# SAM3 输出整理
# -----------------------------------------------------------------------------


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0)
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def normalize_masks(value: Any) -> List[np.ndarray]:
    """将 N×1×H×W、N×H×W 或 H×W 统一为 bool mask 列表。"""
    array = to_numpy(value)
    if array.size == 0:
        return []
    if array.ndim == 2:
        array = array[None, ...]
    else:
        array = array.reshape(-1, array.shape[-2], array.shape[-1])

    threshold = 0.0 if array.min() < 0.0 or array.max() > 1.0 else 0.5
    return [(mask > threshold).astype(bool) for mask in array if (mask > threshold).any()]


def mask_to_box(mask: np.ndarray) -> Box:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 1, 1
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def prediction_from_processor_state(state: Optional[Dict[str, Any]]) -> Prediction:
    if not state:
        return Prediction()

    masks = normalize_masks(state.get("masks"))
    score_array = to_numpy(state.get("scores")).reshape(-1)
    box_array = to_numpy(state.get("boxes"))
    if box_array.size:
        box_array = box_array.reshape(-1, 4)

    scores = [float(score_array[i]) if i < len(score_array) else 0.0 for i in range(len(masks))]
    boxes = [
        tuple(int(round(v)) for v in box_array[i]) if i < len(box_array) else mask_to_box(mask)
        for i, mask in enumerate(masks)
    ]
    return Prediction(masks=masks, scores=scores, boxes=boxes)


def filter_prediction_by_score(
    prediction: Prediction,
    min_score: float,
) -> Prediction:
    """按置信度过滤预测，并保持 mask、score、box 一一对应。"""
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score 必须位于 [0, 1]")

    kept_indices = [
        index
        for index, score in enumerate(prediction.scores)
        if score >= min_score
    ]
    return Prediction(
        masks=[prediction.masks[index] for index in kept_indices],
        scores=[prediction.scores[index] for index in kept_indices],
        boxes=[prediction.boxes[index] for index in kept_indices],
    )


def best_interactive_mask(masks: Any, scores: Any) -> Tuple[Optional[np.ndarray], float]:
    """从 predict_inst 的多 mask 输出中选择质量分最高的一张。"""
    mask_list = normalize_masks(masks)
    score_array = to_numpy(scores).reshape(-1)
    if not mask_list:
        return None, 0.0

    valid_scores = score_array[: len(mask_list)]
    best_index = int(np.argmax(valid_scores)) if len(valid_scores) else 0
    best_score = float(valid_scores[best_index]) if len(valid_scores) else 0.0
    return mask_list[best_index], best_score


# -----------------------------------------------------------------------------
# 四种推理方案
# -----------------------------------------------------------------------------


def clone_image_state(base_state: Dict[str, Any]) -> Dict[str, Any]:
    """复制 state 的字典结构，共享只读图像特征，避免重复运行视觉编码器。"""
    state = dict(base_state)
    state["backbone_out"] = dict(base_state["backbone_out"])
    return state


def predict_text(
    processor: Any,
    base_state: Dict[str, Any],
    prompt: str,
    min_score: float = 0.0,
) -> Prediction:
    """执行 text prompt，并在输出和融合前统一过滤低置信实例。"""
    state = clone_image_state(base_state)
    output_state = processor.set_text_prompt(state=state, prompt=prompt)
    prediction = prediction_from_processor_state(output_state)
    return filter_prediction_by_score(prediction, min_score)


def predict_box(model: Any, base_state: Dict[str, Any], instances: Sequence[PromptInstance]) -> Prediction:
    """每个轮廓的 box 独立调用交互式实例分割，防止多个实例被合并。"""
    prediction = Prediction()
    for instance in instances:
        input_box = np.asarray(instance.box, dtype=np.float32)[None, :]
        masks, scores, _ = model.predict_inst(
            base_state,
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        mask, score = best_interactive_mask(masks, scores)
        if mask is not None:
            prediction.masks.append(mask)
            prediction.scores.append(score)
            prediction.boxes.append(mask_to_box(mask))
    return prediction


def predict_point(model: Any, base_state: Dict[str, Any], instances: Sequence[PromptInstance]) -> Prediction:
    prediction = Prediction()
    for instance in instances:
        if not instance.points:
            continue
        points = np.asarray(instance.points, dtype=np.float32)
        labels = np.ones(len(points), dtype=np.int32)
        masks, scores, _ = model.predict_inst(
            base_state,
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        mask, score = best_interactive_mask(masks, scores)
        if mask is not None:
            prediction.masks.append(mask)
            prediction.scores.append(score)
            prediction.boxes.append(mask_to_box(mask))
    return prediction


def predict_box_text_joint(
    processor: Any,
    base_state: Dict[str, Any],
    prompt: str,
    instances: Sequence[PromptInstance],
    image_size: Tuple[int, int],
    strong_text_prediction: Prediction,
    weak_text_prediction: Prediction,
    gate: JointGateConfig,
    stats: Optional[Dict[str, Any]] = None,
) -> Prediction:
    """
    对每个 box 独立执行 text+box 联合提示，并过滤错误 box 产生的伪结果。

    关键设计：
    1. 每个 box 都从干净 state 开始，避免多个正 box 在同一 state 中累积放大误检；
    2. 每个 box 最多保留一个候选；
    3. 候选必须与当前 box 有空间关联；
    4. 默认必须得到高阈值 text-only 结果支持；
    5. 仅在 allow_new_instances=True 时，允许高置信候选借助低阈值文本响应恢复漏检。
    """
    counters: Dict[str, Any] = {
        "box_count": len(instances),
        "raw_candidate_count": 0,
        "accepted_box_count": 0,
        "accepted_existing_count": 0,
        "accepted_new_count": 0,
        "rejected_low_score": 0,
        "rejected_low_box_support": 0,
        "rejected_no_semantic_support": 0,
    }
    accepted = Prediction()

    for instance in instances:
        # 每个 box 使用独立 state，绝不把错误 box 累积到其他 box 上。
        state = clone_image_state(base_state)
        processor.set_text_prompt(state=state, prompt=prompt)
        output_state = processor.add_geometric_prompt(
            box=xyxy_to_normalized_cxcywh(instance.box, image_size),
            label=True,
            state=state,
        )
        candidates = prediction_from_processor_state(output_state)
        counters["raw_candidate_count"] += candidates.count

        best: Optional[Tuple[np.ndarray, float, Box, str]] = None
        best_rank = -1.0

        for mask, score, output_box in zip(
            candidates.masks, candidates.scores, candidates.boxes
        ):
            if score < gate.min_joint_score:
                counters["rejected_low_score"] += 1
                continue

            inside_ratio = fraction_inside_box(mask, instance.box)
            if inside_ratio < gate.min_inside_box:
                counters["rejected_low_box_support"] += 1
                continue

            strong_overlap, strong_coverage, _ = best_semantic_support(
                mask, strong_text_prediction
            )
            existing_supported = (
                strong_overlap >= gate.min_strong_overlap
                and strong_coverage >= gate.min_strong_candidate_coverage
            )

            weak_overlap, weak_coverage, _ = best_semantic_support(
                mask, weak_text_prediction
            )
            new_supported = (
                gate.allow_new_instances
                and score >= gate.min_new_instance_score
                and weak_overlap >= gate.min_weak_overlap
                and weak_coverage >= gate.min_weak_candidate_coverage
            )

            if existing_supported:
                support_type = "existing"
                semantic_quality = math.sqrt(max(0.0, strong_overlap * strong_coverage))
            elif new_supported:
                support_type = "new"
                semantic_quality = math.sqrt(max(0.0, weak_overlap * weak_coverage))
            else:
                counters["rejected_no_semantic_support"] += 1
                continue

            # 置信度、空间关联和文本语义支持共同决定当前 box 的最佳候选。
            rank = (
                float(score)
                * (0.35 + 0.65 * inside_ratio)
                * (0.35 + 0.65 * semantic_quality)
            )
            if rank > best_rank:
                best_rank = rank
                best = (mask, float(score), output_box, support_type)

        # 每个 box 最多贡献一个联合结果，避免一次提示返回整幅图所有文本目标。
        if best is not None:
            mask, score, output_box, support_type = best
            accepted.masks.append(mask.copy())
            accepted.scores.append(score)
            accepted.boxes.append(output_box if output_box else mask_to_box(mask))
            counters["accepted_box_count"] += 1
            counters[f"accepted_{support_type}_count"] += 1

    result = deduplicate_prediction(accepted, iou_threshold=0.80)
    counters["accepted_before_dedup"] = accepted.count
    counters["accepted_after_dedup"] = result.count
    if stats is not None:
        stats.clear()
        stats.update(counters)
    return result


# -----------------------------------------------------------------------------
# box+text 保守融合
# -----------------------------------------------------------------------------


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def mask_overlap_min(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """交集占较小 mask 面积的比例，适合比较完整 mask 与局部文本响应。"""
    area_a, area_b = int(mask_a.sum()), int(mask_b.sum())
    if area_a == 0 or area_b == 0:
        return 0.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    return float(intersection / min(area_a, area_b))


def best_semantic_support(
    candidate: np.ndarray,
    reference: Prediction,
) -> Tuple[float, float, int]:
    """
    返回候选与参考文本 mask 的最佳语义支持：
    (较小区域重叠率、候选被参考覆盖的比例、参考 mask 下标)。

    同时约束两种比例，可以避免一个很小的弱文本噪点为巨大错误 mask 背书。
    """
    candidate_area = int(candidate.sum())
    if candidate_area == 0 or not reference.masks:
        return 0.0, 0.0, -1

    best_overlap, best_coverage, best_index, best_rank = 0.0, 0.0, -1, -1.0
    for index, ref_mask in enumerate(reference.masks):
        intersection = int(np.logical_and(candidate, ref_mask).sum())
        ref_area = int(ref_mask.sum())
        if intersection == 0 or ref_area == 0:
            continue
        overlap = intersection / min(candidate_area, ref_area)
        coverage = intersection / candidate_area
        rank = math.sqrt(overlap * coverage)
        if rank > best_rank:
            best_overlap = float(overlap)
            best_coverage = float(coverage)
            best_index = index
            best_rank = rank
    return best_overlap, best_coverage, best_index


def fraction_inside_box(mask: np.ndarray, box: Box) -> float:
    x0, y0, x1, y1 = box
    area = int(mask.sum())
    if area == 0:
        return 0.0
    return float(mask[y0:y1, x0:x1].sum() / area)


def deduplicate_prediction(prediction: Prediction, iou_threshold: float = 0.85) -> Prediction:
    """按分数从高到低移除高度重复实例。"""
    order = sorted(range(prediction.count), key=lambda i: prediction.scores[i], reverse=True)
    kept: List[int] = []
    for index in order:
        if all(mask_iou(prediction.masks[index], prediction.masks[j]) < iou_threshold for j in kept):
            kept.append(index)
    return Prediction(
        masks=[prediction.masks[i] for i in kept],
        scores=[prediction.scores[i] for i in kept],
        boxes=[prediction.boxes[i] for i in kept],
    )


def fuse_text_and_joint(
    text_prediction: Prediction,
    joint_prediction: Prediction,
    match_overlap: float = 0.45,
    min_joint_coverage: float = 0.05,
    max_union_growth: float = 1.40,
    allow_new_instances: bool = False,
) -> Prediction:
    """
    以 text-only 为不可丢弃基线，保守融合门控后的 box+text 候选。

    - 匹配已有 text mask：仅在面积增长可控时取并集，用 box 补全阴影或缺失边界；
    - 不匹配已有 text mask：默认丢弃，避免错误 box 强制产生新实例；
    - allow_new_instances=True 时，才允许已经通过弱文本门控的联合候选恢复漏检。
    """
    fused = Prediction(
        masks=[mask.copy() for mask in text_prediction.masks],
        scores=list(text_prediction.scores),
        boxes=list(text_prediction.boxes),
    )

    for joint_mask, joint_score in zip(joint_prediction.masks, joint_prediction.scores):
        overlap, joint_coverage, best_index = best_semantic_support(joint_mask, fused)

        if (
            best_index >= 0
            and overlap >= match_overlap
            and joint_coverage >= min_joint_coverage
        ):
            text_mask = fused.masks[best_index]
            union_mask = np.logical_or(text_mask, joint_mask)
            growth = float(union_mask.sum() / max(1, int(text_mask.sum())))

            # 只允许可控扩张；异常膨胀时原样保留 text 结果。
            if growth <= max_union_growth:
                fused.masks[best_index] = union_mask
                fused.scores[best_index] = max(fused.scores[best_index], joint_score)
                fused.boxes[best_index] = mask_to_box(union_mask)
        elif allow_new_instances:
            fused.masks.append(joint_mask.copy())
            fused.scores.append(joint_score)
            fused.boxes.append(mask_to_box(joint_mask))

    return deduplicate_prediction(fused, iou_threshold=0.85)


# -----------------------------------------------------------------------------
# 保存与可视化
# -----------------------------------------------------------------------------


def save_prompt_visualization(
    image: Image.Image,
    label_mask: np.ndarray,
    instances: Sequence[PromptInstance],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image)
    ax.contour(label_mask.astype(np.uint8), levels=[0.5], linewidths=2.0)

    for instance in instances:
        x0, y0, x1, y1 = instance.box
        ax.add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linestyle="--", linewidth=2)
        )
        if instance.points:
            points = np.asarray(instance.points)
            ax.scatter(points[:, 0], points[:, 1], marker="*", s=90, edgecolors="white")
        ax.text(x0, y0, str(instance.index), fontsize=9, bbox=dict(facecolor="white", alpha=0.7))

    ax.set_title(f"Label contours and prompts | instances={len(instances)}")
    ax.axis("off")
    fig.savefig(out_path, dpi=250, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_prediction(
    name: str,
    image: Image.Image,
    label_mask: np.ndarray,
    prediction: Prediction,
    instances: Sequence[PromptInstance],
    out_dir: Path,
) -> None:
    """保存实例 mask、合并 mask 和叠加图。"""
    mask_dir = out_dir / "instance_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 清理同名方案上一次运行遗留的实例 mask，避免旧的 17 个结果混入新输出。
    for stale_path in mask_dir.glob(f"{name}_*.png"):
        stale_path.unlink()

    merged = np.zeros(label_mask.shape, dtype=bool)
    for index, mask in enumerate(prediction.masks):
        if mask.shape != label_mask.shape:
            resized = cv2.resize(
                mask.astype(np.uint8),
                (label_mask.shape[1], label_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = resized.astype(bool)
            prediction.masks[index] = mask
        merged |= mask
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / f"{name}_{index:02d}.png")

    Image.fromarray(merged.astype(np.uint8) * 255).save(out_dir / f"{name}_mask.png")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image)
    colors = plt.get_cmap("tab20", max(1, prediction.count))

    for index, (mask, score, box) in enumerate(
        zip(prediction.masks, prediction.scores, prediction.boxes)
    ):
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = (*colors(index)[:3], 0.42)
        ax.imshow(overlay)
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=1.5))
        ax.text(x0, y0, f"{score:.3f}", fontsize=8, bbox=dict(facecolor="white", alpha=0.7))

    # label 轮廓始终保留，便于观察偏移和完整性差异。
    ax.contour(label_mask.astype(np.uint8), levels=[0.5], linewidths=2.0)
    for instance in instances:
        x0, y0, x1, y1 = instance.box
        ax.add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linestyle="--", linewidth=1.2)
        )

    ax.set_title(f"{name} | masks={prediction.count}")
    ax.axis("off")
    fig.savefig(out_dir / f"{name}_overlay.png", dpi=250, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def safe_predict(
    name: str,
    function: Callable[[], Prediction],
    log: Dict[str, Any],
) -> Prediction:
    try:
        prediction = function()
        log[name] = {
            "ok": True,
            "mask_count": prediction.count,
            "scores": prediction.scores,
        }
        print(f"[完成] {name}: {prediction.count} 个 mask")
        return prediction
    except Exception as error:  # 单个方案失败时继续保存其他方案
        log[name] = {"ok": False, "error": repr(error)}
        print(f"[错误] {name}: {error!r}")
        return Prediction()


def autocast_context(device: str):
    if device != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------

@torch.inference_mode() 
def main(
    image_path: Path,
    label_path: Path,
    out_dir: Path,
    text_prompt: str,
    label_threshold: int = 0,
    box_scale: float = 1.5,
    points_per_instance: int = 3,
    confidence_threshold: float = 0.25,
    text_score_threshold: float = 0.40,
    weak_text_threshold: Optional[float] = None,
    min_joint_score: float = 0.40,
    min_inside_box: float = 0.25,
    min_strong_overlap: float = 0.30,
    min_weak_overlap: float = 0.35,
    min_new_instance_score: float = 0.55,
    match_overlap: float = 0.45,
    max_union_growth: float = 1.40,
    allow_new_instances: bool = False,
    run_text: bool = True,
    run_box: bool = True,
    run_point: bool = True,
    run_box_text: bool = True,
) -> None:
    if not text_prompt.strip():
        raise ValueError("text_prompt 不能为空")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold 必须位于 [0, 1]")
    if not 0.0 <= text_score_threshold <= 1.0:
        raise ValueError("text_score_threshold 必须位于 [0, 1]")

    # 低阈值文本结果只用于语义门控，不会直接写入最终输出。
    if weak_text_threshold is None:
        weak_text_threshold = max(0.05, confidence_threshold * 0.40)
    weak_text_threshold = min(weak_text_threshold, max(0.0, confidence_threshold - 1e-4))

    gate = JointGateConfig(
        min_joint_score=min_joint_score,
        min_inside_box=min_inside_box,
        min_strong_overlap=min_strong_overlap,
        min_weak_overlap=min_weak_overlap,
        min_new_instance_score=min_new_instance_score,
        allow_new_instances=allow_new_instances,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb_image(image_path)
    label_mask = load_binary_label(label_path, image.size, label_threshold)
    if not label_mask.any():
        raise ValueError(f"label 中没有前景像素: {label_path}")

    instances = extract_prompt_instances(label_mask, box_scale, points_per_instance)
    if not instances:
        raise ValueError("未能从 label 中提取到外部轮廓")

    save_prompt_visualization(image, label_mask, instances, out_dir / "00_label_prompts.png")

    # 延迟导入，便于在未安装 SAM3 的环境中单独测试 label 几何处理函数。
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载 SAM3，设备: {device}")
    model = build_sam3_image_model(
        device=device,
        enable_inst_interactivity=True,  # point/box 的 predict_inst 必须启用
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    weak_processor = (
        Sam3Processor(model, device=device, confidence_threshold=weak_text_threshold)
        if run_box_text
        else None
    )

    log: Dict[str, Any] = {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "text_prompt": text_prompt,
        "device": device,
        "image_size": list(image.size),
        "box_scale": box_scale,
        "points_per_instance": points_per_instance,
        "contour_count": len(instances),
        "confidence_threshold": confidence_threshold,
        "text_score_threshold": text_score_threshold,
        "weak_text_threshold": weak_text_threshold,
        "joint_gate": {
            "min_joint_score": gate.min_joint_score,
            "min_inside_box": gate.min_inside_box,
            "min_strong_overlap": gate.min_strong_overlap,
            "min_strong_candidate_coverage": gate.min_strong_candidate_coverage,
            "min_weak_overlap": gate.min_weak_overlap,
            "min_weak_candidate_coverage": gate.min_weak_candidate_coverage,
            "min_new_instance_score": gate.min_new_instance_score,
            "allow_new_instances": gate.allow_new_instances,
            "match_overlap": match_overlap,
            "max_union_growth": max_union_growth,
        },
        "instances": [
            {
                "index": item.index,
                "box_xyxy": list(item.box),
                "points_xy": [list(point) for point in item.points],
                "contour_area": item.contour_area,
            }
            for item in instances
        ],
    }

    predictions: Dict[str, Prediction] = {}
    with torch.inference_mode(), autocast_context(device):
        # 只编码一次图像；后续方案共享只读视觉特征。
        base_state = processor.set_image(image)

        if run_text or run_box_text:
            predictions["01_text_prompt"] = safe_predict(
                "01_text_prompt",
                lambda: predict_text(
                    processor,
                    base_state,
                    text_prompt,
                    min_score=text_score_threshold,
                ),
                log,
            )

        if run_box:
            predictions["02_box_prompt"] = safe_predict(
                "02_box_prompt",
                lambda: predict_box(model, base_state, instances),
                log,
            )

        if run_point:
            predictions["03_point_prompt"] = safe_predict(
                "03_point_prompt",
                lambda: predict_point(model, base_state, instances),
                log,
            )

        if run_box_text:
            if weak_processor is None:
                raise RuntimeError("weak_processor 初始化失败")

            # 弱文本结果仅作为内部语义证据，不单独保存图像。
            weak_text_prediction = safe_predict(
                "04_weak_text_reference_internal",
                lambda: predict_text(
                    weak_processor,
                    base_state,
                    text_prompt,
                    min_score=weak_text_threshold,
                ),
                log,
            )
            text_prediction = predictions.get("01_text_prompt", Prediction())
            gate_stats: Dict[str, Any] = {}

            # 输出名称保持不变，但内部已改为逐 box 独立推理和文本语义门控。
            joint = safe_predict(
                "04_box_text_joint",
                lambda: predict_box_text_joint(
                    processor=processor,
                    base_state=base_state,
                    prompt=text_prompt,
                    instances=instances,
                    image_size=image.size,
                    strong_text_prediction=text_prediction,
                    weak_text_prediction=weak_text_prediction,
                    gate=gate,
                    stats=gate_stats,
                ),
                log,
            )
            predictions["04_box_text_joint"] = joint
            log["04_box_text_joint"]["filter_stats"] = gate_stats
            log["04_box_text_joint"]["note"] = (
                "每个 box 独立推理；错误 box 必须通过空间和文本语义门控才会保留"
            )

            fused = fuse_text_and_joint(
                text_prediction=text_prediction,
                joint_prediction=joint,
                match_overlap=match_overlap,
                max_union_growth=max_union_growth,
                allow_new_instances=allow_new_instances,
            )
            predictions["05_box_text_fused"] = fused
            log["05_box_text_fused"] = {
                "ok": True,
                "mask_count": fused.count,
                "scores": fused.scores,
                "note": (
                    f"保留 score >= {text_score_threshold:.2f} 的 text-only 结果；"
                    "默认仅用联合候选补全已有文本实例，禁止错误 box 单独新增实例"
                ),
            }
            print(f"[完成] 05_box_text_fused: {fused.count} 个 mask")

    for name, prediction in predictions.items():
        save_prediction(name, image, label_mask, prediction, instances, out_dir)

    with open(out_dir / "run_info.json", "w", encoding="utf-8") as file:
        json.dump(log, file, ensure_ascii=False, indent=2)

    print(f"全部完成，结果目录: {out_dir}")


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # -------------------------------------------------------------------------
    # # 路径与运行参数：日常调用只需要修改这里
    # # -------------------------------------------------------------------------
    # IMAGE_PATH = Path(
    #     "/home/zhaoming/ohsome2label/ohsome2label-master/"
    #     "example_result_Phoenix_tennis_court/images/18.49527.105146.png"
    # )
    # LABEL_PATH = Path(
    #     "/home/zhaoming/ohsome2label/ohsome2label-master/"
    #     "example_result_Phoenix_tennis_court/labels/18.49527.105146.png"
    # )
    # OUT_DIR = Path(
    #     "/home/zhaoming/sam3-main/sam3_mask_check/"
    #     "Phoenix_tennis_court/18.49527.105146.png-0.55"
    # )
    # TEXT_PROMPT = "tennis court"

    # main(
    #     image_path=IMAGE_PATH,
    #     label_path=LABEL_PATH,
    #     out_dir=OUT_DIR,
    #     text_prompt=TEXT_PROMPT,
    #     label_threshold=0,
    #     box_scale=1.5,             # 每个轮廓的外接 box 围绕中心扩大 1.5 倍
    #     points_per_instance=3,     # 每个轮廓独立采样正点
    #     confidence_threshold=0.25,  # SAM3 processor 的基础输出阈值
    #     text_score_threshold=0.55, # 01_text_prompt 及最终融合仅保留 score >= 0.35
    #     weak_text_threshold=0.10,  # 仅用于判断 box 区域是否存在弱文本语义
    #     min_joint_score=0.40,      # 联合候选最低置信度
    #     min_inside_box=0.25,       # 联合 mask 至少有多少比例位于当前提示 box 内
    #     min_strong_overlap=0.30,   # 与正式 text mask 的最小语义重叠
    #     min_weak_overlap=0.35,     # 恢复漏检时与弱 text mask 的最小语义重叠
    #     min_new_instance_score=0.55,
    #     match_overlap=0.45,        # 联合候选与 text mask 达到该值才用于补全
    #     max_union_growth=1.40,     # 补全后面积最多增长 40%，防止异常膨胀
    #     allow_new_instances=False, # 当前 text 最理想：禁止 box 单独新增实例
    #     run_text=True,
    #     run_box=True,              # 保留 02_box_prompt 对比输出
    #     run_point=True,            # 保留 03_point_prompt 对比输出
    #     run_box_text=True,
    # )

    # -------------------------------------------------------------------------
    # # 路径与运行参数：批量调用
    # # -------------------------------------------------------------------------
    from tqdm import tqdm
    
    AREA = "Phoenix"
    CATEGORY = "tennis_court"

    BASE = Path("/home/zhaoming/ohsome2label/ohsome2label-master")
    IMAGE_DIR = BASE / f"example_result_{AREA}_{CATEGORY}" / "images"
    LABEL_DIR = BASE / f"example_result_{AREA}_{CATEGORY}" / "labels"
    OUT_ROOT = Path(f"/home/zhaoming/sam3-main/{AREA}_sam3_mask_check/{CATEGORY}")

    TEXT_PROMPT = CATEGORY.replace("_", " ")   # "tennis_court" -> "tennis court"

    # 批量运行
    img_paths = sorted(IMAGE_DIR.glob("*.png"))

    for img_path in tqdm(img_paths, desc=f"{AREA}/{CATEGORY}", unit="img"):
        image_path = IMAGE_DIR / img_path.name
        label_path = LABEL_DIR / img_path.name
        out_dir = OUT_ROOT / f"{img_path.name}"
        try:
            main(
                image_path=image_path,
                label_path=label_path,
                out_dir=out_dir,
                text_prompt=TEXT_PROMPT,
                label_threshold=0,
                box_scale=1.5,             # 每个轮廓的外接 box 围绕中心扩大 1.5 倍
                points_per_instance=3,     # 每个轮廓独立采样正点
                confidence_threshold=0.25,  # SAM3 processor 的基础输出阈值
                text_score_threshold=0.55, # 01_text_prompt 及最终融合仅保留 score >= 0.35
                weak_text_threshold=0.10,  # 仅用于判断 box 区域是否存在弱文本语义
                min_joint_score=0.40,      # 联合候选最低置信度
                min_inside_box=0.25,       # 联合 mask 至少有多少比例位于当前提示 box 内
                min_strong_overlap=0.30,   # 与正式 text mask 的最小语义重叠
                min_weak_overlap=0.35,     # 恢复漏检时与弱 text mask 的最小语义重叠
                min_new_instance_score=0.55,
                match_overlap=0.45,        # 联合候选与 text mask 达到该值才用于补全
                max_union_growth=1.40,     # 补全后面积最多增长 40%，防止异常膨胀
                allow_new_instances=False, # 当前 text 最理想：禁止 box 单独新增实例
                run_text=False,
                run_box=False,              # 保留 02_box_prompt 对比输出
                run_point=False,            # 保留 03_point_prompt 对比输出
                run_box_text=True,
            )
        except torch.cuda.OutOfMemoryError:
            # 打印三件套
            print(f"\n OOM at {img_path.name}")
            print(f"  allocated  : {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
            print(f"  reserved   : {torch.cuda.memory_reserved()/1024**3:.2f} GiB")
            print(f"  gap (碎片) : {(torch.cuda.memory_reserved()-torch.cuda.memory_allocated())/1024**3:.2f} GiB")
            print(f"  img size   : {Image.open(img_path).size}")  # 看看是不是大图触发的
            torch.cuda.empty_cache()
            continue