from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def load_mask(path):
    """读取二值 mask，非零像素均视为前景。"""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取 mask: {path}")
    return mask > 0


def semantic_iou(pred, gt):
    """完整语义 mask IoU。"""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return None if union == 0 else float(inter / union)


def split_instances(mask, connectivity=8):
    """通过连通域从二值 mask 中自动提取实例。"""
    n, labels = cv2.connectedComponents(
        mask.astype(np.uint8),
        connectivity=connectivity,
    )
    return [labels == i for i in range(1, n)]


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(inter / union)


def boundary(mask):
    """统一提取 1-pixel 内边界。"""
    eroded = cv2.erode(
        mask.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return mask & ~eroded


def assd(pred, gt):
    """
    ASSD = 0.5 * [mean(P->G) + mean(G->P)]
    距离单位：pixels。
    """
    bp, bg = boundary(pred), boundary(gt)

    if not bp.any():
        return None, "预测实例无法提取有效边界"
    if not bg.any():
        return None, "参考实例无法提取有效边界"

    p = np.argwhere(bp).astype(np.float64)
    g = np.argwhere(bg).astype(np.float64)

    d_pg = cKDTree(g).query(p, k=1)[0].mean()
    d_gp = cKDTree(p).query(g, k=1)[0].mean()

    value = 0.5 * (d_pg + d_gp)

    if not np.isfinite(value):
        return None, "ASSD 为 NaN 或 Inf"

    return float(value), "OK"


def match_instances(pred_instances, gt_instances, min_iou=0.5):

    npred, ngt = len(pred_instances), len(gt_instances)

    if npred == 0 or ngt == 0:
        return [], list(range(npred)), list(range(ngt))

    ious = np.zeros((npred, ngt), dtype=np.float64)

    for i, p in enumerate(pred_instances):
        for j, g in enumerate(gt_instances):
            ious[i, j] = mask_iou(p, g)

    valid = ious >= min_iou

    # 足够大的常数保证“匹配数量”优先于“IoU和”
    bonus = max(npred, ngt) + 1.0
    score = np.where(valid, bonus + ious, 0.0)

    rows, cols = linear_sum_assignment(-score)

    matches = [
        (i, j, float(ious[i, j]))
        for i, j in zip(rows, cols)
        if valid[i, j]
    ]

    matched_pred = {i for i, _, _ in matches}
    matched_gt = {j for _, j, _ in matches}

    unmatched_pred = [i for i in range(npred) if i not in matched_pred]
    unmatched_gt = [j for j in range(ngt) if j not in matched_gt]

    return matches, unmatched_pred, unmatched_gt


def evaluate_masks(pred_path, gt_path, min_match_iou=0.5):
    """自动实例匹配并计算 IoU + ASSD。"""
    pred = load_mask(pred_path)
    gt = load_mask(gt_path)

    if pred.shape != gt.shape:
        print(
            f"[ERROR] 无法评价：mask尺寸不一致，"
            f"pred={pred.shape}, gt={gt.shape}"
        )
        return None

    sem_iou = semantic_iou(pred, gt)

    pred_instances = split_instances(pred)
    gt_instances = split_instances(gt)

    print("========== Input ==========")
    print(f"Prediction instances : {len(pred_instances)}")
    print(f"Reference instances  : {len(gt_instances)}")
    print(
        f"Semantic IoU         : "
        f"{sem_iou:.4f}" if sem_iou is not None
        else "Semantic IoU         : N/A (两张mask均为空)"
    )

    if not pred_instances:
        print("\nASSD = SKIPPED")
        print("Reason: 预测 mask 中没有任何实例")
        return None

    if not gt_instances:
        print("\nASSD = SKIPPED")
        print("Reason: 参考 mask 中没有任何实例")
        return None

    matches, unmatched_pred, unmatched_gt = match_instances(
        pred_instances,
        gt_instances,
        min_iou=min_match_iou,
    )

    assd_values = []
    iou_values = []

    print("\n========== Matched instances ==========")

    if not matches:
        print(
            f"无有效一对一匹配实例 "
            f"(要求 IoU >= {min_match_iou:.2f})"
        )

    for k, (pi, gi, iou) in enumerate(matches, 1):
        value, reason = assd(
            pred_instances[pi],
            gt_instances[gi],
        )

        if value is None:
            print(
                f"[Match {k:02d}] "
                f"P{pi + 1} <-> G{gi + 1} | "
                f"IoU={iou:.4f} | "
                f"ASSD=SKIPPED | Reason={reason}"
            )
            continue

        print(
            f"[Match {k:02d}] "
            f"P{pi + 1} <-> G{gi + 1} | "
            f"IoU={iou:.4f} | "
            f"ASSD={value:.4f} px"
        )

        iou_values.append(iou)
        assd_values.append(value)

    print("\n========== Unmatched instances ==========")

    for i in unmatched_pred:
        best = max(
            (mask_iou(pred_instances[i], g) for g in gt_instances),
            default=0.0,
        )
        print(
            f"P{i + 1}: SKIPPED "
            f"(无满足一对一匹配条件的GT，best IoU={best:.4f})"
        )

    for j in unmatched_gt:
        best = max(
            (mask_iou(p, gt_instances[j]) for p in pred_instances),
            default=0.0,
        )
        print(
            f"G{j + 1}: SKIPPED "
            f"(无满足一对一匹配条件的预测实例，best IoU={best:.4f})"
        )

    print("\n========== Summary ==========")
    print(f"Valid matches       : {len(assd_values)}")
    print(f"Unmatched prediction: {len(unmatched_pred)}")
    print(f"Unmatched reference : {len(unmatched_gt)}")

    if iou_values:
        mean_iou = float(np.mean(iou_values))
        print(f"Mean matched IoU    : {mean_iou:.4f}")
    else:
        mean_iou = None
        print("Mean matched IoU    : N/A")

    if assd_values:
        mean_assd = float(np.mean(assd_values))
        print(f"Mean ASSD           : {mean_assd:.4f} pixels")
    else:
        mean_assd = None
        print("Mean ASSD           : N/A")
        print("Reason              : 没有可用于ASSD统计的有效匹配实例")

    return {
        "semantic_iou": sem_iou,
        "mean_matched_iou": mean_iou,
        "mean_assd": mean_assd,
        "pred_instance_count": len(pred_instances),
        "gt_instance_count": len(gt_instances),
        "valid_match_count": len(assd_values),
        "unmatched_pred_count": len(unmatched_pred),
        "unmatched_gt_count": len(unmatched_gt),
    }



if __name__ == "__main__":
    PRED_MASK = Path("/home/zhaoming/sam3-main/Phoenix_sam3_mask_check_v3/tennis_court/18.49625.105220.png/05_fused_candidates_mask.png")
    GT_MASK = Path("/home/zhaoming/sam3-main/Phoenix_sam3_mask_check_v3/tennis_court/18.49625.105220.png/00_osm_label.png")

    result = evaluate_masks(
        PRED_MASK,
        GT_MASK,
        min_match_iou=0.5,
    )
    