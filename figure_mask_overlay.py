from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ============================================================
# Configuration
# ============================================================

AREA = "Phoenix"
CATEGORY = "swimming_pool"
SAMPLE_NAME = "18.49464.105140.png"

OHSM_BASE = Path("/home/zhaoming/ohsome2label/ohsome2label-master")
IMAGE_DIR = OHSM_BASE / f"example_result_{AREA}_{CATEGORY}" / "images"
OSM_DIR = OHSM_BASE / f"example_result_{AREA}_{CATEGORY}" / "labels"

SAM3_DIR = Path(
    f"/home/zhaoming/sam3-main/{AREA}_sam3_mask_check_v3/{CATEGORY}"
)

OUTPUT_DIR = Path(
    f"/home/zhaoming/sam3-main/{AREA}_single_sample_visual/{CATEGORY}"
)


# ============================================================
# Visualization style
# ============================================================

FINAL_FILL = "#FF0000"       # mask 填充：纯红
FINAL_EDGE = "#960000"       # mask 轮廓：深红
FINAL_ALPHA = 0.38           # mask 填充透明度
FINAL_EDGE_WIDTH = 3         # mask 轮廓线宽（像素）


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """将 #RRGGBB 转换为 RGB tuple。"""
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"非法颜色值: {color}")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def save_final_overlay(
    image_path: Path,
    final_mask_path: Path,
    output_path: Path,
    fill_color: str = FINAL_FILL,
    edge_color: str = FINAL_EDGE,
    alpha: float = FINAL_ALPHA,
    edge_width: int = FINAL_EDGE_WIDTH,
) -> None:
    """
    将 08_final_mask.png 叠加到原始影像。

    效果：
    1. 最终 mask 区域使用半透明红色填充；
    2. mask 外边界使用深红色加粗描边；
    3. 不绘制 SAM 单实例颜色；
    4. 不绘制 OSM；
    5. 不绘制 bbox、score、prompt point 或五角星；
    6. 输出尺寸与原始影像完全一致。
    """

    if not image_path.exists():
        raise FileNotFoundError(f"原始影像不存在: {image_path}")

    if not final_mask_path.exists():
        raise FileNotFoundError(f"08_final_mask 不存在: {final_mask_path}")

    # --------------------------------------------------------
    # 1. 读取原始影像
    # --------------------------------------------------------
    image = Image.open(image_path).convert("RGB")
    base = np.asarray(image, dtype=np.uint8).copy()

    height, width = base.shape[:2]

    # --------------------------------------------------------
    # 2. 读取最终二值 mask
    # --------------------------------------------------------
    mask_img = Image.open(final_mask_path).convert("L")
    mask = np.asarray(mask_img, dtype=np.uint8)

    # 若尺寸不一致，使用最近邻插值，避免改变 mask 几何边界。
    if mask.shape != (height, width):
        print(
            f"[WARN] mask尺寸 {mask.shape[::-1]} 与影像尺寸 "
            f"{(width, height)} 不一致，自动使用 NEAREST resize。"
        )
        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    # 支持 0/1、0/255 或其他非零前景编码
    mask_bool = mask > 0

    # --------------------------------------------------------
    # 3. 半透明红色填充
    # --------------------------------------------------------
    if mask_bool.any():
        alpha = float(np.clip(alpha, 0.0, 1.0))

        fill_rgb = np.asarray(
            hex_to_rgb(fill_color),
            dtype=np.float32,
        )

        original_pixels = base[mask_bool].astype(np.float32)

        blended_pixels = (
            (1.0 - alpha) * original_pixels
            + alpha * fill_rgb
        )

        base[mask_bool] = np.clip(
            blended_pixels,
            0,
            255,
        ).astype(np.uint8)

        # ----------------------------------------------------
        # 4. 深红色加粗 mask 外轮廓
        # ----------------------------------------------------
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if contours:
            edge_rgb = hex_to_rgb(edge_color)

            cv2.drawContours(
                base,
                contours,
                contourIdx=-1,
                color=edge_rgb,
                thickness=max(1, int(edge_width)),
                lineType=cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # 5. 保存
    # --------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(base).save(output_path)

    print(f"[OK] image : {image_path}")
    print(f"[OK] mask  : {final_mask_path}")
    print(f"[OK] output: {output_path}")
    print(
        f"[STYLE] fill={fill_color}, "
        f"alpha={alpha:.2f}, "
        f"edge={edge_color}, "
        f"edge_width={edge_width}px"
    )


if __name__ == "__main__":

    # 原始遥感影像
    image_path = IMAGE_DIR / SAMPLE_NAME

    # 按现有 SAM3 脚本的输出结构：
    #
    # CATEGORY/
    # └── 18.49566.105044.png/
    #     ├── 08_final_mask.png
    #     ├── 08_final_overlay.png
    #     └── 08_final_instances/
    #
    final_mask_path = (
        SAM3_DIR
        / SAMPLE_NAME
        / "08_final_mask.png"
    )

    # 单独整理到 sample_visual_db
    output_path = (
        OUTPUT_DIR
        / SAMPLE_NAME
        / "08_final_overlay.png"
    )

    save_final_overlay(
        image_path=image_path,
        final_mask_path=final_mask_path,
        output_path=output_path,
    )