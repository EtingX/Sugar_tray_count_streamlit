# Seedling Counter - Streamlit Web App (Fixed Clear & Folder Upload)
# ---------------------------------------------------
# - 支持 ZIP 文件夹或多张图片上传
# - 画布标注矩形框 + 输入这些框内“总幼苗数”
# - PlantCV 分割，计算“每株平均白像素”，可复用上一次均值
# - 导出 results.csv + 透明背景分割 PNG（ZIP 打包下载）
# - Clear 按钮用 nonce 强制重建画布
# - 重要修复：大图自动缩放，背景以 NumPy RGB 方式喂给画布；坐标还原到原图
# ---------------------------------------------------

import io
import os
import zipfile
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None            # 允许超大图
ImageFile.LOAD_TRUNCATED_IMAGES = True   # 允许“看似截断”的图片

import cv2
from plantcv import plantcv as pcv

# 画布组件：优先修复版，其次原版（两者只装一个即可）
try:
    from streamlit_drawable_c a nvas import st_canvas  # 修复版包也暴露同名模块
except Exception:
    try:
        from streamlit_drawable_canvas_fix import st_canvas  # 少数镜像用这个导入名
    except Exception as e:
        st.error(
            "未找到 drawable-canvas。请在 requirements.txt 中仅保留 "
            "`streamlit-drawable-canvas-fix==0.9.7`。"
        )
        raise

# -----------------------------
# 版本提示（可忽略）
# -----------------------------
def _version_note():
    try:
        import streamlit_drawable_canvas as sdc
        sdc_ver = getattr(sdc, "__version__", "unknown")
    except Exception:
        try:
            import streamlit_drawable_canvas_fix as sdc
            sdc_ver = getattr(sdc, "__version__", "fix-0.9.7")
        except Exception:
            sdc_ver = "unknown"
    st.sidebar.caption(f"Streamlit {st.__version__} | drawable-canvas {sdc_ver}")

# -----------------------------
# 核心函数
# -----------------------------
def segmentation(image_path: str) -> Tuple[int, np.ndarray]:
    """Return (white_pixels_count, clean_mask_0_255)."""
    img, _, _ = pcv.readimage(image_path)
    h = pcv.rgb2gray_hsv(rgb_img=img, channel='h')
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    mask_h = pcv.threshold.binary(gray_img=h, threshold=70, object_type='dark')
    mask_a = pcv.threshold.binary(gray_img=a, threshold=120, object_type='dark')
    mask_combined = pcv.logical_and(mask_h, mask_a)
    clean = pcv.fill(bin_img=mask_combined, size=1000)
    white_pixels = int(np.sum(clean == 255))
    return white_pixels, clean

def box_white_pixels(bin_img: np.ndarray, box: Tuple[int, int, int, int]) -> int:
    h, w = bin_img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return 0
    roi = bin_img[y1:y2, x1:x2]
    return int(np.count_nonzero(roi))

def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr

def create_seg_cutout_rgba(image_path: str, mask: np.ndarray) -> np.ndarray:
    """保留植株 (mask==255)，背景透明，返回 uint8 RGBA。"""
    img, _, _ = pcv.readimage(image_path)
    base = _to_uint8_rgb(img)

    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.dtype != np.uint8:
        m = np.clip(m, 0, 255).astype(np.uint8)

    keep = (m == 255)
    rgb = base.copy()
    rgb[~keep] = 0

    alpha = np.zeros(m.shape, dtype=np.uint8)
    alpha[keep] = 255

    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    return rgba

# -----------------------------
# 数据模型
# -----------------------------
@dataclass
class ImageResult:
    sample_id: str
    seedlings_estimated: int
    avg_white_per_seedling: float
    image_path: str
    seg_cutout_rgba: Optional[np.ndarray]

# -----------------------------
# 工具函数
# -----------------------------
def save_np_rgba_to_png_bytes(rgba: np.ndarray) -> bytes:
    im = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def ensure_tempdir() -> str:
    if "_tmp_root" not in st.session_state:
        st.session_state["_tmp_root"] = tempfile.mkdtemp(prefix="seedling_counter_")
    return st.session_state["_tmp_root"]

def persist_uploaded_file(upload, dst_dir: str) -> str:
    dst_path = os.path.join(dst_dir, upload.name)
    with open(dst_path, "wb") as f:
        f.write(upload.getbuffer())
    return dst_path

def extract_zip_to_dir(zip_bytes: bytes, dst_dir: str) -> List[str]:
    paths: List[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not info.filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                continue
            out_path = os.path.join(dst_dir, os.path.basename(info.filename))
            with zf.open(info) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            paths.append(out_path)
    return sorted(paths)

def list_images_in_dir(d: str) -> List[str]:
    return sorted([
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    ])

# -----------------------------
# 页面搭建
# -----------------------------
st.set_page_config(page_title="Seedling Counter (Web)", layout="wide")
st.title("🌱 Seedling Counter - Web App")
_version_note()

with st.sidebar:
    st.header("1) Upload images")
    upload_mode = st.radio(
        "Choose upload mode",
        ("Upload ZIP folder (recommended)", "Upload individual images"),
    )

    tmp_root = ensure_tempdir()
    image_dir = os.path.join(tmp_root, "images")
    os.makedirs(image_dir, exist_ok=True)

    if upload_mode == "Upload ZIP folder (recommended)":
        z = st.file_uploader("Upload a .zip containing your image folder", type=["zip"], accept_multiple_files=False)
        if st.button("Load ZIP"):
            if z is None:
                st.warning("Please select a .zip file first.")
            else:
                # 清空旧图
                for f in list_images_in_dir(image_dir):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                paths = extract_zip_to_dir(z.getvalue(), image_dir)
                if not paths:
                    st.warning("No images found inside ZIP.")
                else:
                    st.success(f"Loaded {len(paths)} images from ZIP.")
                    st.session_state["images"] = paths
                    st.session_state["idx"] = 0
                    st.session_state["results"] = []
                    st.session_state["prev_avg_wps"] = None
                    st.session_state["canvas_nonce"] = 0
    else:
        imgs = st.file_uploader(
            "Upload images (PNG/JPG/TIF)",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
        )
        if st.button("Load Images"):
            if not imgs:
                st.warning("Please upload at least one image.")
            else:
                for f in list_images_in_dir(image_dir):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                paths = []
                for up in imgs:
                    p = persist_uploaded_file(up, image_dir)
                    paths.append(p)
                st.success(f"Loaded {len(paths)} images.")
                st.session_state["images"] = sorted(paths)
                st.session_state["idx"] = 0
                st.session_state["results"] = []
                st.session_state["prev_avg_wps"] = None
                st.session_state["canvas_nonce"] = 0

    st.divider()
    st.header("2) Export")
    st.caption("After processing images, export CSV and plant-only PNGs.")
    export_clicked = st.button("📦 Export CSV & PNGs (ZIP)")

# 状态
images: List[str] = st.session_state.get("images", [])
idx: int = st.session_state.get("idx", 0)
results: List[ImageResult] = st.session_state.get("results", [])
prev_avg_wps: Optional[float] = st.session_state.get("prev_avg_wps", None)
if "canvas_nonce" not in st.session_state:
    st.session_state["canvas_nonce"] = 0

colL, colR = st.columns([2.5, 1])

with colL:
    st.header("3) Annotate & Compute")
    if not images:
        st.info("Upload images on the left to start.")
    else:
        idx = max(0, min(idx, len(images) - 1))
        curr_path = images[idx]

        # ---------- 背景图：缩放 + NumPy 喂入 ----------
        img_pil = Image.open(curr_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        MAX_W = 1400
        scale = 1.0 if orig_w <= MAX_W else MAX_W / float(orig_w)
        disp_w = int(round(orig_w * scale))
        disp_h = int(round(orig_h * scale))
        disp_pil = img_pil if scale == 1.0 else img_pil.resize((disp_w, disp_h), Image.LANCZOS)
        disp_np = np.array(disp_pil).astype("uint8")  # H×W×3

        st.write(
            f"**Image {idx+1} / {len(images)}:** `{os.path.basename(curr_path)}` "
            f"(display {disp_w}×{disp_h}, original {orig_w}×{orig_h})"
        )

        canvas_key = f"canvas_{idx}_{st.session_state['canvas_nonce']}"
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#ff0000",
            background_image=disp_np,   # 关键：NumPy RGB
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            key=canvas_key,
        )

        # 读取矩形框（还原到原图坐标）
        boxes: List[Tuple[int, int, int, int]] = []
        if canvas_result.json_data is not None:
            inv = (1.0 / scale) if scale != 0 else 1.0
            for obj in canvas_result.json_data.get("objects", []):
                if obj.get("type") == "rect":
                    left = int(round(obj.get("left", 0)))
                    top = int(round(obj.get("top", 0)))
                    width = int(round(obj.get("width", 0)))
                    height = int(round(obj.get("height", 0)))
                    x1 = int(round(left * inv))
                    y1 = int(round(top * inv))
                    x2 = int(round((left + width) * inv))
                    y2 = int(round((top + height) * inv))
                    boxes.append((x1, y1, x2, y2))

with colR:
    st.header("Controls")
    if images:
        seedlings_total = st.number_input("Seedlings (total inside drawn boxes)", min_value=0, step=1, value=0)
        use_prev = st.checkbox("Use previous avg white/seedling", value=(prev_avg_wps is not None))
        st.caption(f"Prev avg: {f'{prev_avg_wps:.2f}' if prev_avg_wps else 'N/A'}")

        compute = st.button("Compute avg from boxes")
        confirm = st.button("✅ Confirm & Next")
        clear_boxes = st.button("Clear boxes on this image")

        if "_curr_avg" not in st.session_state:
            st.session_state["_curr_avg"] = None
        curr_avg = st.session_state.get("_curr_avg")

        if clear_boxes:
            st.session_state['canvas_nonce'] += 1
            st.session_state["_curr_avg"] = None
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

        if compute:
            if not boxes:
                st.warning("Draw at least one box on the image.")
            elif seedlings_total <= 0:
                st.warning("Seedlings must be > 0.")
            else:
                try:
                    white_pixels_total, clean_mask = segmentation(curr_path)
                    per_box_whites = [box_white_pixels(clean_mask, b) for b in boxes]
                    total_white = int(sum(per_box_whites))
                    if total_white <= 0:
                        st.warning("No white pixels inside boxes. Please redraw.")
                    else:
                        curr_avg = float(total_white) / float(seedlings_total)
                        st.session_state["_curr_avg"] = curr_avg
                        st.success(f"Current avg white/seedling = {curr_avg:.2f}")
                except Exception as e:
                    st.error(f"Segmentation failed: {e}")

        if confirm:
            if use_prev:
                if prev_avg_wps is None or prev_avg_wps <= 0:
                    st.warning("No valid previous average available.")
                    st.stop()
                avg_wps = float(prev_avg_wps)
            else:
                if curr_avg is None or curr_avg <= 0:
                    st.warning("Please click 'Compute avg from boxes' first.")
                    st.stop()
                avg_wps = float(curr_avg)

            try:
                white_pixels_total, clean_mask = segmentation(curr_path)
                if clean_mask is None:
                    _, clean_mask = segmentation(curr_path)
                seg_cutout = None
                try:
                    seg_cutout = create_seg_cutout_rgba(curr_path, clean_mask)
                except Exception as e:
                    st.warning(f"Failed to create plant-only PNG: {e}")

                base = os.path.splitext(os.path.basename(curr_path))[0]
                sample_id = base
                est = int(round(white_pixels_total / avg_wps))

                res = ImageResult(sample_id, est, avg_wps, curr_path, seg_cutout)
                results.append(res)
                st.session_state["results"] = results
                st.session_state["prev_avg_wps"] = avg_wps
                st.session_state["_curr_avg"] = None

                new_idx = idx + 1
                if new_idx < len(images):
                    st.session_state["idx"] = new_idx
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                else:
                    st.success("All images have been analyzed.")
            except Exception as e:
                st.error(f"Processing failed: {e}")

# 结果表
st.divider()
st.subheader("Results")
if results:
    import pandas as pd
    df = pd.DataFrame([
        {
            "sample_id": r.sample_id,
            "seedlings_estimated": r.seedlings_estimated,
            "avg_white_per_seedling": r.avg_white_per_seedling,
        }
        for r in results
    ])
    st.dataframe(df, use_container_width=True)
else:
    st.write("No results yet.")

# 导出 ZIP
export_clicked = st.session_state.get("_export_clicked", False) or export_clicked
if export_clicked:
    if not results:
        st.warning("No results to export.")
    else:
        try:
            import csv
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                # CSV
                csv_bytes = io.StringIO()
                writer = csv.writer(csv_bytes)
                writer.writerow(["sample_id", "seedlings_estimated", "avg_white_per_seedling"])
                for r in results:
                    writer.writerow([r.sample_id, r.seedlings_estimated, f"{r.avg_white_per_seedling:.6f}"])
                zf.writestr("results.csv", csv_bytes.getvalue())

                # 透明背景 PNG
                for r in results:
                    if r.seg_cutout_rgba is None:
                        try:
                            _, clean_mask = segmentation(r.image_path)
                            rgba = create_seg_cutout_rgba(r.image_path, clean_mask)
                        except Exception:
                            continue
                    else:
                        rgba = r.seg_cutout_rgba
                    base = os.path.splitext(os.path.basename(r.image_path))[0]
                    png_bytes = save_np_rgba_to_png_bytes(rgba)
                    zf.writestr(f"{base}_seg.png", png_bytes)

            zip_buf.seek(0)
            st.download_button(
                label="Download results.zip",
                data=zip_buf,
                file_name="seedling_results.zip",
                mime="application/zip",
            )
        except Exception as e:
            st.error(f"Export failed: {e}")

# Footer
st.caption("Tip: 用 ZIP 上传整文件夹；先画框并填总幼苗数 → Compute → Confirm & Next。")
