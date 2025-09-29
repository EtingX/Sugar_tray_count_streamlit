import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
from plantcv import plantcv as pcv

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


# -----------------------------
# Core functions
# -----------------------------

def segmentation(image_path: str) -> Tuple[int, np.ndarray]:
    """Return (white_pixels_count, clean_mask_0_255)."""
    img, path, filename = pcv.readimage(image_path)
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
    """Ensure RGB uint8 for saving."""
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
    """
    Create an RGBA image: keep only plant (mask==255), background transparent.
    Returns uint8 RGBA array suitable for PNG saving.
    """
    img, _, _ = pcv.readimage(image_path)  # RGB
    base = _to_uint8_rgb(img)

    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.dtype != np.uint8:
        m = np.clip(m, 0, 255).astype(np.uint8)

    keep = (m == 255)

    # Zero background in RGB
    rgb = base.copy()
    rgb[~keep] = 0

    # Alpha 255 for plant, 0 for background
    alpha = np.zeros(m.shape, dtype=np.uint8)
    alpha[keep] = 255

    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    return rgba


# -----------------------------
# GUI Components
# -----------------------------

@dataclass
class ImageResult:
    sample_id: str
    seedlings_estimated: int
    avg_white_per_seedling: float
    image_path: str
    seg_cutout_rgba: Optional[np.ndarray]  # 只保留植株、背景透明的PNG数据（内存中暂存，导出时统一保存）


class ImageCanvas(QWidget):
    """Matplotlib canvas with rectangle drawing support (RGB display)."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.fig = Figure(figsize=(9, 6), tight_layout=True)  # larger canvas
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        # Ensure RGB display for the user view
        from matplotlib import image as mpimg
        self.img_data = mpimg.imread(image_path)
        if self.img_data.ndim == 2:
            self.ax.imshow(self.img_data, cmap='gray')
        else:
            self.ax.imshow(self.img_data)
        self._boxes: List[Tuple[int, int, int, int]] = []
        self.selector: Optional[RectangleSelector] = None

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        self._init_selector()

    def _init_selector(self):
        def onselect(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if None in (x1, y1, x2, y2):
                return
            x1, x2 = sorted([int(round(x1)), int(round(x2))])
            y1, y2 = sorted([int(round(y1)), int(round(y2))])
            self._boxes.append((x1, y1, x2, y2))
            self.ax.add_patch(self._mpl_rect(x1, y1, x2 - x1, y2 - y1))
            self.canvas.draw_idle()

        self.selector = RectangleSelector(
            self.ax,
            onselect,
            button=[1],
            minspanx=2,
            minspany=2,
            useblit=False,
            spancoords='pixels',
            interactive=False,
        )

    @staticmethod
    def _mpl_rect(x, y, w, h):
        import matplotlib.patches as patches
        return patches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)

    def clear_boxes(self):
        """Clear drawn rectangles AND their visuals completely (hard reset of axes)."""
        self._boxes.clear()
        if self.selector is not None:
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.imshow(self.img_data)
        self.canvas.draw_idle()
        self._init_selector()

    @property
    def boxes(self) -> List[Tuple[int, int, int, int]]:
        return list(self._boxes)


class SeedlingDialog(QDialog):
    """Popup dialog for a single image with explicit controls.

    Workflow:
      1) Draw boxes and input total seedlings in those boxes.
      2) Click "Compute avg from boxes" (runs segmentation) to compute avg_white_per_seedling.
      3) Optionally Clear to re-draw / re-enter; or tick "Use previous avg".
      4) Click "Confirm & Next" to finalize this image.
    """

    def __init__(self, image_path: str, prev_avg_wps: Optional[float] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(image_path))
        self.resize(1200, 700)  # dialog size (w, h)
        self.image_path = image_path
        self.prev_avg_wps = prev_avg_wps
        self._result: Optional[ImageResult] = None

        # state caches
        self._avg_wps_manual: Optional[float] = None
        self._white_pixels_total: Optional[int] = None
        self._clean_mask: Optional[np.ndarray] = None

        # Widgets
        self.sample_id_edit = QLineEdit()
        self.sample_id_edit.setPlaceholderText("Sample ID (leave blank to use image name)")

        self.canvas = ImageCanvas(image_path)

        self.seedlings_spin = QSpinBox()
        self.seedlings_spin.setRange(0, 10_000)
        self.seedlings_spin.setValue(0)

        self.use_prev_chk = QCheckBox("Use previous avg white/seedling")
        self.use_prev_chk.setChecked(prev_avg_wps is not None)
        self.use_prev_chk.setEnabled(prev_avg_wps is not None)

        self.prev_value_label = QLabel(
            f"Prev avg: {prev_avg_wps:.2f}" if prev_avg_wps else "Prev avg: N/A"
        )
        self.curr_avg_label = QLabel("Current avg: N/A")

        self.compute_avg_btn = QPushButton("Compute avg from boxes")
        self.clear_btn = QPushButton("Clear")
        self.ok_btn = QPushButton("Confirm & Next")
        self.cancel_btn = QPushButton("Cancel")

        # Layouts
        top = QHBoxLayout()
        top.addWidget(QLabel("Sample ID:"))
        top.addWidget(self.sample_id_edit)

        right = QVBoxLayout()
        right.addWidget(QLabel("Seedlings (total in boxes):"))
        right.addWidget(self.seedlings_spin)
        right.addWidget(self.use_prev_chk)
        right.addWidget(self.prev_value_label)
        right.addWidget(self.curr_avg_label)
        right.addSpacing(8)
        right.addWidget(self.compute_avg_btn)
        right.addWidget(self.clear_btn)
        right.addWidget(self.ok_btn)
        right.addWidget(self.cancel_btn)
        right.addStretch(1)

        center = QHBoxLayout()
        center.addWidget(self.canvas, stretch=4)
        right_box = QGroupBox("Controls")
        rb_lay = QVBoxLayout(right_box)
        rb_lay.addLayout(right)
        center.addWidget(right_box, stretch=1)

        main = QVBoxLayout(self)
        main.addLayout(top)
        main.addLayout(center)

        # Defaults
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        self.sample_id_edit.setText(base_name)

        # Signals
        self.compute_avg_btn.clicked.connect(self._compute_avg_from_boxes)
        self.clear_btn.clicked.connect(self._clear_inputs)
        self.ok_btn.clicked.connect(self._accept)
        self.cancel_btn.clicked.connect(self.reject)

    def _compute_avg_from_boxes(self):
        boxes = self.canvas.boxes
        if not boxes:
            QMessageBox.warning(self, "Error", "Draw at least one box.")
            return
        seedlings = int(self.seedlings_spin.value())
        if seedlings <= 0:
            QMessageBox.warning(self, "Error", "Seedlings must be > 0.")
            return
        white_pixels, clean = segmentation(self.image_path)
        self._white_pixels_total = white_pixels
        self._clean_mask = clean
        per_box_whites = [box_white_pixels(clean, b) for b in boxes]
        total_white = int(sum(per_box_whites))
        if total_white <= 0:
            QMessageBox.warning(self, "Error", "No white pixels inside boxes. Please redraw.")
            return
        self._avg_wps_manual = float(total_white) / float(seedlings)
        self.curr_avg_label.setText(f"Current avg: {self._avg_wps_manual:.2f}")

    def _clear_inputs(self):
        self.canvas.clear_boxes()
        self.seedlings_spin.setValue(0)
        self._avg_wps_manual = None
        self._white_pixels_total = None
        self._clean_mask = None
        self.curr_avg_label.setText("Current avg: N/A")

    def _accept(self):
        # Decide which average to use
        if self.use_prev_chk.isChecked():
            if self.prev_avg_wps is None or self.prev_avg_wps <= 0:
                QMessageBox.warning(self, "Error", "No valid previous average available.")
                return
            avg_wps = float(self.prev_avg_wps)
        else:
            if self._avg_wps_manual is None or self._avg_wps_manual <= 0:
                QMessageBox.warning(self, "Error", "Please click 'Compute avg from boxes' first.")
                return
            avg_wps = float(self._avg_wps_manual)

        # Use cached segmentation if available; otherwise run once now
        if self._white_pixels_total is not None:
            white_pixels = int(self._white_pixels_total)
            clean_mask = self._clean_mask
        else:
            white_pixels, clean_mask = segmentation(self.image_path)

        if clean_mask is None:
            _, clean_mask = segmentation(self.image_path)

        # Build RGBA cutout (do not save now)
        try:
            seg_cutout = create_seg_cutout_rgba(self.image_path, clean_mask)
        except Exception as e:
            seg_cutout = None
            QMessageBox.warning(self, "Warning", f"Failed to create plant-only PNG:\n{e}")

        sample_id = self.sample_id_edit.text().strip() or os.path.splitext(os.path.basename(self.image_path))[0]
        est = int(round(white_pixels / avg_wps))
        self._result = ImageResult(sample_id, est, avg_wps, self.image_path, seg_cutout)

        self.accept()

    def result_payload(self) -> ImageResult:
        return self._result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seedling Counter UI")
        self.resize(1200, 800)

        self.folder: Optional[str] = None
        self.image_paths: List[str] = []
        self.results: List[ImageResult] = []
        self.prev_avg_wps: Optional[float] = None

        self._build_ui()

    def _build_ui(self):
        open_folder_act = QAction("Open Folder", self)
        open_folder_act.triggered.connect(self.choose_folder)
        export_act = QAction("Export CSV & Plant-only PNGs", self)
        export_act.triggered.connect(self.export_csv)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(open_folder_act)
        file_menu.addAction(export_act)

        splitter = QSplitter()

        self.list_widget = QListWidget()
        splitter.addWidget(self.list_widget)

        right_panel = QWidget()
        rlay = QVBoxLayout(right_panel)
        rlay.addWidget(QLabel("Results (processed images)"))
        self.results_list = QListWidget()
        rlay.addWidget(self.results_list)

        self.process_next_btn = QPushButton("Process Selected / Next")
        self.process_next_btn.clicked.connect(self.process_selected)
        rlay.addWidget(self.process_next_btn)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        self.folder = folder
        self.image_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
        ]
        self.list_widget.clear()
        for p in self.image_paths:
            self.list_widget.addItem(QListWidgetItem(os.path.basename(p)))
        self.results.clear()
        self.results_list.clear()

    def process_selected(self):
        if not self.image_paths:
            QMessageBox.information(self, "Info", "Open a folder first.")
            return
        row = self.list_widget.currentRow()
        if row < 0:
            row = 0
        if row >= len(self.image_paths):
            QMessageBox.information(self, "Info", "No more images.")
            return
        image_path = self.image_paths[row]
        dlg = SeedlingDialog(image_path, prev_avg_wps=self.prev_avg_wps, parent=self)
        if dlg.exec() == QDialog.Accepted:
            payload = dlg.result_payload()
            if payload is None:
                return
            self.prev_avg_wps = payload.avg_white_per_seedling
            self.results.append(payload)
            self.results_list.addItem(
                QListWidgetItem(
                    f"{payload.sample_id}: seedlings={payload.seedlings_estimated} | avg={payload.avg_white_per_seedling:.2f}"
                )
            )
            next_row = row + 1
            if next_row < len(self.image_paths):
                self.list_widget.setCurrentRow(next_row)
            if len(self.results) >= len(self.image_paths):
                QMessageBox.information(self, "Completed", "All images have been analyzed.")

    def export_csv(self):
        if not self.results:
            QMessageBox.information(self, "Info", "No results to export.")
            return
        # Select CSV path; all plant-only PNGs will be saved to the SAME directory
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV and plant-only PNGs",
                                              "results.csv", "CSV Files (*.csv)")
        if not path:
            return

        csv_dir = os.path.dirname(path)
        import csv
        try:
            # 1) Save CSV
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "seedlings_estimated", "avg_white_per_seedling"])
                for r in self.results:
                    writer.writerow([r.sample_id, r.seedlings_estimated, f"{r.avg_white_per_seedling:.6f}"])

            # 2) Save ALL plant-only PNGs (RGBA with transparent background) to the same directory
            saved_count = 0
            for r in self.results:
                if r.seg_cutout_rgba is None:
                    continue
                base = os.path.splitext(os.path.basename(r.image_path))[0]
                seg_path = os.path.join(csv_dir, f"{base}_seg.png")  # PNG to keep transparency
                try:
                    # Convert RGBA (RGB order) -> BGRA for OpenCV saving
                    bgr = cv2.cvtColor(r.seg_cutout_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                    a = r.seg_cutout_rgba[:, :, 3]
                    bgra = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], a])
                    cv2.imwrite(seg_path, bgra)
                    saved_count += 1
                except Exception as e:
                    QMessageBox.warning(self, "Warning",
                                        f"Failed to save plant-only PNG for {base}:\n{e}")

            QMessageBox.information(self, "Saved",
                                    f"CSV saved to:\n{path}\n\nPlant-only PNGs saved: {saved_count}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
