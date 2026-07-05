from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from camo_engine import DEFAULT_OPTIONS, PALETTES, PATTERNS, render_camo

RGB = Tuple[int, int, int]


def pil_to_pixmap(img: Image.Image) -> QPixmap:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


class SliderRow(QWidget):
    def __init__(self, name: str, minimum: int, maximum: int, value: int, suffix: str = ""):
        super().__init__()
        self.name = name
        self.suffix = suffix
        self.label = QLabel(name)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(48)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.slider, 3)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._update_label)
        self._update_label(self.slider.value())

    def _update_label(self, value: int):
        self.value_label.setText(f"{value}{self.suffix}")

    def value(self) -> int:
        return int(self.slider.value())

    def setValue(self, value: int):
        self.slider.setValue(int(value))


class ColorRow(QGroupBox):
    def __init__(self, index: int, color: RGB, enabled: bool = True):
        super().__init__(f"Color {index + 1}")
        self._updating = False
        self.enabled_box = QCheckBox("Use")
        self.enabled_box.setChecked(enabled)
        self.swatch = QPushButton("Pick")
        self.swatch.setMinimumWidth(70)
        self.r = QSlider(Qt.Orientation.Horizontal)
        self.g = QSlider(Qt.Orientation.Horizontal)
        self.b = QSlider(Qt.Orientation.Horizontal)
        self.h = QSlider(Qt.Orientation.Horizontal)
        self.s = QSlider(Qt.Orientation.Horizontal)
        self.v = QSlider(Qt.Orientation.Horizontal)
        for slider in (self.r, self.g, self.b):
            slider.setRange(0, 255)
        self.h.setRange(0, 359)
        self.s.setRange(0, 100)
        self.v.setRange(0, 100)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(92)

        grid = QGridLayout(self)
        grid.addWidget(self.enabled_box, 0, 0)
        grid.addWidget(self.swatch, 0, 1)
        grid.addWidget(self.value_label, 0, 2, 1, 2)
        labels = ["R", "G", "B", "H", "S", "V"]
        sliders = [self.r, self.g, self.b, self.h, self.s, self.v]
        for row, (label, slider) in enumerate(zip(labels, sliders), start=1):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(slider, row, 1, 1, 3)

        self.swatch.clicked.connect(self.pick_color)
        for slider in (self.r, self.g, self.b):
            slider.valueChanged.connect(self._rgb_changed)
        for slider in (self.h, self.s, self.v):
            slider.valueChanged.connect(self._hsv_changed)
        self.set_color(color)

    def set_color(self, color: RGB):
        r, g, b = [int(max(0, min(255, c))) for c in color]
        self._updating = True
        self.r.setValue(r)
        self.g.setValue(g)
        self.b.setValue(b)
        qc = QColor(r, g, b)
        h = qc.hue() if qc.hue() >= 0 else 0
        self.h.setValue(h)
        self.s.setValue(qc.saturation() * 100 // 255)
        self.v.setValue(qc.value() * 100 // 255)
        self._updating = False
        self._paint_swatch()

    def color(self) -> RGB:
        return (self.r.value(), self.g.value(), self.b.value())

    def is_enabled(self) -> bool:
        return self.enabled_box.isChecked()

    def _rgb_changed(self):
        if self._updating:
            return
        self._updating = True
        qc = QColor(self.r.value(), self.g.value(), self.b.value())
        h = qc.hue() if qc.hue() >= 0 else 0
        self.h.setValue(h)
        self.s.setValue(qc.saturation() * 100 // 255)
        self.v.setValue(qc.value() * 100 // 255)
        self._updating = False
        self._paint_swatch()

    def _hsv_changed(self):
        if self._updating:
            return
        self._updating = True
        qc = QColor.fromHsv(self.h.value(), self.s.value() * 255 // 100, self.v.value() * 255 // 100)
        self.r.setValue(qc.red())
        self.g.setValue(qc.green())
        self.b.setValue(qc.blue())
        self._updating = False
        self._paint_swatch()

    def _paint_swatch(self):
        r, g, b = self.color()
        self.value_label.setText(f"#{r:02X}{g:02X}{b:02X}")
        self.swatch.setStyleSheet(f"QPushButton {{ background-color: rgb({r},{g},{b}); color: {'white' if r+g+b < 370 else 'black'}; }}")

    def pick_color(self):
        r, g, b = self.color()
        chosen = QColorDialog.getColor(QColor(r, g, b), self, "Pick camo color")
        if chosen.isValid():
            self.set_color((chosen.red(), chosen.green(), chosen.blue()))


class CamoFoundry(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camo Foundry")
        self.resize(1380, 900)
        self.preview_size = 640
        self.current_preview: Image.Image | None = None
        self._building = False

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.render_preview)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        controls = self._build_controls()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        splitter.addWidget(scroll)
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([510, 850])

        self.apply_palette(DEFAULT_OPTIONS["palette"])
        self.schedule_preview()

    def _build_controls(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)

        pattern_box = QGroupBox("Pattern")
        form = QFormLayout(pattern_box)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(PATTERNS)
        self.pattern_combo.setCurrentText(DEFAULT_OPTIONS["pattern"])
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(PALETTES.keys())
        self.palette_combo.setCurrentText(DEFAULT_OPTIONS["palette"])
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(DEFAULT_OPTIONS["seed"])
        self.random_seed_button = QPushButton("Randomize Seed")
        seed_row = QWidget()
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.addWidget(self.seed_spin)
        seed_layout.addWidget(self.random_seed_button)
        form.addRow("Camo type", self.pattern_combo)
        form.addRow("Palette preset", self.palette_combo)
        form.addRow("Seed", seed_row)
        layout.addWidget(pattern_box)

        self.sliders: Dict[str, SliderRow] = {}
        slider_defs = [
            ("scale", "Patch scale", 5, 400, DEFAULT_OPTIONS["scale"], " px"),
            ("detail", "Detail octaves", 1, 8, DEFAULT_OPTIONS["detail"], ""),
            ("density", "Density", 0, 100, DEFAULT_OPTIONS["density"], "%"),
            ("contrast", "Contrast", 0, 100, DEFAULT_OPTIONS["contrast"], "%"),
            ("roughness", "Roughness", 0, 100, DEFAULT_OPTIONS["roughness"], "%"),
            ("blur", "Blur", 0, 30, DEFAULT_OPTIONS["blur"], ""),
            ("edge_softness", "Edge softness", 0, 100, DEFAULT_OPTIONS["edge_softness"], "%"),
            ("stripe_width", "Stripe/stroke width", 2, 260, DEFAULT_OPTIONS["stripe_width"], " px"),
            ("stripe_spacing", "Stripe spacing", 10, 420, DEFAULT_OPTIONS["stripe_spacing"], " px"),
            ("stripe_wiggle", "Stripe wiggle", 0, 300, DEFAULT_OPTIONS["stripe_wiggle"], " px"),
            ("block_size", "Digital block size", 2, 180, DEFAULT_OPTIONS["block_size"], " px"),
            ("dot_size", "Dot/spray size", 1, 90, DEFAULT_OPTIONS["dot_size"], " px"),
            ("speckle", "Speckle blocks", 0, 100, DEFAULT_OPTIONS["speckle"], "%"),
            ("background_noise", "Grit/noise", 0, 100, DEFAULT_OPTIONS["background_noise"], "%"),
            ("hsv_jitter", "Palette HSV jitter", 0, 100, DEFAULT_OPTIONS["hsv_jitter"], "%"),
            ("color_bleed", "Color bleed", 0, 100, DEFAULT_OPTIONS["color_bleed"], "%"),
            ("rotation", "Rotation", -90, 90, DEFAULT_OPTIONS["rotation"], "°"),
        ]
        slider_box = QGroupBox("Shape Options")
        slider_layout = QVBoxLayout(slider_box)
        for key, label, minimum, maximum, value, suffix in slider_defs:
            row = SliderRow(label, minimum, maximum, value, suffix)
            self.sliders[key] = row
            slider_layout.addWidget(row)
            row.slider.valueChanged.connect(self.schedule_preview)
        layout.addWidget(slider_box)

        toggle_box = QGroupBox("Toggles")
        toggle_layout = QVBoxLayout(toggle_box)
        self.seamless_box = QCheckBox("Try to make tile edges match")
        self.invert_box = QCheckBox("Invert color ordering")
        self.shuffle_box = QCheckBox("Shuffle palette per seed")
        self.outline_box = QCheckBox("Add dark contour/outline passes")
        self.live_box = QCheckBox("Live preview")
        self.live_box.setChecked(True)
        self.smooth_box = QCheckBox("Smooth preview scaling")
        self.smooth_box.setChecked(True)
        for box in (self.seamless_box, self.invert_box, self.shuffle_box, self.outline_box, self.live_box, self.smooth_box):
            toggle_layout.addWidget(box)
            box.stateChanged.connect(self.schedule_preview)
        layout.addWidget(toggle_box)

        colors_box = QGroupBox("Colors — RGB + HSV")
        colors_layout = QVBoxLayout(colors_box)
        self.color_rows: List[ColorRow] = []
        default_colors = PALETTES[DEFAULT_OPTIONS["palette"]]
        for i in range(6):
            color = default_colors[i] if i < len(default_colors) else (0, 0, 0)
            row = ColorRow(i, color, enabled=i < len(default_colors))
            row.enabled_box.stateChanged.connect(self.schedule_preview)
            for slider in (row.r, row.g, row.b, row.h, row.s, row.v):
                slider.valueChanged.connect(self.schedule_preview)
            colors_layout.addWidget(row)
            self.color_rows.append(row)
        layout.addWidget(colors_box)

        utility_box = QGroupBox("Settings")
        utility_layout = QVBoxLayout(utility_box)
        self.save_settings_button = QPushButton("Save Settings JSON")
        self.load_settings_button = QPushButton("Load Settings JSON")
        self.reset_button = QPushButton("Reset Defaults")
        utility_layout.addWidget(self.save_settings_button)
        utility_layout.addWidget(self.load_settings_button)
        utility_layout.addWidget(self.reset_button)
        layout.addWidget(utility_box)

        self.pattern_combo.currentTextChanged.connect(self.schedule_preview)
        self.palette_combo.currentTextChanged.connect(self.apply_palette)
        self.seed_spin.valueChanged.connect(self.schedule_preview)
        self.random_seed_button.clicked.connect(self.randomize_seed)
        self.save_settings_button.clicked.connect(self.save_settings)
        self.load_settings_button.clicked.connect(self.load_settings)
        self.reset_button.clicked.connect(self.reset_defaults)
        layout.addStretch(1)
        return root

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(512, 512)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(self.preview_label, 1)

        button_row = QWidget()
        buttons = QHBoxLayout(button_row)
        self.render_button = QPushButton("Render Preview")
        self.export_button = QPushButton("Export 2048×2048 PNG")
        self.export_jpg_button = QPushButton("Export 2048×2048 JPG")
        buttons.addWidget(self.render_button)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.export_jpg_button)
        layout.addWidget(button_row)

        self.status = QLabel("Ready")
        layout.addWidget(self.status)
        self.render_button.clicked.connect(self.render_preview)
        self.export_button.clicked.connect(lambda: self.export_image("PNG"))
        self.export_jpg_button.clicked.connect(lambda: self.export_image("JPEG"))
        return panel

    def schedule_preview(self):
        if self._building:
            return
        if getattr(self, "live_box", None) and self.live_box.isChecked():
            self.timer.start(260)

    def build_options(self) -> Dict:
        opts = dict(DEFAULT_OPTIONS)
        opts["pattern"] = self.pattern_combo.currentText()
        opts["palette"] = self.palette_combo.currentText()
        opts["seed"] = self.seed_spin.value()
        for key, row in self.sliders.items():
            opts[key] = row.value()
        opts["seamless"] = self.seamless_box.isChecked()
        opts["invert"] = self.invert_box.isChecked()
        opts["shuffle_colors"] = self.shuffle_box.isChecked()
        opts["outline"] = self.outline_box.isChecked()
        opts["smooth_preview"] = self.smooth_box.isChecked()
        return opts

    def active_colors(self) -> List[RGB]:
        colors = [row.color() for row in self.color_rows if row.is_enabled()]
        if len(colors) < 2:
            colors = PALETTES[self.palette_combo.currentText()]
        return colors

    def apply_palette(self, palette_name: str):
        if palette_name not in PALETTES:
            return
        self._building = True
        colors = PALETTES[palette_name]
        for i, row in enumerate(self.color_rows):
            row.enabled_box.setChecked(i < len(colors))
            row.set_color(colors[i] if i < len(colors) else (0, 0, 0))
        self._building = False
        self.schedule_preview()

    def render_preview(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.status.setText("Rendering preview…")
            QApplication.processEvents()
            img = render_camo(self.preview_size, self.build_options(), self.active_colors())
            self.current_preview = img
            pix = pil_to_pixmap(img)
            mode = Qt.TransformationMode.SmoothTransformation if self.smooth_box.isChecked() else Qt.TransformationMode.FastTransformation
            pix = pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, mode)
            self.preview_label.setPixmap(pix)
            self.status.setText("Preview rendered. Export creates a full 2048×2048 image.")
        except Exception as exc:  # noqa: BLE001 - UI needs to show unexpected generator errors
            QMessageBox.critical(self, "Render failed", str(exc))
            self.status.setText("Render failed.")
        finally:
            QApplication.restoreOverrideCursor()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_preview is not None:
            pix = pil_to_pixmap(self.current_preview)
            mode = Qt.TransformationMode.SmoothTransformation if self.smooth_box.isChecked() else Qt.TransformationMode.FastTransformation
            self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, mode))

    def export_image(self, fmt: str):
        suffix = "png" if fmt == "PNG" else "jpg"
        filters = "PNG image (*.png)" if fmt == "PNG" else "JPEG image (*.jpg *.jpeg)"
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt}", f"camo_{self.seed_spin.value()}.{suffix}", filters)
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.status.setText(f"Rendering full 2048×2048 {fmt}…")
            QApplication.processEvents()
            img = render_camo(2048, self.build_options(), self.active_colors())
            save_kwargs = {}
            if fmt == "JPEG":
                save_kwargs = {"quality": 95, "subsampling": 0}
                img = img.convert("RGB")
            img.save(path, fmt, **save_kwargs)
            self.status.setText(f"Saved: {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            self.status.setText("Export failed.")
        finally:
            QApplication.restoreOverrideCursor()

    def randomize_seed(self):
        self.seed_spin.setValue(random.randint(0, 2_147_483_647))

    def reset_defaults(self):
        self._building = True
        self.pattern_combo.setCurrentText(DEFAULT_OPTIONS["pattern"])
        self.palette_combo.setCurrentText(DEFAULT_OPTIONS["palette"])
        self.seed_spin.setValue(DEFAULT_OPTIONS["seed"])
        for key, row in self.sliders.items():
            row.setValue(DEFAULT_OPTIONS[key])
        self.seamless_box.setChecked(DEFAULT_OPTIONS["seamless"])
        self.invert_box.setChecked(DEFAULT_OPTIONS["invert"])
        self.shuffle_box.setChecked(DEFAULT_OPTIONS["shuffle_colors"])
        self.outline_box.setChecked(DEFAULT_OPTIONS["outline"])
        self.smooth_box.setChecked(True)
        self._building = False
        self.apply_palette(DEFAULT_OPTIONS["palette"])
        self.schedule_preview()

    def save_settings(self):
        data = {
            "options": self.build_options(),
            "colors": self.active_colors(),
        }
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "camo_settings.json", "JSON settings (*.json)")
        if path:
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.status.setText(f"Settings saved: {path}")

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "JSON settings (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            opts = data.get("options", {})
            colors = data.get("colors", [])
            self._building = True
            if opts.get("pattern") in PATTERNS:
                self.pattern_combo.setCurrentText(opts["pattern"])
            if opts.get("palette") in PALETTES:
                self.palette_combo.setCurrentText(opts["palette"])
            self.seed_spin.setValue(int(opts.get("seed", self.seed_spin.value())))
            for key, row in self.sliders.items():
                if key in opts:
                    row.setValue(int(opts[key]))
            self.seamless_box.setChecked(bool(opts.get("seamless", False)))
            self.invert_box.setChecked(bool(opts.get("invert", False)))
            self.shuffle_box.setChecked(bool(opts.get("shuffle_colors", False)))
            self.outline_box.setChecked(bool(opts.get("outline", False)))
            for i, row in enumerate(self.color_rows):
                row.enabled_box.setChecked(i < len(colors))
                if i < len(colors) and len(colors[i]) == 3:
                    row.set_color(tuple(colors[i]))
            self._building = False
            self.schedule_preview()
            self.status.setText(f"Settings loaded: {path}")
        except Exception as exc:  # noqa: BLE001
            self._building = False
            QMessageBox.critical(self, "Load failed", str(exc))


def main():
    app = QApplication(sys.argv)
    window = CamoFoundry()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
