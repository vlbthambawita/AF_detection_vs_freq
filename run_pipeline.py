"""
Tkinter launcher for the ECG AFIB Detection Pipeline.

This GUI launcher is aligned with the updated main.py workflow:

1. Use existing prepared data or prepare new data from raw PTB-XL.
2. Select default, suggested, or custom integer sampling frequencies.
3. Support train.py split modes: auto, kfold, and manual.
4. Support ecg_data_prepare.py split structures: kfold and manual.
5. Run training or test-only evaluation across selected sampling rates.
6. Keep validation/test split detection consistent with main.py.
7. Keep plotting, LIME, Grad-CAM, live logging, and image preview support.

Expected train.py support:
- --split_mode auto
- --split_mode kfold
- --split_mode manual
- --early_stopping_patience

Expected prepared data layout:
- <prepared_root>/<rate>hz/fold_1.pt ... fold_k.pt for k-fold
- <prepared_root>/<rate>hz/train.pt and val.pt for manual split
- Optional test split at either test/test.pt or test.pt
"""

from __future__ import annotations

import os
import sys
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from typing import Iterable
import webbrowser

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageTk = None
    PIL_AVAILABLE = False


APP_TITLE = "ECG AFIB Detection Pipeline"
APP_SUBTITLE = "GUI launcher aligned with main.py: preparation, auto/kfold/manual training, evaluation, plotting, LIME and Grad-CAM"

DEFAULT_RATES = [62, 100, 250, 500]
SUGGESTED_RATES = [50, 62, 75, 100, 125, 150, 200, 250, 300, 400, 500]

MODELS = ["cnn1d", "cnn_lstm"]
TRAIN_SPLIT_MODES = ["auto", "kfold", "manual"]
PREP_SPLIT_MODES = ["kfold", "manual"]
BALANCE_MODES = ["train", "none", "fold", "global"]
TRAIN_BALANCE = ["downsample", "none"]
DEVICES = ["auto", "cuda", "cpu"]
REP_CASES = [
    "",
    "correct_afib",
    "correct_normal",
    "false_positive",
    "false_negative",
    "uncertain",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ----------------------------------------------------------------------
# Shared helpers aligned with main.py
# ----------------------------------------------------------------------

def norm_path(value: str | Path) -> Path:
    """Normalize a string/path from GUI fields."""
    return Path(str(value).strip().strip('"')).expanduser()


def default_project_root() -> Path:
    """
    Detect project root.

    Same idea as main.py:
    - if this launcher is inside src/, project root is its parent
    - otherwise, project root is the launcher folder
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent

    if base.name == "src":
        return base.parent

    return base


def find_existing_file(candidates: Iterable[Path]) -> Path | None:
    """Return the first existing file from candidates."""
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p.resolve()
        except OSError:
            continue

    return None


def parse_custom_frequencies(text: str) -> list[int]:
    """
    Parse custom integer sampling frequencies.

    Identical rule as main.py:
    - integer values only
    - positive values only
    - values above 500 Hz are rejected because PTB-XL source is 500 Hz
    """
    cleaned = text.replace(",", " ").replace(";", " ")
    parts = [p.strip() for p in cleaned.split() if p.strip()]

    if not parts:
        return DEFAULT_RATES.copy()

    selected: list[int] = []

    for part in parts:
        if "." in part:
            raise ValueError(
                f"Float frequency '{part}' is not supported. "
                "Use an integer value such as 62 instead of 62.5."
            )

        try:
            fs = int(part)
        except ValueError:
            raise ValueError(
                f"Invalid frequency '{part}'. Frequencies must be integers."
            )

        if fs <= 0:
            raise ValueError(
                f"Invalid frequency '{fs}'. Frequency must be greater than 0 Hz."
            )

        if fs > 500:
            raise ValueError(
                f"Invalid frequency '{fs}'. Values above 500 Hz are not allowed "
                "because they would upsample instead of downsample."
            )

        if fs not in selected:
            selected.append(fs)

    return sorted(selected)


def rate_dir_from_prepared_root(prepared_root: Path, rate: int) -> Path:
    """
    Resolve a rate directory without assuming a fixed dataset name.

    Supported inputs:
    - prepared_root = .../<dataset>             -> returns .../<dataset>/<rate>hz
    - prepared_root = .../<dataset>/<rate>hz    -> returns prepared_root directly
    """
    prepared_root = prepared_root.expanduser().resolve()
    rate_name = f"{rate}hz"

    if prepared_root.name.lower() == rate_name.lower():
        return prepared_root

    return prepared_root / rate_name


def detect_prepared_split_mode(rate_path: Path, requested_mode: str, kfolds: int) -> str | None:
    """
    Detect prepared data type for one frequency folder.

    Returns:
    - kfold if fold_1.pt ... fold_k.pt exist
    - manual if train.pt and val.pt exist
    - None if not valid for the requested mode
    """
    fold_files = [rate_path / f"fold_{i}.pt" for i in range(1, kfolds + 1)]
    has_all_folds = all(p.exists() for p in fold_files)
    has_manual = (rate_path / "train.pt").exists() and (rate_path / "val.pt").exists()

    if requested_mode == "kfold":
        return "kfold" if has_all_folds else None

    if requested_mode == "manual":
        return "manual" if has_manual else None

    if has_all_folds:
        return "kfold"

    if has_manual:
        return "manual"

    return None


def test_file_status(rate_path: Path) -> str:
    """Check whether a test file exists inside the prepared frequency folder."""
    if (rate_path / "test" / "test.pt").exists():
        return "test/test.pt"

    if (rate_path / "test.pt").exists():
        return "test.pt"

    return "not found"


def validate_prepared_rate(
    prepared_root: Path,
    rate: int,
    split_mode: str,
    kfolds: int,
) -> tuple[bool, str]:
    """Validate that one selected frequency has a usable prepared data structure."""
    rate_path = rate_dir_from_prepared_root(prepared_root, rate)

    if not rate_path.exists():
        return False, f"missing rate folder: {rate_path}"

    detected = detect_prepared_split_mode(rate_path, split_mode, kfolds)

    if detected is None:
        expected_folds = ", ".join(f"fold_{i}.pt" for i in range(1, kfolds + 1))
        return (
            False,
            "split structure not valid. Expected "
            f"{expected_folds} for kfold or train.pt + val.pt for manual."
        )

    test_status = test_file_status(rate_path)
    return True, f"{detected} split detected | test: {test_status} | {rate_path}"


def validate_ptbxl_path(raw_data_path: Path) -> tuple[bool, str]:
    """Validate that the given directory looks like a PTB-XL dataset folder."""
    raw_data_path = raw_data_path.expanduser().resolve()

    if not raw_data_path.exists():
        return False, f"Directory does not exist: {raw_data_path}"

    db_path = raw_data_path / "ptbxl_database.csv"

    if not db_path.exists():
        return False, f"ptbxl_database.csv not found in: {raw_data_path}"

    return True, f"Found PTB-XL database: {db_path}"


# ----------------------------------------------------------------------
# Command runner
# ----------------------------------------------------------------------

class CommandRunner:
    """Run one or many commands in a worker thread and stream output to the GUI."""

    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.process: subprocess.Popen | None = None
        self.worker: threading.Thread | None = None
        self.q: queue.Queue[str] = queue.Queue()
        self.stop_requested = False

    def running(self) -> bool:
        if self.process is not None and self.process.poll() is None:
            return True

        if self.worker is not None and self.worker.is_alive():
            return True

        return False

    def run(self, cmd: list[str], cwd: Path, title: str = "COMMAND") -> None:
        self.run_many([(title, cmd)], cwd=cwd)

    def run_many(self, commands: list[tuple[str, list[str]]], cwd: Path) -> None:
        if self.running():
            messagebox.showwarning(
                "Command already running",
                "Please stop or wait for the current command to finish.",
            )
            return

        if not commands:
            self.q.put("[No command to run]")
            return

        cwd = cwd.resolve()
        self.stop_requested = False

        def target() -> None:
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"

                total = len(commands)

                for index, (title, cmd) in enumerate(commands, start=1):
                    if self.stop_requested:
                        self.q.put("\n[Command queue stopped]\n")
                        break

                    self.q.put("\n" + "=" * 110)
                    self.q.put(f"[{datetime.now().strftime('%H:%M:%S')}] {title} ({index}/{total})")
                    self.q.put("Command:")
                    self.q.put(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
                    self.q.put(f"Working directory: {cwd}")
                    self.q.put("=" * 110 + "\n")

                    self.process = subprocess.Popen(
                        cmd,
                        cwd=str(cwd),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                        env=env,
                    )

                    assert self.process.stdout is not None

                    for line in iter(self.process.stdout.readline, ""):
                        if line:
                            self.q.put(line.rstrip("\n"))

                        if self.stop_requested:
                            try:
                                self.process.terminate()
                            except Exception:
                                pass

                    code = self.process.wait()
                    self.q.put(f"\n[Finished with exit code {code}] {title}\n")

                    self.process = None

            except Exception as exc:
                self.q.put(f"\n[ERROR] {exc}\n")

            finally:
                self.process = None
                self.stop_requested = False
                self.q.put("[QUEUE DONE]")

        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_requested = True

        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.log_callback("\n[Stop requested]\n")
            except Exception as exc:
                self.log_callback(f"\n[Stop failed] {exc}\n")


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------

class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1520x920")
        self.minsize(1220, 780)

        self.colors = {
            "bg": "#f4f7fb",
            "panel": "#ffffff",
            "header": "#1f3a5f",
            "header_sub": "#d9e8ff",
            "border": "#b8c7d9",
            "text": "#17202a",
            "muted": "#5d6d7e",
            "primary": "#245c9c",
            "primary_hover": "#1d4f87",
            "success": "#2e7d32",
            "danger": "#b71c1c",
            "log_bg": "#0f1720",
            "log_fg": "#e8f0f7",
        }

        root = default_project_root()

        # Project and data paths
        self.project_root = tk.StringVar(value=str(root))
        self.raw_data_path = tk.StringVar(value=str(root / "data"))
        self.dataset_name = tk.StringVar(value="ptbl-xl")
        self.out_root = tk.StringVar(value=str(root / "prepared_data"))
        self.prepared_root = tk.StringVar(value=str(root / "prepared_data" / "ptbl-xl"))
        self.checkpoint_root = tk.StringVar(value=str(root / "checkpoints"))
        self.output_dir = tk.StringVar(value="")

        # Main workflow
        self.rates_text = tk.StringVar(value="62 100 250 500")
        self.train_split_mode = tk.StringVar(value="auto")
        self.prepare_split_mode = tk.StringVar(value="kfold")
        self.balance_mode = tk.StringVar(value="train")
        self.train_balance = tk.StringVar(value="downsample")

        # Preparation settings
        self.kfolds = tk.StringVar(value="5")
        self.holdout_test_ratio = tk.StringVar(value="0.2")
        self.manual_train_ratio = tk.StringVar(value="0.7")
        self.manual_val_ratio = tk.StringVar(value="0.2")
        self.manual_test_ratio = tk.StringVar(value="0.1")
        self.create_extra_holdout = tk.BooleanVar(value=False)
        self.extra_holdout_ratio = tk.StringVar(value="0.2")
        self.flatline_seconds = tk.StringVar(value="3.0")

        # Training settings
        self.model = tk.StringVar(value="cnn1d")
        self.device = tk.StringVar(value="auto")
        self.batch_size = tk.StringVar(value="8")
        self.epochs = tk.StringVar(value="50")
        self.lr = tk.StringVar(value="0.001")
        self.early_stopping_patience = tk.StringVar(value="10")

        # Evaluation and plotting
        self.threshold = tk.StringVar(value="0.5")
        self.bins = tk.StringVar(value="10")

        # Explainability
        self.sample_idx = tk.StringVar(value="0")
        self.window_sec = tk.StringVar(value="0.5")
        self.num_perturbations = tk.StringVar(value="512")
        self.uncertainty_margin = tk.StringVar(value="0.10")
        self.representative_case = tk.StringVar(value="")

        self.preview_image_tk = None
        self.preview_image_path: Path | None = None
        self.preview_images: list[Path] = []
        self.preview_index = -1

        self.configure(bg=self.colors["bg"])
        self._setup_style()

        self.runner = CommandRunner(self._log)

        self._build_ui()
        self.after(100, self._poll_runner)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_style(self) -> None:
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        c = self.colors

        style.configure(
            ".",
            font=("Segoe UI", 10),
            background=c["bg"],
            foreground=c["text"],
        )
        style.configure("Main.TFrame", background=c["bg"])
        style.configure("Panel.TFrame", background=c["panel"])
        style.configure("Header.TFrame", background=c["header"])

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 20, "bold"),
            background=c["header"],
            foreground="white",
        )
        style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI", 10),
            background=c["header"],
            foreground=c["header_sub"],
        )

        style.configure(
            "Section.TLabelframe",
            background=c["panel"],
            bordercolor=c["border"],
            relief="solid",
        )
        style.configure(
            "Section.TLabelframe.Label",
            font=("Segoe UI", 10, "bold"),
            background=c["bg"],
            foreground=c["primary"],
        )

        style.configure("TNotebook", background=c["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(14, 7),
            background="#dfe6ef",
            foreground=c["text"],
            font=("Segoe UI", 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", c["panel"])],
            foreground=[("selected", c["primary"])],
        )

        style.configure(
            "TEntry",
            fieldbackground="white",
            bordercolor=c["border"],
            lightcolor=c["border"],
            darkcolor=c["border"],
        )
        style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
            bordercolor=c["border"],
        )

        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=8,
            foreground="white",
            background=c["primary"],
        )
        style.map(
            "Primary.TButton",
            background=[("active", c["primary_hover"])],
            foreground=[("active", "white")],
        )

        style.configure(
            "Success.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=8,
            foreground="white",
            background=c["success"],
        )
        style.configure(
            "Danger.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=7,
            foreground="white",
            background=c["danger"],
        )
        style.configure("TButton", padding=6)

    def _build_ui(self) -> None:
        c = self.colors

        header = ttk.Frame(
            self,
            style="Header.TFrame",
            padding=(22, 18, 22, 14),
        )
        header.pack(fill="x")

        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text=APP_SUBTITLE, style="Subtitle.TLabel").pack(
            anchor="w",
            pady=(3, 0),
        )

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=10)

        left = ttk.Frame(body, padding=8, style="Main.TFrame")
        right = ttk.Frame(body, padding=8, style="Main.TFrame")

        body.add(left, weight=3)
        body.add(right, weight=2)

        self.tabs = ttk.Notebook(left)
        self.tabs.pack(fill="both", expand=True)

        self._tab_project()
        self._tab_prepare()
        self._tab_train()
        self._tab_eval_plot()
        self._tab_xai()

        right_pane = ttk.PanedWindow(right, orient="vertical")
        right_pane.pack(fill="both", expand=True)

        log_container = ttk.Frame(right_pane, style="Main.TFrame")
        preview_container = ttk.Frame(right_pane, style="Main.TFrame")

        right_pane.add(log_container, weight=3)
        right_pane.add(preview_container, weight=2)

        log_frame = ttk.LabelFrame(
            log_container,
            text="Command log",
            style="Section.TLabelframe",
            padding=8,
        )
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            font=("Consolas", 9),
            height=18,
            bg=c["log_bg"],
            fg=c["log_fg"],
            insertbackground="white",
            relief="flat",
            padx=10,
            pady=8,
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

        self.log_text.tag_config("ok", foreground="#4caf50")
        self.log_text.tag_config("missing", foreground="#ef5350")
        self.log_text.tag_config("header", foreground="#64b5f6")
        self.log_text.tag_config("normal", foreground=self.colors["log_fg"])

        actions = ttk.Frame(
            log_container,
            padding=(0, 8, 0, 0),
            style="Main.TFrame",
        )
        actions.pack(fill="x")

        ttk.Button(
            actions,
            text="Stop running command",
            style="Danger.TButton",
            command=self.runner.stop,
        ).pack(side="left")

        ttk.Button(
            actions,
            text="Clear log",
            command=lambda: self.log_text.delete("1.0", "end"),
        ).pack(side="left", padx=8)

        ttk.Button(
            actions,
            text="Open project folder",
            command=self._open_project_folder,
        ).pack(side="right")

        preview_frame = ttk.LabelFrame(
            preview_container,
            text="Plot preview",
            style="Section.TLabelframe",
            padding=8,
        )
        preview_frame.pack(fill="both", expand=True)

        preview_toolbar = ttk.Frame(preview_frame, style="Panel.TFrame")
        preview_toolbar.pack(fill="x", pady=(0, 6))

        ttk.Button(
            preview_toolbar,
            text="Refresh latest plot",
            command=self.refresh_latest_plot,
        ).pack(side="left")

        ttk.Button(
            preview_toolbar,
            text="Previous plot",
            command=self.previous_plot,
        ).pack(side="left", padx=6)

        ttk.Button(
            preview_toolbar,
            text="Next plot",
            command=self.next_plot,
        ).pack(side="left", padx=6)

        ttk.Button(
            preview_toolbar,
            text="Open preview image",
            command=self.open_preview_image,
        ).pack(side="left", padx=6)

        ttk.Button(
            preview_toolbar,
            text="Open figures folder",
            command=self.open_figures_folder,
        ).pack(side="left", padx=6)

        self.preview_status = ttk.Label(
            preview_toolbar,
            text="No plot loaded yet.",
            foreground=c["muted"],
        )
        self.preview_status.pack(side="right")

        self.preview_canvas = tk.Label(
            preview_frame,
            text=(
                "Generated plots will appear here.\n"
                "Run a plotting, LIME, or Grad-CAM command, then refresh the latest image."
            ),
            bg="white",
            fg=c["muted"],
            anchor="center",
            justify="center",
            relief="solid",
            bd=1,
        )
        self.preview_canvas.pack(fill="both", expand=True)

        self._log("Ready. Set the project root first, validate paths, then choose an action.")

    # ------------------------------------------------------------------
    # Reusable UI helpers
    # ------------------------------------------------------------------

    def _row(
        self,
        parent,
        label: str,
        var: tk.StringVar,
        browse=None,
        width: int = 60,
    ):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=4)

        ttk.Label(
            row,
            text=label,
            width=25,
            background=self.colors["panel"],
        ).pack(side="left")

        ent = ttk.Entry(row, textvariable=var, width=width)
        ent.pack(side="left", fill="x", expand=True)

        if browse:
            ttk.Button(row, text="Browse", command=browse).pack(
                side="left",
                padx=(6, 0),
            )

        return ent

    def _combo_row(
        self,
        parent,
        label: str,
        var: tk.StringVar,
        values: list[str],
    ):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=4)

        ttk.Label(
            row,
            text=label,
            width=25,
            background=self.colors["panel"],
        ).pack(side="left")

        cb = ttk.Combobox(
            row,
            textvariable=var,
            values=values,
            state="readonly",
            width=24,
        )
        cb.pack(side="left")

        return cb

    def _check_row(self, parent, label: str, var: tk.BooleanVar):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=4)

        ttk.Label(
            row,
            text="",
            width=25,
            background=self.colors["panel"],
        ).pack(side="left")

        chk = ttk.Checkbutton(row, text=label, variable=var)
        chk.pack(side="left")

        return chk

    def _rates_box(self, parent) -> None:
        box = ttk.LabelFrame(
            parent,
            text="Sampling frequency selection",
            style="Section.TLabelframe",
            padding=10,
        )
        box.pack(fill="x", pady=8)

        self._row(box, "Sampling rates", self.rates_text, width=70)

        button_row = ttk.Frame(box, style="Panel.TFrame")
        button_row.pack(fill="x", pady=(4, 0))

        ttk.Label(
            button_row,
            text="",
            width=25,
            background=self.colors["panel"],
        ).pack(side="left")

        ttk.Button(
            button_row,
            text="Use default thesis rates",
            command=lambda: self.rates_text.set(" ".join(str(x) for x in DEFAULT_RATES)),
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            button_row,
            text="Use suggested list",
            command=lambda: self.rates_text.set(" ".join(str(x) for x in SUGGESTED_RATES)),
        ).pack(side="left", padx=6)

        ttk.Button(
            button_row,
            text="Validate rates",
            command=self.validate_rates,
        ).pack(side="left", padx=6)

        ttk.Label(
            box,
            text="Write integer frequencies separated by spaces or commas. Values above 500 Hz and float values are rejected.",
            justify="left",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
        ).pack(anchor="w", pady=(7, 0))

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _tab_project(self) -> None:
        tab = ttk.Frame(self.tabs, padding=14, style="Panel.TFrame")
        self.tabs.add(tab, text="Project")

        ttk.Label(
            tab,
            text="Project paths",
            font=("Segoe UI", 12, "bold"),
            background=self.colors["panel"],
            foreground=self.colors["primary"],
        ).pack(anchor="w", pady=(0, 8))

        self._row(
            tab,
            "Project root",
            self.project_root,
            lambda: self._browse_dir(self.project_root),
        )
        self._row(
            tab,
            "Raw PTB-XL data",
            self.raw_data_path,
            lambda: self._browse_dir(self.raw_data_path),
        )
        self._row(tab, "Dataset output name", self.dataset_name)
        self._row(
            tab,
            "Preparation output root",
            self.out_root,
            lambda: self._browse_dir(self.out_root),
        )
        self._row(
            tab,
            "Prepared dataset root",
            self.prepared_root,
            lambda: self._browse_dir(self.prepared_root),
        )
        self._row(
            tab,
            "Checkpoint root",
            self.checkpoint_root,
            lambda: self._browse_dir(self.checkpoint_root),
        )

        ttk.Button(
            tab,
            text="Validate selected paths",
            style="Primary.TButton",
            command=self.validate_paths,
        ).pack(anchor="w", pady=(8, 0))

        link_box = ttk.LabelFrame(
            tab,
            text="Dataset access",
            style="Section.TLabelframe",
            padding=12,
        )
        link_box.pack(anchor="w", fill="x", pady=(28, 0))

        ttk.Label(
            link_box,
            text="PTB-XL can be downloaded from PhysioNet:",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
        ).pack(anchor="w")

        link = tk.Label(
            link_box,
            text="Open PTB-XL Dataset Page",
            fg="#1565c0",
            bg=self.colors["panel"],
            cursor="hand2",
            font=("Segoe UI", 10, "bold underline"),
        )
        link.pack(anchor="w", pady=(6, 0))
        link.bind(
            "<Button-1>",
            lambda _e: webbrowser.open("https://physionet.org/content/ptb-xl/1.0.3/"),
        )

    def _tab_prepare(self) -> None:
        tab = ttk.Frame(self.tabs, padding=14, style="Panel.TFrame")
        self.tabs.add(tab, text="Prepare Data")

        self._rates_box(tab)

        prep = ttk.LabelFrame(
            tab,
            text="Data preparation settings",
            style="Section.TLabelframe",
            padding=10,
        )
        prep.pack(fill="x", pady=10)

        self._combo_row(prep, "Prepared split structure", self.prepare_split_mode, PREP_SPLIT_MODES)
        self._combo_row(prep, "Preprocessing balance mode", self.balance_mode, BALANCE_MODES)
        self._row(prep, "K-folds", self.kfolds, width=18)
        self._row(prep, "K-fold hold-out ratio", self.holdout_test_ratio, width=18)

        ratio_box = ttk.LabelFrame(
            tab,
            text="Manual split settings",
            style="Section.TLabelframe",
            padding=10,
        )
        ratio_box.pack(fill="x", pady=10)

        self._row(ratio_box, "Manual train ratio", self.manual_train_ratio, width=18)
        self._row(ratio_box, "Manual validation ratio", self.manual_val_ratio, width=18)
        self._row(ratio_box, "Manual test ratio", self.manual_test_ratio, width=18)
        self._check_row(
            ratio_box,
            "Create additional patient-safe hold-out test/test.pt for manual split",
            self.create_extra_holdout,
        )
        self._row(ratio_box, "Additional hold-out ratio", self.extra_holdout_ratio, width=18)

        qc_box = ttk.LabelFrame(
            tab,
            text="Signal quality settings",
            style="Section.TLabelframe",
            padding=10,
        )
        qc_box.pack(fill="x", pady=10)

        self._row(qc_box, "Flatline seconds", self.flatline_seconds, width=18)

        ttk.Button(
            tab,
            text="Run data preparation for selected rates",
            style="Primary.TButton",
            command=self.run_prepare,
        ).pack(anchor="e", pady=(8, 0))

        note = (
            "Use kfold preparation with train split mode auto/kfold.\n"
            "Use manual preparation with train split mode auto/manual.\n"
            "The output folder becomes <Preparation output root>/<Dataset output name>/<rate>hz."
        )
        ttk.Label(
            tab,
            text=note,
            justify="left",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
        ).pack(anchor="w", pady=12)

    def _tab_train(self) -> None:
        tab = ttk.Frame(self.tabs, padding=14, style="Panel.TFrame")
        self.tabs.add(tab, text="Train / Test")

        self._rates_box(tab)

        split = ttk.LabelFrame(
            tab,
            text="Split mode",
            style="Section.TLabelframe",
            padding=10,
        )
        split.pack(fill="x", pady=10)

        self._combo_row(split, "Train split mode", self.train_split_mode, TRAIN_SPLIT_MODES)
        self._row(split, "K-folds if needed", self.kfolds, width=18)

        train = ttk.LabelFrame(
            tab,
            text="Training settings",
            style="Section.TLabelframe",
            padding=10,
        )
        train.pack(fill="x", pady=10)

        self._combo_row(train, "Model", self.model, MODELS)
        self._combo_row(train, "Device", self.device, DEVICES)
        self._combo_row(train, "Runtime train balance", self.train_balance, TRAIN_BALANCE)
        self._row(train, "Batch size", self.batch_size, width=18)
        self._row(train, "Maximum epochs", self.epochs, width=18)
        self._row(train, "Learning rate", self.lr, width=18)
        self._row(train, "Early stopping patience", self.early_stopping_patience, width=18)

        action_box = ttk.LabelFrame(
            tab,
            text="Actions",
            style="Section.TLabelframe",
            padding=10,
        )
        action_box.pack(fill="x", pady=10)

        ttk.Button(
            action_box,
            text="Train and run final test evaluation",
            style="Success.TButton",
            command=self.run_training,
        ).pack(side="left", padx=4, pady=4)

        ttk.Button(
            action_box,
            text="Test only using existing checkpoints",
            style="Primary.TButton",
            command=self.run_test_only,
        ).pack(side="left", padx=4, pady=4)

        ttk.Button(
            action_box,
            text="Validate prepared data",
            command=self.validate_prepared_data,
        ).pack(side="left", padx=4, pady=4)

        note = (
            "Hardware guidance: batch size 8 for CNN1D, 4 or 8 for CNN-LSTM. "
            "Run 500 Hz separately on limited laptops."
        )
        ttk.Label(
            tab,
            text=note,
            justify="left",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
        ).pack(anchor="w", pady=10)

    def _tab_eval_plot(self) -> None:
        tab = ttk.Frame(self.tabs, padding=14, style="Panel.TFrame")
        self.tabs.add(tab, text="Evaluate & Plot")

        self._rates_box(tab)
        self._combo_row(tab, "Model", self.model, MODELS)
        self._combo_row(tab, "Device", self.device, DEVICES)
        self._combo_row(tab, "Train split mode", self.train_split_mode, TRAIN_SPLIT_MODES)
        self._row(tab, "Batch size", self.batch_size, width=18)
        self._row(tab, "K-folds", self.kfolds, width=18)
        self._row(tab, "Threshold", self.threshold, width=18)
        self._row(tab, "ECE bins", self.bins, width=18)

        eval_box = ttk.LabelFrame(
            tab,
            text="Evaluation",
            style="Section.TLabelframe",
            padding=10,
        )
        eval_box.pack(fill="x", pady=10)

        ttk.Button(
            eval_box,
            text="Run test-only evaluation",
            style="Primary.TButton",
            command=self.run_test_only,
        ).pack(side="left", padx=4, pady=4)

        ttk.Button(
            eval_box,
            text="Generate test metrics CSV",
            command=self.run_test_metrics_summary,
        ).pack(side="left", padx=4, pady=4)

        ttk.Button(
            eval_box,
            text="Generate validation metrics CSV",
            command=self.run_validation_metrics_summary,
        ).pack(side="left", padx=4, pady=4)

        plot_box = ttk.LabelFrame(
            tab,
            text="Plots",
            style="Section.TLabelframe",
            padding=10,
        )
        plot_box.pack(fill="x", pady=10)

        ttk.Button(
            plot_box,
            text="ROC + PR grid",
            command=self.run_roc_pr,
        ).pack(side="left", padx=4, pady=4)

        ttk.Button(
            plot_box,
            text="Confusion matrix",
            command=self.run_confusion,
        ).pack(side="left", padx=4, pady=4)

        ttk.Button(
            plot_box,
            text="Calibration curves",
            command=self.run_calibration,
        ).pack(side="left", padx=4, pady=4)

    def _tab_xai(self) -> None:
        tab = ttk.Frame(self.tabs, padding=14, style="Panel.TFrame")
        self.tabs.add(tab, text="Explainability")

        self._rates_box(tab)
        self._combo_row(tab, "Model", self.model, MODELS)
        self._combo_row(tab, "Device", self.device, DEVICES)
        self._combo_row(tab, "Train split mode", self.train_split_mode, TRAIN_SPLIT_MODES)

        self._row(tab, "Sample index", self.sample_idx, width=18)
        self._combo_row(tab, "Representative case", self.representative_case, REP_CASES)
        self._row(tab, "Window seconds", self.window_sec, width=18)
        self._row(tab, "Perturbations", self.num_perturbations, width=18)
        self._row(tab, "Uncertainty margin", self.uncertainty_margin, width=18)
        self._row(
            tab,
            "Output directory",
            self.output_dir,
            lambda: self._browse_dir(self.output_dir),
        )

        box = ttk.LabelFrame(
            tab,
            text="Post-hoc explanation",
            style="Section.TLabelframe",
            padding=10,
        )
        box.pack(fill="x", pady=10)

        ttk.Button(
            box,
            text="Run LIME explanation",
            style="Primary.TButton",
            command=self.run_lime,
        ).pack(side="left", padx=4, pady=4)

        ttk.Button(
            box,
            text="Run Grad-CAM explanation",
            style="Primary.TButton",
            command=self.run_gradcam,
        ).pack(side="left", padx=4, pady=4)

        ttk.Button(
            box,
            text="Open LIME results folder",
            command=self.open_lime_results,
        ).pack(side="left", padx=4, pady=4)

        note = (
            "For LIME and Grad-CAM, choose exactly one sampling frequency.\n"
            "If Representative case is empty, Sample index is used manually.\n"
            "If Representative case is selected, the script chooses a matching test sample when supported.\n"
            "LIME expects test/test.pt in the selected prepared rate folder."
        )
        ttk.Label(
            tab,
            text=note,
            justify="left",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
        ).pack(anchor="w", pady=12)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _browse_dir(self, var: tk.StringVar) -> None:
        initial = var.get().strip() or str(self._project_root())
        path = filedialog.askdirectory(initialdir=initial)

        if path:
            var.set(path)

    def _project_root(self) -> Path:
        return norm_path(self.project_root.get()).resolve()

    def _py_base(self) -> str:
        root = self._project_root()

        candidates: list[Path] = []

        if sys.platform.startswith("win"):
            candidates += [
                root / "venv" / "Scripts" / "python.exe",
                root / ".venv" / "Scripts" / "python.exe",
            ]
        else:
            candidates += [
                root / "venv" / "bin" / "python",
                root / ".venv" / "bin" / "python",
            ]

        for p in candidates:
            if p.exists():
                return str(p)

        return sys.executable

    def _py_cmd(self) -> list[str]:
        return [self._py_base(), "-u"]

    def _selected_rates(self) -> list[int]:
        try:
            rates = parse_custom_frequencies(self.rates_text.get())
        except ValueError as exc:
            messagebox.showerror("Invalid sampling rates", str(exc))
            self._log(f"[RATE ERROR] {exc}", "missing")
            return []

        if not rates:
            messagebox.showwarning(
                "No sampling rate selected",
                "Select at least one sampling rate.",
            )

        return rates

    def _script(self, *parts: str) -> Path:
        return self._project_root().joinpath(*parts)

    def _find_script(self, candidates: Iterable[str]) -> Path | None:
        root = self._project_root()
        paths = [root / c for c in candidates]
        script = find_existing_file(paths)

        if script:
            return script

        messagebox.showerror(
            "Missing script",
            "Could not find any of:\n" + "\n".join(str(p) for p in paths),
        )
        return None

    def _find_existing_script_silent(self, candidates: Iterable[str]) -> Path | None:
        root = self._project_root()
        paths = [root / c for c in candidates]
        return find_existing_file(paths)

    def _find_train_script(self) -> Path | None:
        return self._find_script(
            [
                "src/train.py",
                "train.py",
                "old_AF/train.py",
            ]
        )

    def _find_prepare_script(self) -> Path | None:
        return self._find_script(
            [
                "src/ecg_preprocessing/ecg_data_prepare.py",
                "ecg_preprocessing/ecg_data_prepare.py",
                "ecg_data_prepare.py",
                "old_AF/ecg_data_prepare.py",
            ]
        )

    def _run_script_candidates(self, candidates: list[str], title: str = "SCRIPT") -> None:
        script = self._find_script(candidates)

        if script:
            self._run([*self._py_cmd(), str(script)], title=title)

    def _run(self, cmd: list[str], title: str = "COMMAND") -> None:
        self.runner.run(cmd, cwd=self._project_root(), title=title)

    def _run_many(self, commands: list[tuple[str, list[str]]]) -> None:
        self.runner.run_many(commands, cwd=self._project_root())

    def _resolve_prepared_root(self, rates: list[int] | None = None) -> Path:
        """
        Resolve prepared root like main.py, but with GUI convenience.

        Accepts:
        - direct dataset folder: prepared_data/ptbl-xl
        - direct rate folder: prepared_data/ptbl-xl/100hz
        - parent folder: prepared_data, using Dataset output name if available
        """
        raw = norm_path(self.prepared_root.get()).resolve()
        rates = rates or self._selected_rates()

        if not rates:
            return raw

        # Direct dataset folder or direct rate folder.
        if any(rate_dir_from_prepared_root(raw, r).exists() for r in rates):
            return raw

        # Parent prepared_data folder + dataset_name.
        ds_name = self.dataset_name.get().strip()
        if ds_name:
            candidate = raw / ds_name
            if any(rate_dir_from_prepared_root(candidate, r).exists() for r in rates):
                return candidate

        # If only one child dataset folder contains selected rates, use it.
        try:
            if raw.exists() and raw.is_dir():
                candidates = sorted(
                    p for p in raw.iterdir()
                    if p.is_dir() and any(rate_dir_from_prepared_root(p, r).exists() for r in rates)
                )

                if len(candidates) == 1:
                    self._log(f"[Prepared root resolved] {raw} -> {candidates[0]}", "ok")
                    return candidates[0]

                if len(candidates) > 1:
                    self._log(
                        "[Prepared root warning] Multiple dataset folders found. "
                        f"Using first sorted candidate: {candidates[0]}",
                        "header",
                    )
                    return candidates[0]
        except OSError:
            pass

        return raw

    def _prepared_rate_dir(self, rate: int) -> Path:
        return rate_dir_from_prepared_root(self._resolve_prepared_root([rate]), rate)

    def _checkpoint_model_dir(self, rate: int) -> Path:
        """
        Checkpoint structure follows train.py:
        checkpoints/<dataset_name>/<rate>hz/<model>/...
        where dataset_name = rate_path.parent.name.
        """
        prepared_rate = self._prepared_rate_dir(rate)
        dataset_name = prepared_rate.parent.name
        ckpt_root = norm_path(self.checkpoint_root.get()).resolve()

        if ckpt_root.name == dataset_name:
            return ckpt_root / f"{rate}hz" / self.model.get()

        if (ckpt_root / dataset_name).exists() or ckpt_root.name == "checkpoints":
            return ckpt_root / dataset_name / f"{rate}hz" / self.model.get()

        return ckpt_root / dataset_name / f"{rate}hz" / self.model.get()

    # ------------------------------------------------------------------
    # Image preview
    # ------------------------------------------------------------------

    def _candidate_image_roots(self) -> list[Path]:
        root = self._project_root()

        folders = [
            root / "outputs",
            root / "figures",
            root / "checkpoints",
            root / "lime_results",
            root / "explainable",
            root / "src" / "plotting",
        ]

        ckpt_root = norm_path(self.checkpoint_root.get())
        if ckpt_root.exists():
            folders.insert(0, ckpt_root)

        if self.output_dir.get().strip():
            folders.insert(0, norm_path(self.output_dir.get().strip()))

        return folders

    def _all_image_files(self) -> list[Path]:
        images: list[Path] = []

        for folder in self._candidate_image_roots():
            if not folder.exists():
                continue

            for ext in IMAGE_EXTENSIONS:
                images.extend(folder.rglob(f"*{ext}"))

        valid_images = []
        for p in images:
            try:
                if p.exists() and p.is_file():
                    valid_images.append(p)
            except OSError:
                continue

        return sorted(valid_images, key=lambda p: p.stat().st_mtime)

    def refresh_latest_plot(self) -> None:
        self.preview_images = self._all_image_files()

        if not self.preview_images:
            self.preview_status.configure(text="No PNG/JPG plot found.")
            self.preview_canvas.configure(
                image="",
                text="No generated image found yet.\nRun a plotting, LIME, or Grad-CAM command first.",
            )
            self.preview_image_tk = None
            self.preview_image_path = None
            self.preview_index = -1
            return

        self.preview_index = len(self.preview_images) - 1
        self._show_image(self.preview_images[self.preview_index])

    def previous_plot(self) -> None:
        if not self.preview_images:
            self.refresh_latest_plot()
            return

        self.preview_index = max(0, self.preview_index - 1)
        self._show_image(self.preview_images[self.preview_index])

    def next_plot(self) -> None:
        if not self.preview_images:
            self.refresh_latest_plot()
            return

        self.preview_index = min(len(self.preview_images) - 1, self.preview_index + 1)
        self._show_image(self.preview_images[self.preview_index])

    def _show_image(self, image_path: Path) -> None:
        if not PIL_AVAILABLE:
            self.preview_status.configure(text="Pillow is not installed.")
            self.preview_canvas.configure(
                image="",
                text="Install Pillow to preview plots:\n\npip install pillow",
            )
            return

        try:
            image_path = image_path.resolve()
            img = Image.open(image_path)

            self.preview_canvas.update_idletasks()

            max_w = max(self.preview_canvas.winfo_width() - 20, 300)
            max_h = max(self.preview_canvas.winfo_height() - 20, 220)

            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            self.preview_image_tk = ImageTk.PhotoImage(img)
            self.preview_image_path = image_path

            self.preview_canvas.configure(image=self.preview_image_tk, text="")

            if self.preview_images and image_path in self.preview_images:
                idx = self.preview_images.index(image_path) + 1
                total = len(self.preview_images)
                self.preview_status.configure(text=f"{idx}/{total}  {image_path.name}")
            else:
                self.preview_status.configure(text=image_path.name)

        except Exception as exc:
            self.preview_status.configure(text="Preview failed.")
            self.preview_canvas.configure(
                image="",
                text=f"Could not load image:\n{image_path}\n\n{exc}",
            )
            self.preview_image_tk = None
            self.preview_image_path = None

    def open_preview_image(self) -> None:
        if self.preview_image_path and self.preview_image_path.exists():
            self._open_path(self.preview_image_path)
            return

        self.refresh_latest_plot()

        if self.preview_image_path and self.preview_image_path.exists():
            self._open_path(self.preview_image_path)

    def open_figures_folder(self) -> None:
        figures = self._project_root() / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        self._open_path(figures)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _get_int(self, var: tk.StringVar, name: str) -> int | None:
        try:
            return int(var.get())
        except ValueError:
            messagebox.showerror(f"Invalid {name}", f"{name} must be an integer.")
            return None

    def _get_float(self, var: tk.StringVar, name: str) -> float | None:
        try:
            return float(var.get())
        except ValueError:
            messagebox.showerror(f"Invalid {name}", f"{name} must be a number.")
            return None

    def validate_rates(self) -> None:
        rates = self._selected_rates()
        if rates:
            self._log(f"Selected sampling rates: {rates}", "ok")
            messagebox.showinfo("Sampling rates", f"Valid sampling rates:\n{rates}")

    def validate_paths(self) -> None:
        root = self._project_root()
        rates = self._selected_rates()

        self._log("\nPath validation", "header")
        self._log("-" * 80, "header")

        self._log_status_line(f"Project root: {root}", root.exists())
        self._log(f"Python: {self._py_base()}")

        train_script = find_existing_file(
            [
                root / "src" / "train.py",
                root / "train.py",
                root / "old_AF" / "train.py",
            ]
        )
        prepare_script = find_existing_file(
            [
                root / "src" / "ecg_preprocessing" / "ecg_data_prepare.py",
                root / "ecg_preprocessing" / "ecg_data_prepare.py",
                root / "ecg_data_prepare.py",
                root / "old_AF" / "ecg_data_prepare.py",
            ]
        )
        lime_script = self._find_existing_script_silent(
            [
                "explainable/explain_lime.py",
                "src/explainable/explain_lime.py",
                "explain_lime.py",
                "src/explain_lime.py",
            ]
        )
        gradcam_script = self._find_existing_script_silent(
            [
                "explainable/explain_gradcam.py",
                "src/explainable/explain_gradcam.py",
                "explain_gradcam.py",
            ]
        )

        self._log_status_line(
            f"Train script: {train_script if train_script else 'not found'}",
            train_script is not None,
        )
        self._log_status_line(
            f"Prepare script: {prepare_script if prepare_script else 'not found'}",
            prepare_script is not None,
        )
        self._log_status_line(
            f"LIME script: {lime_script if lime_script else 'not found'}",
            lime_script is not None,
        )
        self._log_status_line(
            f"Grad-CAM script: {gradcam_script if gradcam_script else 'not found'}",
            gradcam_script is not None,
        )

        raw_ok, raw_msg = validate_ptbxl_path(norm_path(self.raw_data_path.get()))
        self._log_status_line(raw_msg, raw_ok)

        prepared_root = self._resolve_prepared_root(rates)
        self._log_status_line(f"Prepared root resolved: {prepared_root}", prepared_root.exists())

        try:
            kfolds = int(self.kfolds.get())
        except ValueError:
            kfolds = 0

        for rate in rates:
            data_dir = rate_dir_from_prepared_root(prepared_root, rate)
            self._log_status_line(f"Prepared {rate} Hz: {data_dir}", data_dir.exists())

            if data_dir.exists():
                detected = detect_prepared_split_mode(data_dir, self.train_split_mode.get(), kfolds)
                self._log_status_line(f"  detected split: {detected}", detected is not None)

                test_status = test_file_status(data_dir)
                self._log_status_line(f"  test: {test_status}", test_status != "not found")

                for i in range(1, kfolds + 1):
                    p = data_dir / f"fold_{i}.pt"
                    self._log_status_line(f"  {p.name}:", p.exists())

                self._log_status_line("  train.pt:", (data_dir / "train.pt").exists())
                self._log_status_line("  val.pt:", (data_dir / "val.pt").exists())

            ckpt_dir = self._checkpoint_model_dir(rate)
            self._log_status_line(
                f"Checkpoint model dir {rate} Hz: {ckpt_dir}",
                ckpt_dir.exists(),
            )

        self._log("-" * 80 + "\n", "header")

    def validate_prepared_data(self) -> None:
        rates = self._selected_rates()
        kfolds = self._get_int(self.kfolds, "K-folds")

        if not rates or kfolds is None:
            return

        prepared_root = self._resolve_prepared_root(rates)

        self._log("\nPrepared data validation", "header")
        self._log("-" * 80, "header")

        missing: list[int] = []

        for rate in rates:
            ok, msg = validate_prepared_rate(
                prepared_root=prepared_root,
                rate=rate,
                split_mode=self.train_split_mode.get(),
                kfolds=kfolds,
            )
            self._log_status_line(f"{rate} Hz: {msg}", ok)
            if not ok:
                missing.append(rate)

        self._log("-" * 80 + "\n", "header")

        if missing:
            messagebox.showwarning(
                "Prepared data incomplete",
                f"Missing or invalid prepared data for rates:\n{missing}",
            )
        else:
            messagebox.showinfo("Prepared data", "Prepared data is valid for selected rates.")

    def _validate_prepare_train_mode_pair(self) -> bool:
        prep_mode = self.prepare_split_mode.get()
        train_mode = self.train_split_mode.get()

        if train_mode == "auto":
            return True

        if prep_mode == "kfold" and train_mode == "manual":
            messagebox.showerror(
                "Invalid mode combination",
                "You selected manual training but k-fold preparation.",
            )
            return False

        if prep_mode == "manual" and train_mode == "kfold":
            messagebox.showerror(
                "Invalid mode combination",
                "You selected k-fold training but manual preparation.",
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Pipeline actions
    # ------------------------------------------------------------------

    def run_prepare(self) -> None:
        rates = self._selected_rates()

        if not rates:
            return

        if not self._validate_prepare_train_mode_pair():
            return

        script = self._find_prepare_script()
        if not script:
            return

        raw_data = norm_path(self.raw_data_path.get())
        raw_ok, raw_msg = validate_ptbxl_path(raw_data)
        if not raw_ok:
            messagebox.showerror("Invalid PTB-XL path", raw_msg)
            self._log(raw_msg, "missing")
            return

        dataset_name = self.dataset_name.get().strip()
        if not dataset_name:
            messagebox.showerror("Missing dataset name", "Dataset output name cannot be empty.")
            return

        out_root = norm_path(self.out_root.get())
        prep_mode = self.prepare_split_mode.get()

        kfolds = self._get_int(self.kfolds, "K-folds")
        flatline_seconds = self._get_float(self.flatline_seconds, "Flatline seconds")

        if kfolds is None or flatline_seconds is None:
            return

        commands: list[tuple[str, list[str]]] = []

        for rate in rates:
            cmd = [
                *self._py_cmd(),
                str(script),
                "--dataset_path", str(raw_data),
                "--name", dataset_name,
                "--fs", str(rate),
                "--out_root", str(out_root),
                "--balance_mode", self.balance_mode.get(),
                "--flatline_seconds", str(flatline_seconds),
            ]

            if prep_mode == "kfold":
                holdout = self._get_float(self.holdout_test_ratio, "K-fold hold-out ratio")
                if holdout is None:
                    return

                cmd += ["--folds", str(kfolds)]
                cmd += ["--test_ratio", str(holdout)]

            elif prep_mode == "manual":
                tr = self._get_float(self.manual_train_ratio, "Manual train ratio")
                va = self._get_float(self.manual_val_ratio, "Manual validation ratio")
                te = self._get_float(self.manual_test_ratio, "Manual test ratio")

                if tr is None or va is None or te is None:
                    return

                total = tr + va + te
                if abs(total - 1.0) > 1e-6:
                    self._log(
                        f"[WARNING] Manual split ratios sum to {total:.4f}. "
                        "They will be passed as entered; recommended sum is 1.0.",
                        "header",
                    )

                cmd += ["--split_ratio", str(tr), str(va), str(te)]

                if self.create_extra_holdout.get():
                    extra = self._get_float(self.extra_holdout_ratio, "Additional hold-out ratio")
                    if extra is None:
                        return
                    cmd += ["--test_ratio", str(extra)]

            else:
                messagebox.showerror("Invalid split mode", f"Unknown preparation split mode: {prep_mode}")
                return

            commands.append((f"DATA PREPARATION: {rate} Hz | {prep_mode}", cmd))

        self.prepared_root.set(str(out_root / dataset_name))
        self._log(f"Prepared root set to: {out_root / dataset_name}", "ok")
        self._run_many(commands)

    def _training_commands(self, test_only: bool) -> list[tuple[str, list[str]]] | None:
        rates = self._selected_rates()

        if not rates:
            return None

        script = self._find_train_script()
        if not script:
            return None

        kfolds = self._get_int(self.kfolds, "K-folds")
        batch_size = self._get_int(self.batch_size, "Batch size")
        epochs = self._get_int(self.epochs, "Maximum epochs")
        lr = self._get_float(self.lr, "Learning rate")
        patience = self._get_int(self.early_stopping_patience, "Early stopping patience")

        if None in (kfolds, batch_size, epochs, lr, patience):
            return None

        assert kfolds is not None
        assert batch_size is not None
        assert epochs is not None
        assert lr is not None
        assert patience is not None

        prepared_root = self._resolve_prepared_root(rates)
        split_mode = self.train_split_mode.get()
        commands: list[tuple[str, list[str]]] = []
        skipped: list[str] = []

        for rate in rates:
            rate_path = rate_dir_from_prepared_root(prepared_root, rate)

            if not rate_path.exists():
                skipped.append(f"{rate} Hz: missing {rate_path}")
                continue

            ok, msg = validate_prepared_rate(
                prepared_root=prepared_root,
                rate=rate,
                split_mode=split_mode,
                kfolds=kfolds,
            )

            if not ok:
                skipped.append(f"{rate} Hz: {msg}")
                continue

            self._log(f"[Detected] {rate} Hz: {msg}", "ok")

            cmd = [
                *self._py_cmd(),
                str(script),
                "--data_path", str(rate_path),
                "--model", self.model.get(),
                "--split_mode", split_mode,
                "--train_balance", self.train_balance.get(),
                "--batch_size", str(batch_size),
                "--epochs", str(epochs if not test_only else 1),
                "--lr", str(lr),
                "--kfolds", str(kfolds),
                "--early_stopping_patience", str(patience),
                "--device", self.device.get(),
            ]

            if test_only:
                cmd.append("--test_only")

            title = f"{'TEST ONLY' if test_only else 'TRAINING'}: {rate} Hz | {self.model.get()} | split_mode={split_mode}"
            commands.append((title, cmd))

        if skipped:
            self._log("\nSkipped rates:", "header")
            for item in skipped:
                self._log(f"  - {item}", "missing")

        if not commands:
            messagebox.showerror(
                "No valid prepared data",
                "No selected sampling rate has a valid prepared data structure.",
            )
            return None

        return commands

    def run_training(self) -> None:
        rates = self._selected_rates()

        if not rates:
            return

        model = self.model.get()
        msg = (
            "Training deep ECG models can be computationally intensive.\n\n"
            "Recommended:\n"
            "- CUDA-enabled GPU\n"
            "- Batch size 8 for CNN1D\n"
            "- Batch size 4 or 8 for CNN-LSTM\n"
            "- Run 500 Hz separately on limited laptops\n\n"
            f"Selected rates: {rates}\n"
            f"Model: {model}\n"
            f"Split mode: {self.train_split_mode.get()}\n\n"
            "Continue?"
        )

        if not messagebox.askyesno("Hardware requirement warning", msg):
            self._log("[Training cancelled by user]")
            return

        commands = self._training_commands(test_only=False)
        if commands:
            self._run_many(commands)

    def run_test_only(self) -> None:
        commands = self._training_commands(test_only=True)
        if commands:
            self._run_many(commands)

    def run_test_metrics_summary(self) -> None:
        script = self._find_script(
            ["src/plotting/compute_auroc_ece_test_ensambling_metric.py"]
        )

        if script:
            cmd = [
                *self._py_cmd(),
                str(script),
                "--root",
                str(self._metrics_root()),
                "--bins",
                self.bins.get(),
                "--threshold",
                self.threshold.get(),
            ]
            self._run(cmd, title="GENERATE TEST METRICS CSV")

    def run_validation_metrics_summary(self) -> None:
        script = self._find_script(["src/plotting/summary_sensitivity_auroc_gen.py"])

        if script:
            cmd = [
                *self._py_cmd(),
                str(script),
                "--root",
                str(self._metrics_root()),
                "--bins",
                self.bins.get(),
            ]
            self._run(cmd, title="GENERATE VALIDATION METRICS CSV")

    def run_roc_pr(self) -> None:
        self._run_script_candidates(
            ["src/plotting/roc_pr_ploting_2x2.py"],
            title="ROC + PR GRID",
        )

    def run_confusion(self) -> None:
        script = self._find_script(["src/plotting/confiousion_matrix_2x2.py"])

        if not script:
            return

        rates = self._selected_rates()
        if not rates:
            return

        freqs = [f"{r}hz" for r in rates]

        cmd = [
            *self._py_cmd(),
            str(script),
            "--root",
            str(self._metrics_root()),
            "--freqs",
            *freqs,
            "--models",
            self.model.get(),
            "--folds",
            self.kfolds.get(),
            "--threshold",
            self.threshold.get(),
        ]

        self._run(cmd, title="CONFUSION MATRIX")

    def run_calibration(self) -> None:
        self._run_script_candidates(
            ["src/plotting/probability_calibiration_curv_2.py"],
            title="CALIBRATION CURVES",
        )

    def _metrics_root(self) -> Path:
        """
        Plot scripts in the original launcher expected checkpoint_root to point
        either to checkpoints/<dataset> or checkpoints. This resolves to
        checkpoints/<dataset> when possible.
        """
        ckpt_root = norm_path(self.checkpoint_root.get()).resolve()
        rates = self._selected_rates()
        prepared_root = self._resolve_prepared_root(rates)

        if rates:
            rate_path = rate_dir_from_prepared_root(prepared_root, rates[0])
            dataset_name = rate_path.parent.name
        else:
            dataset_name = self.dataset_name.get().strip()

        if ckpt_root.name == dataset_name:
            return ckpt_root

        if (ckpt_root / dataset_name).exists():
            return ckpt_root / dataset_name

        return ckpt_root / dataset_name if ckpt_root.name == "checkpoints" else ckpt_root

    def run_lime(self) -> None:
        rates = self._selected_rates()

        if len(rates) != 1:
            messagebox.showwarning(
                "Choose one frequency",
                "LIME needs exactly one sampling frequency.",
            )
            return

        script = self._find_script(
            [
                "explainable/explain_lime.py",
                "src/explainable/explain_lime.py",
                "explain_lime.py",
                "src/explain_lime.py",
            ]
        )

        if not script:
            return

        rate = rates[0]
        data_path = self._prepared_rate_dir(rate)

        if not data_path.exists():
            messagebox.showerror(
                "Missing prepared data",
                f"Prepared data folder was not found:\n{data_path}",
            )
            return

        test_file = data_path / "test" / "test.pt"
        alt_test_file = data_path / "test.pt"

        if not test_file.exists() and not alt_test_file.exists():
            messagebox.showerror(
                "Missing test split",
                f"LIME expects a test split here:\n{test_file}\nor here:\n{alt_test_file}",
            )
            return

        rep = self.representative_case.get().strip()

        cmd = [
            *self._py_cmd(),
            str(script),
            "--data_path",
            str(data_path),
            "--model",
            self.model.get(),
            "--split",
            "test",
            "--folds",
            self.kfolds.get(),
            "--device",
            self.device.get(),
            "--window_sec",
            self.window_sec.get(),
            "--num_perturbations",
            self.num_perturbations.get(),
            "--uncertainty_margin",
            self.uncertainty_margin.get(),
            "--mask_strategy",
            "local_mean",
            "--explain_class",
            "pred",
        ]

        if self.output_dir.get().strip():
            cmd += ["--output_dir", self.output_dir.get().strip()]

        if rep:
            cmd += ["--representative_case", rep]
            self._log(
                f"LIME representative selection enabled: {rep}. "
                "The selected real sample_idx should be printed and written on the plots when supported."
            )
        else:
            cmd += ["--sample_idx", self.sample_idx.get()]
            self._log(
                f"LIME manual sample selection enabled: sample_idx={self.sample_idx.get()}."
            )

        self._run(cmd, title=f"LIME EXPLANATION: {rate} Hz | {self.model.get()}")

    def run_gradcam(self) -> None:
        rates = self._selected_rates()

        if len(rates) != 1:
            messagebox.showwarning(
                "Choose one frequency",
                "Grad-CAM needs exactly one sampling frequency.",
            )
            return

        script = self._find_script(
            [
                "explainable/explain_gradcam.py",
                "src/explainable/explain_gradcam.py",
                "explain_gradcam.py",
            ]
        )

        if not script:
            return

        rate = rates[0]
        rep = self.representative_case.get().strip()

        rep_map = {
            "correct_afib": "correct_afib",
            "correct_normal": "correct_normal",
            "uncertain": "borderline",
        }

        cmd = [
            *self._py_cmd(),
            str(script),
            "--frequency", str(rate),
            "--model_type", self.model.get(),
            "--device", self.device.get(),
        ]

        if self.output_dir.get().strip():
            cmd += ["--output_dir", self.output_dir.get().strip()]

        if rep in rep_map:
            cmd += ["--representative_case", rep_map[rep]]
        else:
            val = self.sample_idx.get().strip()
            if val:
                cmd += ["--sample_idx", val]

        self._run(cmd, title=f"GRAD-CAM EXPLANATION: {rate} Hz | {self.model.get()}")

    def open_lime_results(self) -> None:
        if self.output_dir.get().strip():
            p = norm_path(self.output_dir.get().strip())
        else:
            p = self._project_root() / "lime_results"

        p.mkdir(parents=True, exist_ok=True)
        self._open_path(p)

    # ------------------------------------------------------------------
    # OS helpers and logging
    # ------------------------------------------------------------------

    def _open_project_folder(self) -> None:
        self._open_path(self._project_root())

    def _open_path(self, path: Path) -> None:
        path = path.resolve()

        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Open failed", f"Could not open:\n{path}\n\n{exc}")

    def _log(self, text: str, tag: str | None = None) -> None:
        try:
            self.log_text.insert("end", text + "\n", tag)
            self.log_text.see("end")
            self.update_idletasks()
        except Exception:
            pass

    def _log_status_line(self, text: str, ok: bool) -> None:
        mark = "[OK]" if ok else "[MISSING]"
        tag = "ok" if ok else "missing"
        self._log(f"{mark} {text}", tag)

    def _poll_runner(self) -> None:
        finished = False

        try:
            while True:
                line = self.runner.q.get_nowait()

                if line == "[QUEUE DONE]":
                    finished = True
                else:
                    self._log(line)

        except queue.Empty:
            pass

        if finished:
            self.after(500, self.refresh_latest_plot)

        self.after(80, self._poll_runner)


if __name__ == "__main__":
    app = PipelineGUI()
    app.mainloop()
