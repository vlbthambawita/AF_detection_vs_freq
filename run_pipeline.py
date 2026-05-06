"""
Tkinter launcher for the ECG PTB-XL AFIB Detection Pipeline.

Main features:
- data preparation
- k-fold training
- test-only ensemble evaluation
- metric summary generation
- ROC/PR, confusion matrix, calibration and dataset plots
- LIME explanation launcher
- Grad-CAM notebook launcher
- live command log
- image preview for generated plots
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


APP_TITLE = "ECG PTB-XL AFIB Detection Pipeline"
APP_SUBTITLE = "GUI launcher for training, evaluation, plotting, LIME and Grad-CAM"

FREQS = ["62", "100", "250", "500"]
MODELS = ["cnn1d", "cnn_lstm"]
BALANCE_MODES = ["train", "none", "fold", "global"]
TRAIN_BALANCE = ["downsample", "none"]
DEVICES = ["auto", "cuda", "cpu"]
REP_CASES = ["", "correct_afib", "correct_normal", "false_positive", "false_negative", "uncertain"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def norm_path(value: str | Path) -> Path:
    return Path(str(value).strip().strip('"')).expanduser()


class CommandRunner:
    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.process: subprocess.Popen | None = None
        self.worker: threading.Thread | None = None
        self.q: queue.Queue[str] = queue.Queue()
        

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def run(self, cmd: list[str], cwd: Path) -> None:
        if self.running():
            messagebox.showwarning(
                "Command already running",
                "Please stop or wait for the current command to finish.",
            )
            return

        cwd = cwd.resolve()
        self.log_callback("\n" + "=" * 110)
        self.log_callback(f"[{datetime.now().strftime('%H:%M:%S')}] Running command:")
        self.log_callback(" ".join(str(x) for x in cmd))
        self.log_callback(f"Working directory: {cwd}")
        self.log_callback("=" * 110 + "\n")

        def target():
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"

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

                code = self.process.wait()
                self.q.put(f"\n[Finished with exit code {code}]\n")

            except Exception as exc:
                self.q.put(f"\n[ERROR] {exc}\n")
            finally:
                self.process = None

        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        if self.running() and self.process is not None:
            try:
                self.process.terminate()
                self.log_callback("\n[Stop requested]\n")
            except Exception as exc:
                self.log_callback(f"\n[Stop failed] {exc}\n")


class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1500x900")
        self.minsize(1200, 760)

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

        root = Path.cwd()
        self.project_root = tk.StringVar(value=str(root))
        self.raw_data_path = tk.StringVar(value=str(root / "data"))
        self.prepared_root = tk.StringVar(value=str(root / "prepared_data" / "ptbl-xl"))
        self.dataset_name = tk.StringVar(value="ptbl-xl")
        self.out_root = tk.StringVar(value=str(root / "prepared_data"))
        self.checkpoint_root = tk.StringVar(value=str(root / "checkpoints" / "ptbl-xl"))
        self.output_dir = tk.StringVar(value="")

        self.model = tk.StringVar(value="cnn1d")
        self.device = tk.StringVar(value="auto")
        self.balance_mode = tk.StringVar(value="train")
        self.train_balance = tk.StringVar(value="downsample")
        self.batch_size = tk.StringVar(value="8")
        self.epochs = tk.StringVar(value="2")
        self.lr = tk.StringVar(value="0.001")
        self.kfolds = tk.StringVar(value="5")
        self.test_ratio = tk.StringVar(value="0.2")
        self.flatline_seconds = tk.StringVar(value="3.0")
        self.threshold = tk.StringVar(value="0.5")
        self.bins = tk.StringVar(value="10")
        self.sample_idx = tk.StringVar(value="0")
        self.window_sec = tk.StringVar(value="0.5")
        self.num_perturbations = tk.StringVar(value="512")
        self.representative_case = tk.StringVar(value="")

        self.freq_vars = {f: tk.BooleanVar(value=(f == "100")) for f in FREQS}
        self.preview_image_tk = None
        self.preview_image_path: Path | None = None
        self.preview_images: list[Path] = []
        self.preview_index = -1

        self.configure(bg=self.colors["bg"])
        self._setup_style()
        self.runner = CommandRunner(self._log)
        self._build_ui()
        self.after(100, self._poll_runner)

    def previous_plot(self):
        if not self.preview_images:
            self.refresh_latest_plot()
            return

        self.preview_index = max(0, self.preview_index - 1)
        self._show_image(self.preview_images[self.preview_index])

    def next_plot(self):
        if not self.preview_images:
            self.refresh_latest_plot()
            return

        self.preview_index = min(len(self.preview_images) - 1, self.preview_index + 1)
        self._show_image(self.preview_images[self.preview_index])
    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        c = self.colors
        style.configure(".", font=("Segoe UI", 10), background=c["bg"], foreground=c["text"])
        style.configure("Main.TFrame", background=c["bg"])
        style.configure("Panel.TFrame", background=c["panel"])
        style.configure("Header.TFrame", background=c["header"])
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), background=c["header"], foreground="white")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), background=c["header"], foreground=c["header_sub"])
        style.configure("Section.TLabelframe", background=c["panel"], bordercolor=c["border"], relief="solid")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"), background=c["bg"], foreground=c["primary"])
        style.configure("TNotebook", background=c["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 7), background="#dfe6ef", foreground=c["text"], font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", c["panel"])], foreground=[("selected", c["primary"])])
        style.configure("TEntry", fieldbackground="white", bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"])
        style.configure("TCombobox", fieldbackground="white", background="white", bordercolor=c["border"])
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=8, foreground="white", background=c["primary"])
        style.map("Primary.TButton", background=[("active", c["primary_hover"])], foreground=[("active", "white")])
        style.configure("Success.TButton", font=("Segoe UI", 10, "bold"), padding=8, foreground="white", background=c["success"])
        style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), padding=7, foreground="white", background=c["danger"])
        style.configure("TButton", padding=6)

    def _build_ui(self):
        c = self.colors
        header = ttk.Frame(self, style="Header.TFrame", padding=(22, 18, 22, 14))
        header.pack(fill="x")
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text=APP_SUBTITLE, style="Subtitle.TLabel").pack(anchor="w", pady=(3, 0))

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=10)

        left = ttk.Frame(body, padding=8, style="Main.TFrame")
        right = ttk.Frame(body, padding=8, style="Main.TFrame")
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.tabs = ttk.Notebook(left)
        self.tabs.pack(fill="both", expand=True)
        self._tab_project()
        self._tab_prepare_train()
        self._tab_eval_plot()
        self._tab_xai()

        right_pane = ttk.PanedWindow(right, orient="vertical")
        right_pane.pack(fill="both", expand=True)

        log_container = ttk.Frame(right_pane, style="Main.TFrame")
        preview_container = ttk.Frame(right_pane, style="Main.TFrame")
        right_pane.add(log_container, weight=3)
        right_pane.add(preview_container, weight=2)

        log_frame = ttk.LabelFrame(log_container, text="Command log", style="Section.TLabelframe", padding=8)
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

        actions = ttk.Frame(log_container, padding=(0, 8, 0, 0), style="Main.TFrame")
        actions.pack(fill="x")
        ttk.Button(actions, text="Stop running command", style="Danger.TButton", command=self.runner.stop).pack(side="left")
        ttk.Button(actions, text="Clear log", command=lambda: self.log_text.delete("1.0", "end")).pack(side="left", padx=8)
        ttk.Button(actions, text="Open project folder", command=self._open_project_folder).pack(side="right")

        preview_frame = ttk.LabelFrame(preview_container, text="Plot preview", style="Section.TLabelframe", padding=8)
        preview_frame.pack(fill="both", expand=True)
        preview_toolbar = ttk.Frame(preview_frame, style="Panel.TFrame")
        preview_toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(preview_toolbar, text="Refresh latest plot", command=self.refresh_latest_plot).pack(side="left")
        ttk.Button(preview_toolbar, text="Previous plot", command=self.previous_plot).pack(side="left", padx=6)
        ttk.Button(preview_toolbar, text="Next plot", command=self.next_plot).pack(side="left", padx=6)
        ttk.Button(preview_toolbar, text="Open preview image", command=self.open_preview_image).pack(side="left", padx=6)
        ttk.Button(preview_toolbar, text="Open figures folder", command=self.open_figures_folder).pack(side="left", padx=6)
        self.preview_status = ttk.Label(preview_toolbar, text="No plot loaded yet.", foreground=c["muted"])
        self.preview_status.pack(side="right")
        self.preview_canvas = tk.Label(
            preview_frame,
            text="Generated plots will appear here.\nClick a plotting button, then the latest PNG will be shown automatically.",
            bg="white",
            fg=c["muted"],
            anchor="center",
            justify="center",
            relief="solid",
            bd=1,
        )
        self.preview_canvas.pack(fill="both", expand=True)
        self._log("Ready. Set the project root first, then choose an action.")

    def _row(self, parent, label, var, browse=None, width=60):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=24, background=self.colors["panel"]).pack(side="left")
        ent = ttk.Entry(row, textvariable=var, width=width)
        ent.pack(side="left", fill="x", expand=True)
        if browse:
            ttk.Button(row, text="Browse", command=browse).pack(side="left", padx=(6, 0))
        return ent

    def _combo_row(self, parent, label, var, values):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=24, background=self.colors["panel"]).pack(side="left")
        cb = ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=22)
        cb.pack(side="left")
        return cb

    def _freq_selector(self, parent):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="Sampling rates", width=24, background=self.colors["panel"]).pack(side="left")
        for f in FREQS:
            ttk.Checkbutton(row, text=f"{f} Hz", variable=self.freq_vars[f]).pack(side="left", padx=6)

    def _tab_project(self):
        tab = ttk.Frame(self.tabs, padding=14, style="Panel.TFrame")
        self.tabs.add(tab, text="Project")
        ttk.Label(tab, text="Project paths", font=("Segoe UI", 12, "bold"), background=self.colors["panel"], foreground=self.colors["primary"]).pack(anchor="w", pady=(0, 8))
        self._row(tab, "Project root", self.project_root, lambda: self._browse_dir(self.project_root))
        self._row(tab, "Raw PTB-XL data", self.raw_data_path, lambda: self._browse_dir(self.raw_data_path))
        self._row(tab, "Prepared dataset root", self.prepared_root, lambda: self._browse_dir(self.prepared_root))
        self._row(tab, "Checkpoint root", self.checkpoint_root, lambda: self._browse_dir(self.checkpoint_root))
        self._row(tab, "Dataset name", self.dataset_name)
        self._row(tab, "Preparation output root", self.out_root, lambda: self._browse_dir(self.out_root))

        ttk.Button(tab, text="Validate selected paths", style="Primary.TButton", command=self.validate_paths).pack(anchor="w", pady=(8, 0))

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
        link.bind("<Button-1>", lambda e: webbrowser.open("https://physionet.org/content/ptb-xl/1.0.3/"))

    def _tab_prepare_train(self):
        tab = ttk.Frame(self.tabs, padding=14, style="Panel.TFrame")
        self.tabs.add(tab, text="Prepare & Train")
        self._freq_selector(tab)
        prep = ttk.LabelFrame(tab, text="Data preparation", style="Section.TLabelframe", padding=10)
        prep.pack(fill="x", pady=10)
        self._combo_row(prep, "Balance mode", self.balance_mode, BALANCE_MODES)
        self._row(prep, "K-folds", self.kfolds, width=18)
        self._row(prep, "Hold-out test ratio", self.test_ratio, width=18)
        self._row(prep, "Flatline seconds", self.flatline_seconds, width=18)
        ttk.Button(prep, text="Run data preparation for selected rates", style="Primary.TButton", command=self.run_prepare).pack(anchor="e", pady=(8, 0))
        train = ttk.LabelFrame(tab, text="Training", style="Section.TLabelframe", padding=10)
        train.pack(fill="x", pady=10)
        self._combo_row(train, "Model", self.model, MODELS)
        self._combo_row(train, "Device", self.device, DEVICES)
        self._combo_row(train, "Train balance", self.train_balance, TRAIN_BALANCE)
        self._row(train, "Batch size", self.batch_size, width=18)
        self._row(train, "Epochs", self.epochs, width=18)
        self._row(train, "Learning rate", self.lr, width=18)
        ttk.Button(train, text="Run k-fold training for selected rate", style="Success.TButton", command=self.run_training).pack(anchor="e", pady=(8, 0))
        note = "Tip for quick testing: 100 Hz, cnn1d, CPU, batch size 8, epochs 2.\nFor final runs, increase epochs and use CUDA if available."
        ttk.Label(tab, text=note, justify="left", background=self.colors["panel"], foreground=self.colors["muted"]).pack(anchor="w", pady=10)

    def _tab_eval_plot(self):
        tab = ttk.Frame(self.tabs, padding=14, style="Panel.TFrame")
        self.tabs.add(tab, text="Evaluate & Plot")
        self._freq_selector(tab)
        self._combo_row(tab, "Model", self.model, MODELS)
        self._combo_row(tab, "Device", self.device, DEVICES)
        self._row(tab, "Batch size", self.batch_size, width=18)
        self._row(tab, "K-folds", self.kfolds, width=18)
        self._row(tab, "Threshold", self.threshold, width=18)
        self._row(tab, "ECE bins", self.bins, width=18)
        eval_box = ttk.LabelFrame(tab, text="Evaluation", style="Section.TLabelframe", padding=10)
        eval_box.pack(fill="x", pady=10)
        ttk.Button(eval_box, text="Run test-only ensemble evaluation", style="Primary.TButton", command=self.run_test_only).pack(side="left", padx=4, pady=4)
        ttk.Button(eval_box, text="Generate test metrics CSV", command=self.run_test_metrics_summary).pack(side="left", padx=4, pady=4)
        ttk.Button(eval_box, text="Generate validation metrics CSV", command=self.run_validation_metrics_summary).pack(side="left", padx=4, pady=4)
        plot_box = ttk.LabelFrame(tab, text="Plots", style="Section.TLabelframe", padding=10)
        plot_box.pack(fill="x", pady=10)
        ttk.Button(plot_box, text="ROC + PR grid", command=self.run_roc_pr).pack(side="left", padx=4, pady=4)
        ttk.Button(plot_box, text="Confusion matrix", command=self.run_confusion).pack(side="left", padx=4, pady=4)
        ttk.Button(plot_box, text="Calibration curves", command=self.run_calibration).pack(side="left", padx=4, pady=4)
    
        

    def _tab_xai(self):
        tab = ttk.Frame(self.tabs, padding=14, style="Panel.TFrame")
        self.tabs.add(tab, text="Explainability")
        self._freq_selector(tab)
        self._combo_row(tab, "Model", self.model, MODELS)
        self._combo_row(tab, "Device", self.device, DEVICES)
        self._row(tab, "Sample index", self.sample_idx, width=18)
        self._combo_row(tab, "Representative case", self.representative_case, REP_CASES)
        self._row(tab, "Window seconds", self.window_sec, width=18)
        self._row(tab, "Perturbations", self.num_perturbations, width=18)
        self._row(tab, "Output directory", self.output_dir, lambda: self._browse_dir(self.output_dir))
        box = ttk.LabelFrame(tab, text="Post-hoc explanation", style="Section.TLabelframe", padding=10)
        box.pack(fill="x", pady=10)
        ttk.Button(box, text="Run LIME explanation", style="Primary.TButton", command=self.run_lime).pack(side="left", padx=4, pady=4)
        ttk.Button(box, text="Open Grad-CAM notebook", command=self.open_gradcam_notebook).pack(side="left", padx=4, pady=4)
        ttk.Button(box, text="Open LIME results folder", command=self.open_lime_results).pack(side="left", padx=4, pady=4)
        note = "For LIME, choose exactly one sampling frequency. If Representative case is empty, sample index is used.\nGrad-CAM opens as a notebook because your current Grad-CAM file is an .ipynb."
        ttk.Label(tab, text=note, justify="left", background=self.colors["panel"], foreground=self.colors["muted"]).pack(anchor="w", pady=12)

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(initialdir=var.get() or str(Path.cwd()))
        if path:
            var.set(path)

    def _project_root(self) -> Path:
        return norm_path(self.project_root.get()).resolve()

    def _py_base(self) -> str:
        root = self._project_root()
        candidates = []
        if sys.platform.startswith("win"):
            candidates += [root / "venv" / "Scripts" / "python.exe", root / ".venv" / "Scripts" / "python.exe"]
        else:
            candidates += [root / "venv" / "bin" / "python", root / ".venv" / "bin" / "python"]
        for p in candidates:
            if p.exists():
                return str(p)
        return sys.executable

    def _py_cmd(self) -> list[str]:
        return [self._py_base(), "-u"]

    def _selected_rates(self) -> list[str]:
        rates = [f for f, v in self.freq_vars.items() if v.get()]
        if not rates:
            messagebox.showwarning("No sampling rate selected", "Select at least one sampling rate.")
        return rates

    def _script(self, *parts: str) -> Path:
        return self._project_root().joinpath(*parts)

    def _run(self, cmd: list[str]):
        self.runner.run(cmd, cwd=self._project_root())

    def _prepared_rate_dir(self, rate: str) -> Path:
        prepared = norm_path(self.prepared_root.get())
        if prepared.name.lower() == f"{rate}hz".lower():
            return prepared
        return prepared / f"{rate}hz"

    def _checkpoint_model_dir(self, rate: str) -> Path:
        ckpt = norm_path(self.checkpoint_root.get())
        if ckpt.name.lower() == f"{rate}hz".lower():
            return ckpt / self.model.get()
        return ckpt / f"{rate}hz" / self.model.get()

    def _validate_fold_files(self, rate: str) -> bool:
        data_dir = self._prepared_rate_dir(rate)
        missing = [data_dir / f"fold_{i}.pt" for i in range(1, int(self.kfolds.get()) + 1) if not (data_dir / f"fold_{i}.pt").exists()]
        if missing:
            msg = "Missing fold files. Training cannot start.\n\nExpected folder:\n" + str(data_dir) + "\n\nMissing:\n" + "\n".join(str(p.name) for p in missing)
            messagebox.showerror("Prepared data incomplete", msg)
            self._log("[VALIDATION ERROR] " + msg.replace("\n", " | "))
            return False
        return True

    def _candidate_image_roots(self) -> list[Path]:
        root = self._project_root()
        return [root / "figures", root / "checkpoints", root / "lime_results", root / "explainable"]

    def _all_image_files(self) -> list[Path]:
        images: list[Path] = []
        for folder in self._candidate_image_roots():
            if not folder.exists():
                continue
            for ext in IMAGE_EXTENSIONS:
                images.extend(folder.rglob(f"*{ext}"))

        return sorted(images, key=lambda p: p.stat().st_mtime)

    def _latest_image_file(self) -> Path | None:
        images = self._all_image_files()
        if not images:
            return None
        return images[-1]

    def refresh_latest_plot(self):
        self.preview_images = self._all_image_files()

        if not self.preview_images:
            self.preview_status.configure(text="No PNG/JPG plot found.")
            self.preview_canvas.configure(image="", text="No generated image found yet.\nRun a plotting command first.")
            self.preview_image_tk = None
            self.preview_image_path = None
            self.preview_index = -1
            return

        self.preview_index = len(self.preview_images) - 1
        self._show_image(self.preview_images[self.preview_index])

    def _show_image(self, image_path: Path):
        if not PIL_AVAILABLE:
            self.preview_status.configure(text="Pillow is not installed.")
            self.preview_canvas.configure(image="", text="Install Pillow to preview plots:\n\npip install pillow")
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
            self.preview_canvas.configure(image="", text=f"Could not load image:\n{image_path}\n\n{exc}")
            self.preview_image_tk = None
            self.preview_image_path = None

    def open_preview_image(self):
        if self.preview_image_path and self.preview_image_path.exists():
            self._open_path(self.preview_image_path)
        else:
            self.refresh_latest_plot()
            if self.preview_image_path and self.preview_image_path.exists():
                self._open_path(self.preview_image_path)

    def open_figures_folder(self):
        figures = self._project_root() / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        self._open_path(figures)

    def validate_paths(self):
        root = self._project_root()

        self._log("\nPath validation", "header")
        self._log("-" * 60, "header")

        self._log_status_line(f"Project root: {root}", root.exists())
        self._log(f"Python: {self._py_base()}")

        train_script = self._script("src", "train.py")
        prepare_script = self._script("src", "ecg_preprocessing", "ecg_data_prepare.py")

        self._log_status_line(f"Train script: {train_script}", train_script.exists())
        self._log_status_line(f"Prepare script: {prepare_script}", prepare_script.exists())

        for rate in self._selected_rates():
            data_dir = self._prepared_rate_dir(rate)
            self._log_status_line(f"Prepared {rate} Hz: {data_dir}", data_dir.exists())

            for i in range(1, int(self.kfolds.get()) + 1):
                p = data_dir / f"fold_{i}.pt"
                self._log_status_line(f"  {p.name}:", p.exists())

        self._log("-" * 60 + "\n", "header")

    def run_prepare(self):
        rates = self._selected_rates()
        if not rates:
            return

        script = self._script("src", "ecg_preprocessing", "ecg_data_prepare.py")
        if not script.exists():
            messagebox.showerror("Missing script", f"Could not find:\n{script}")
            return

        for rate in rates:
            cmd = [
                *self._py_cmd(),
                str(script),
                "--dataset_path", self.raw_data_path.get(),
                "--name", self.dataset_name.get(),
                "--fs", rate,
                "--out_root", self.out_root.get(),
                "--folds", self.kfolds.get(),
                "--test_ratio", self.test_ratio.get(),
                "--balance_mode", self.balance_mode.get(),
                "--flatline_seconds", self.flatline_seconds.get(),
            ]
            self._run(cmd)

    def run_training(self):
        rates = self._selected_rates()
        if not rates:
            return
        script = self._script("src", "train.py")
        if not script.exists():
            messagebox.showerror("Missing script", f"Could not find:\n{script}")
            return
        rate = rates[0]
        if not self._validate_fold_files(rate):
            return
        data_path = self._prepared_rate_dir(rate)
        cmd = [*self._py_cmd(), str(script), "--data_path", str(data_path), "--model", self.model.get(), "--train_balance", self.train_balance.get(), "--batch_size", self.batch_size.get(), "--epochs", self.epochs.get(), "--lr", self.lr.get(), "--kfolds", self.kfolds.get(), "--device", self.device.get()]
        self._run(cmd)
        if len(rates) > 1:
            self._log("Note: training starts one selected rate at a time. This is safer for GPU/RAM. Run again for the next rate.")

    def run_test_only(self):
        rates = self._selected_rates()
        if not rates:
            return
        script = self._script("src", "train.py")
        if not script.exists():
            messagebox.showerror("Missing script", f"Could not find:\n{script}")
            return
        rate = rates[0]
        data_path = self._prepared_rate_dir(rate)
        cmd = [*self._py_cmd(), str(script), "--data_path", str(data_path), "--model", self.model.get(), "--test_only", "--batch_size", self.batch_size.get(), "--kfolds", self.kfolds.get(), "--device", self.device.get()]
        self._run(cmd)

    def run_test_metrics_summary(self):
        script = self._find_script(["src/plotting/compute_auroc_ece_test_ensambling_metric.py"])
        if script:
            cmd = [*self._py_cmd(), str(script), "--root", self.checkpoint_root.get(), "--bins", self.bins.get(), "--threshold", self.threshold.get()]
            self._run(cmd)

    def run_validation_metrics_summary(self):
        script = self._find_script(["src/plotting/summary_sensitivity_auroc_gen.py"])
        if script:
            cmd = [*self._py_cmd(), str(script), "--root", self.checkpoint_root.get(), "--bins", self.bins.get(), "--threshold", self.threshold.get(), "--folds", self.kfolds.get()]
            self._run(cmd)

    def run_roc_pr(self):
        self._run_script_candidates(["src/plotting/roc_pr_ploting_2x2.py"])

    def run_confusion(self):
        script = self._find_script(["src/plotting/confiousion_matrix_2x2.py"])
        if script:
            rates = self._selected_rates()
            if not rates:
                return
            freqs = [f"{r}hz" for r in rates]
            cmd = [*self._py_cmd(), str(script), "--root", self.checkpoint_root.get(), "--freqs", *freqs, "--models", self.model.get(), "--folds", self.kfolds.get(), "--threshold", self.threshold.get()]
            self._run(cmd)

    def run_calibration(self):
        self._run_script_candidates(["src/plotting/probability_calibiration_curv_2.py"])

    def run_sampling_plot(self):
        self._run_script_candidates(["src/thesis_report_plotting/figure_resampling.py"])

    def run_balancing_plot(self):
        self._run_script_candidates(["src/thesis_report_plotting/different_balancing_mode.py"])

    def run_afib_norm_stats(self):
        self._run_script_candidates(["src/thesis_report_plotting/plot_ptbxl_afib_norm_statistics_percent.py"])

    def run_lime(self):
        rates = self._selected_rates()
        if len(rates) != 1:
            messagebox.showwarning("Choose one frequency", "LIME needs exactly one sampling frequency.")
            return
        script = self._find_script(["explain_lime.py", "explainable/explain_lime.py"])
        if not script:
            return
        rate = rates[0]
        data_path = self._prepared_rate_dir(rate)
        cmd = [*self._py_cmd(), str(script), "--data_path", str(data_path), "--model", self.model.get(), "--split", "test", "--folds", self.kfolds.get(), "--device", self.device.get(), "--window_sec", self.window_sec.get(), "--num_perturbations", self.num_perturbations.get(), "--mask_strategy", "local_mean", "--explain_class", "pred"]
        if self.output_dir.get().strip():
            cmd += ["--output_dir", self.output_dir.get().strip()]
        rep = self.representative_case.get().strip()
        if rep:
            cmd += ["--representative_case", rep]
        else:
            cmd += ["--sample_idx", self.sample_idx.get()]
        self._run(cmd)

    def open_gradcam_notebook(self):
        nb = self._project_root() / "xai_gradcam.ipynb"
        if not nb.exists():
            nb = self._project_root() / "src" / "xai_gradcam.ipynb"
        if not nb.exists():
            messagebox.showerror("Notebook not found", "Could not find xai_gradcam.ipynb in project root or src/.")
            return
        self._run([*self._py_cmd(), "-m", "jupyter", "notebook", str(nb)])

    def open_lime_results(self):
        p = norm_path(self.output_dir.get().strip() or (self._project_root() / "lime_results"))
        p.mkdir(parents=True, exist_ok=True)
        self._open_path(p)

    def _find_script(self, candidates: Iterable[str]) -> Path | None:
        for c in candidates:
            p = self._project_root() / c
            if p.exists():
                return p
        messagebox.showerror("Missing script", "Could not find any of:\n" + "\n".join(candidates))
        return None

    def _run_script_candidates(self, candidates: list[str]):
        script = self._find_script(candidates)
        if script:
            self._run([*self._py_cmd(), str(script)])

    def _open_project_folder(self):
        self._open_path(self._project_root())

    def _open_path(self, path: Path):
        path = path.expanduser().resolve()
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))

    def _log(self, text: str, tag: str | None = None):
        try:
            self.log_text.insert("end", text + "\n", tag)
            self.log_text.see("end")
            self.update_idletasks()
        except Exception:
            pass

    def _log_status_line(self, label: str, ok: bool):
        try:
            self.log_text.insert("end", label + " ", "normal")
            if ok:
                self.log_text.insert("end", "OK\n", "ok")
            else:
                self.log_text.insert("end", "MISSING\n", "missing")
            self.log_text.see("end")
            self.update_idletasks()
        except Exception:
            pass

    def _poll_runner(self):
        finished = False
        try:
            while True:
                line = self.runner.q.get_nowait()
                self._log(line)
                if "[Finished with exit code" in line:
                    finished = True
        except queue.Empty:
            pass
        if finished:
            self.after(500, self.refresh_latest_plot)
        self.after(80, self._poll_runner)


if __name__ == "__main__":
    app = PipelineGUI()
    app.mainloop()
