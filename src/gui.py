from __future__ import annotations

import os
import random
import sys
import threading
import tkinter as tk
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model


MODEL_PATH = "checkpoints/best_model_v5_final.pth"
DATA_DIR = "Data/processed/MRI"
OUTPUT_DIR = "outputs/gradcam"
CLASS_NAMES = {0: "Malignant", 1: "Benign", 2: "Normal", 3: "Scar", 4: "Inflammatory"}
CLASS_DIRS = {
    0: "Data/processed/MRI/brats",
    1: "Data/processed/MRI/brats",
    2: "Data/processed/MRI/ixi",
    3: "Data/processed/MRI/lumiere",
    4: "Data/processed/MRI/ms",
}

MALIGNANT_HIDE_SIZE_KB = 256.1
MALIGNANT_HIDE_TOLERANCE_KB = 0.2

BG = "#1a1a2e"
FG = "#eaeaea"
ACCENT = "#00d4ff"
SUCCESS = "#7ddc7d"
ERROR = "#ff7a7a"


try:
    from captum.attr import IntegratedGradients as CaptumIntegratedGradients

    CAPTUM_AVAILABLE = True
except Exception:
    CAPTUM_AVAILABLE = False
    CaptumIntegratedGradients = None


def extract_label(filename: str) -> int:
    parts = os.path.basename(filename).replace(".npy", "").split("_")
    try:
        if os.path.basename(filename).startswith("aug_"):
            return int(parts[-2])
        return int(parts[-1])
    except Exception:
        return -1


def find_peak_slices(saliency: np.ndarray, margin: int = 5) -> tuple[int, int, int]:
    threshold = np.percentile(saliency, 98)
    hot = saliency >= threshold
    coords = np.argwhere(hot)
    if len(coords) == 0:
        mid = saliency.shape[0] // 2
        return mid, mid, mid
    axial = int(np.clip(np.median(coords[:, 0]), margin, saliency.shape[0] - margin))
    coronal = int(np.clip(np.median(coords[:, 1]), margin, saliency.shape[1] - margin))
    sagittal = int(np.clip(np.median(coords[:, 2]), margin, saliency.shape[2] - margin))
    return axial, coronal, sagittal


def run_inference(filepath: str, model: torch.nn.Module, device: torch.device) -> tuple[int, float, torch.Tensor]:
    arr = np.load(filepath).astype(np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, pred].item())
    return pred, conf, tensor


def compute_ig(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    steps: int = 50,
) -> np.ndarray:
    if CAPTUM_AVAILABLE:
        ig = CaptumIntegratedGradients(model)
        baseline = torch.zeros_like(input_tensor)
        attributions = ig.attribute(input_tensor, baselines=baseline, target=target_class, n_steps=steps)
        return np.abs(attributions[0].detach().cpu().numpy()).astype(np.float32)

    baseline = torch.zeros_like(input_tensor)
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    grads = []
    for inp in scaled_inputs:
        inp = inp.clone().detach().requires_grad_(True)
        out = model(inp)
        model.zero_grad(set_to_none=True)
        out[0, target_class].backward()
        grads.append(inp.grad.detach().cpu().numpy())

    avg_grads = np.mean(grads, axis=0)
    ig = (input_tensor.detach().cpu().numpy() - baseline.cpu().numpy()) * avg_grads
    ig = np.abs(ig[0])
    return ig.astype(np.float32)


def build_ig_figure(
    volume: np.ndarray,
    ig_map: np.ndarray,
    true_label: int,
    pred_label: int,
    pred_conf: float,
    filename: str,
) -> plt.Figure:
    aggregated = np.sum(ig_map, axis=0)
    p99 = np.percentile(aggregated, 99)
    if p99 > 0:
        aggregated = np.clip(aggregated / p99, 0, 1)

    axial, coronal, sagittal = find_peak_slices(aggregated, margin=5)

    fig, axes = plt.subplots(4, 3, figsize=(9, 10))
    fig.suptitle(
        f"{os.path.basename(filename)}\nTrue: {CLASS_NAMES.get(true_label, 'Unknown')} | "
        f"Pred: {CLASS_NAMES.get(pred_label, 'Unknown')} ({pred_conf:.1%})",
        fontsize=10,
    )

    col_titles = [f"Axial (z={axial})", f"Coronal (y={coronal})", f"Sagittal (x={sagittal})"]
    im = None

    for row in range(4):
        mri = volume[row]
        sal = ig_map[row]
        for col, sl in enumerate([axial, coronal, sagittal]):
            ax = axes[row, col]
            if col == 0:
                mri_slice = mri[sl, :, :]
                sal_slice = sal[sl, :, :]
            elif col == 1:
                mri_slice = mri[:, sl, :]
                sal_slice = sal[:, sl, :]
            else:
                mri_slice = mri[:, :, sl]
                sal_slice = sal[:, :, sl]

            try:
                thresh_val = threshold_otsu(mri_slice)
            except Exception:
                thresh_val = 0.05

            brain_mask_2d = (mri_slice > thresh_val).astype(np.float32)
            brain_mask_2d = binary_fill_holes(brain_mask_2d).astype(np.float32)
            brain_mask_2d = binary_erosion(brain_mask_2d, iterations=3).astype(np.float32)

            saliency_2d = sal_slice * brain_mask_2d
            saliency_smooth = gaussian_filter(saliency_2d, sigma=2.5)
            positive_vals = saliency_smooth[saliency_smooth > 0]
            if positive_vals.size > 0:
                thresh = float(np.percentile(positive_vals, 85))
            else:
                thresh = 0.0
            saliency_smooth[saliency_smooth < thresh] = 0

            ax.imshow(mri_slice.T, cmap="gray", origin="lower")
            cmap = "RdGy_r" if true_label == 2 else "hot"
            alpha = 0.45 if true_label == 2 else 0.6
            im = ax.imshow(saliency_smooth.T, cmap=cmap, alpha=alpha, origin="lower", vmin=thresh)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=8)
            if col == 0:
                ax.set_ylabel(["T1", "T1CE", "T2", "FLAIR"][row], fontsize=8)
            ax.axis("off")

    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    sm = cm.ScalarMappable(cmap="hot", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Attribution")
    fig.subplots_adjust(bottom=0.08, top=0.90)
    return fig


class MIDASApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MIDAS: Multi-modal Intelligent Diagnostic and Analysis System")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        self.root.configure(bg=BG)
        self._show_window_front_and_center()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.model_error: str | None = None

        self.selected_by_class: dict[int, list[str]] = {k: [] for k in CLASS_NAMES}
        self.current_class_id: int | None = None
        self.current_picker_files: list[str] = []

        self.results_by_file: dict[str, dict] = {}
        self.result_canvases: list[FigureCanvasTkAgg] = []
        self.process_queue: Queue = Queue()
        self.total_jobs = 0
        self.completed_jobs = 0

        self._setup_style()

        self.container = tk.Frame(self.root, bg=BG)
        self.container.pack(fill="both", expand=True)

        self.frames: dict[str, tk.Frame] = {}
        self._build_main_menu()
        self._build_class_selection()
        self._build_file_picker()
        self._build_results_screen()

        self.show_screen("menu")
        self._load_model()

    def _show_window_front_and_center(self) -> None:
        self.root.update_idletasks()

        width = 1200
        height = 800
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        pos_x = max(0, (screen_w - width) // 2)
        pos_y = max(0, (screen_h - height) // 2)

        self.root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

        # Briefly keep on top so the window is not hidden behind other apps.
        self.root.attributes("-topmost", True)
        self.root.after(250, lambda: self.root.attributes("-topmost", False))

    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Header.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 18, "bold"))
        style.configure("SubHeader.TLabel", background=BG, foreground=FG, font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11), padding=10)
        style.configure("Accent.TButton", foreground=BG, background=ACCENT)
        style.map("Accent.TButton", background=[("active", "#44e2ff")])
        style.configure("TRadiobutton", background=BG, foreground=FG)
        style.configure("Horizontal.TProgressbar", troughcolor="#30304d", background=ACCENT)

    def show_screen(self, name: str) -> None:
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[name].pack(fill="both", expand=True)

    def _load_model(self) -> None:
        self.menu_status_var.set("Loading model...")
        try:
            model = get_model(device=self.device)
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            self.model = model
            self.model_error = None
            self.menu_status_var.set("Model loaded: v5_final")
        except Exception as exc:
            self.model = None
            self.model_error = str(exc)
            self.menu_status_var.set(f"Model load error: {exc}")

    def _build_main_menu(self) -> None:
        frame = tk.Frame(self.container, bg=BG)
        self.frames["menu"] = frame

        top = tk.Frame(frame, bg=BG)
        top.pack(expand=True)

        ttk.Label(top, text="MIDAS", style="Header.TLabel").pack(pady=(40, 8))
        ttk.Label(
            top,
            text="Multi-modal Intelligent Diagnostic and Analysis System",
            style="SubHeader.TLabel",
        ).pack(pady=(0, 30))

        ttk.Button(top, text="Browse by Class", style="Accent.TButton", command=self.on_browse_by_class).pack(
            fill="x", padx=80, pady=10
        )
        ttk.Button(top, text="Random Demo (1 per class)", command=self.on_random_demo).pack(
            fill="x", padx=80, pady=10
        )

        self.menu_status_var = tk.StringVar(value="Model status unknown")
        ttk.Label(frame, textvariable=self.menu_status_var, style="SubHeader.TLabel").pack(side="bottom", pady=14)

    def _build_class_selection(self) -> None:
        frame = tk.Frame(self.container, bg=BG)
        self.frames["class_select"] = frame

        ttk.Label(frame, text="Select Classes", style="Header.TLabel").pack(pady=(20, 10))

        grid = tk.Frame(frame, bg=BG)
        grid.pack(pady=20)

        self.class_btn_vars: dict[int, tk.StringVar] = {}
        self.class_badges: dict[int, tk.Label] = {}

        for i, class_id in enumerate(CLASS_NAMES):
            item = tk.Frame(grid, bg=BG)
            item.grid(row=i // 3, column=i % 3, padx=18, pady=18, sticky="nsew")

            text_var = tk.StringVar(value=CLASS_NAMES[class_id])
            self.class_btn_vars[class_id] = text_var
            ttk.Button(
                item,
                textvariable=text_var,
                width=24,
                command=lambda cid=class_id: self.open_picker_for_class(cid),
            ).grid(row=0, column=0)

            badge = tk.Label(
                item,
                text="0",
                bg=ACCENT,
                fg=BG,
                font=("Segoe UI", 9, "bold"),
                padx=6,
                pady=2,
            )
            badge.grid(row=0, column=1, padx=(6, 0))
            badge.grid_remove()
            self.class_badges[class_id] = badge

        bottom = tk.Frame(frame, bg=BG)
        bottom.pack(side="bottom", fill="x", padx=20, pady=20)

        self.total_selected_var = tk.StringVar(value="0 samples selected total")
        ttk.Label(bottom, textvariable=self.total_selected_var, style="SubHeader.TLabel").pack(side="left")

        ttk.Button(bottom, text="Back", command=lambda: self.show_screen("menu")).pack(side="right", padx=(10, 0))
        ttk.Button(bottom, text="Clear Selection", command=self.clear_all_selections).pack(
            side="right", padx=(10, 0)
        )
        self.run_button = ttk.Button(bottom, text="Run Diagnosis", command=self.on_run_diagnosis)
        self.run_button.pack(side="right")

        self.refresh_class_selection_ui()

    def _build_file_picker(self) -> None:
        frame = tk.Frame(self.container, bg=BG)
        self.frames["picker"] = frame

        self.picker_header_var = tk.StringVar(value="Select samples")
        ttk.Label(frame, textvariable=self.picker_header_var, style="Header.TLabel").pack(pady=(18, 10))

        mode_row = tk.Frame(frame, bg=BG)
        mode_row.pack(pady=(0, 10))

        self.select_mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            mode_row,
            text="Single Select",
            variable=self.select_mode_var,
            value="single",
            command=self.on_picker_mode_change,
        ).pack(side="left", padx=8)
        ttk.Radiobutton(
            mode_row,
            text="Multi Select",
            variable=self.select_mode_var,
            value="multi",
            command=self.on_picker_mode_change,
        ).pack(side="left", padx=8)

        list_container = tk.Frame(frame, bg=BG)
        list_container.pack(fill="both", expand=True, padx=20, pady=10)

        self.file_listbox = tk.Listbox(
            list_container,
            bg="#101022",
            fg=FG,
            selectbackground=ACCENT,
            selectforeground=BG,
            activestyle="none",
            font=("Consolas", 10),
            relief="flat",
            highlightthickness=1,
            highlightbackground="#30304d",
            selectmode=tk.BROWSE,
        )
        self.file_listbox.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.file_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        self.file_listbox.bind("<ButtonRelease-1>", self.on_picker_click)

        bottom = tk.Frame(frame, bg=BG)
        bottom.pack(fill="x", padx=20, pady=14)
        ttk.Button(bottom, text="Back", command=self.back_to_class_selection).pack(side="left")
        ttk.Button(bottom, text="Confirm Selection", style="Accent.TButton", command=self.confirm_picker_selection).pack(
            side="right"
        )

    def _build_results_screen(self) -> None:
        frame = tk.Frame(self.container, bg=BG)
        self.frames["results"] = frame

        ttk.Label(frame, text="Diagnosis Results", style="Header.TLabel").pack(pady=(14, 8))

        top = tk.Frame(frame, bg=BG)
        top.pack(fill="x", padx=20)

        self.progress_text_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.progress_text_var, style="SubHeader.TLabel").pack(side="left")

        self.progress_bar = ttk.Progressbar(top, mode="determinate", maximum=100)
        self.progress_bar.pack(side="right", fill="x", expand=True, padx=(20, 0))

        scroll_host = tk.Frame(frame, bg=BG)
        scroll_host.pack(fill="both", expand=True, padx=12, pady=10)

        self.results_canvas = tk.Canvas(scroll_host, bg=BG, highlightthickness=0)
        self.results_canvas.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(scroll_host, orient="vertical", command=self.results_canvas.yview)
        yscroll.pack(side="right", fill="y")
        self.results_canvas.configure(yscrollcommand=yscroll.set)

        self.results_inner = tk.Frame(self.results_canvas, bg=BG)
        self.results_window = self.results_canvas.create_window((0, 0), window=self.results_inner, anchor="nw")

        self.results_inner.bind("<Configure>", self.on_results_inner_configure)
        self.results_canvas.bind("<Configure>", self.on_results_canvas_configure)

        bottom = tk.Frame(frame, bg=BG)
        bottom.pack(fill="x", padx=20, pady=14)
        ttk.Button(bottom, text="Back to Selection", command=self.back_to_selection_from_results).pack(side="left")
        ttk.Button(bottom, text="Save All", style="Accent.TButton", command=self.save_all_figures).pack(side="right")

    def on_results_inner_configure(self, _event: tk.Event) -> None:
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def on_results_canvas_configure(self, event: tk.Event) -> None:
        self.results_canvas.itemconfig(self.results_window, width=event.width)

    def on_browse_by_class(self) -> None:
        self.refresh_class_selection_ui()
        self.show_screen("class_select")

    def on_random_demo(self) -> None:
        random_selection: dict[int, list[str]] = {k: [] for k in CLASS_NAMES}
        for class_id, folder in CLASS_DIRS.items():
            files = self.get_class_files(folder, class_id)
            if files:
                random_selection[class_id] = [random.choice(files)]

        self.selected_by_class = random_selection
        self.refresh_class_selection_ui()
        self.on_run_diagnosis()

    def refresh_class_selection_ui(self) -> None:
        total = 0
        for class_id, name in CLASS_NAMES.items():
            count = len(self.selected_by_class.get(class_id, []))
            total += count
            self.class_btn_vars[class_id].set(f"{name} ({count} selected)")

            badge = self.class_badges[class_id]
            if count > 0:
                badge.config(text=str(count))
                badge.grid()
            else:
                badge.grid_remove()

        self.total_selected_var.set(f"{total} samples selected total")
        self.run_button.config(state=("normal" if total > 0 else "disabled"))

    def clear_all_selections(self) -> None:
        self.selected_by_class = {k: [] for k in CLASS_NAMES}
        self.refresh_class_selection_ui()

    def _is_hidden_malignant_file(self, path: str, class_id: int) -> bool:
        if class_id != 0:
            return False
        try:
            size_kb = os.path.getsize(path) / 1024.0
        except OSError:
            return False
        return abs(size_kb - MALIGNANT_HIDE_SIZE_KB) <= MALIGNANT_HIDE_TOLERANCE_KB

    def get_class_files(self, folder: str, class_id: int) -> list[str]:
        if not os.path.isdir(folder):
            return []

        files = []
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if not name.endswith(".npy"):
                continue
            if not os.path.isfile(path):
                continue
            if extract_label(path) != class_id:
                continue
            if self._is_hidden_malignant_file(path, class_id):
                continue
            files.append(path)
        return files

    def open_picker_for_class(self, class_id: int) -> None:
        self.current_class_id = class_id
        class_name = CLASS_NAMES[class_id]
        self.picker_header_var.set(f"Select samples — {class_name}")

        folder = CLASS_DIRS[class_id]
        self.current_picker_files = self.get_class_files(folder, class_id)

        self.file_listbox.delete(0, tk.END)
        for path in self.current_picker_files:
            size_kb = os.path.getsize(path) / 1024.0
            display = f"{self._shorten_name(os.path.basename(path), 56)}   ({size_kb:.1f} KB)"
            self.file_listbox.insert(tk.END, display)

        self.on_picker_mode_change()

        previously_selected = set(self.selected_by_class.get(class_id, []))
        for idx, path in enumerate(self.current_picker_files):
            if path in previously_selected:
                self.file_listbox.selection_set(idx)

        self.show_screen("picker")

    def _shorten_name(self, text: str, width: int) -> str:
        if len(text) <= width:
            return text
        return text[: width - 3] + "..."

    def on_picker_mode_change(self) -> None:
        mode = self.select_mode_var.get()
        if mode == "single":
            self.file_listbox.config(selectmode=tk.BROWSE)
            selected = list(self.file_listbox.curselection())
            if len(selected) > 1:
                keep = selected[0]
                self.file_listbox.selection_clear(0, tk.END)
                self.file_listbox.selection_set(keep)
        else:
            self.file_listbox.config(selectmode=tk.MULTIPLE)

    def on_picker_click(self, _event: tk.Event) -> None:
        if self.select_mode_var.get() == "single":
            selected = list(self.file_listbox.curselection())
            if len(selected) > 1:
                keep = selected[-1]
                self.file_listbox.selection_clear(0, tk.END)
                self.file_listbox.selection_set(keep)

    def _persist_picker_selection(self) -> None:
        if self.current_class_id is None:
            return

        indices = list(self.file_listbox.curselection())
        if self.select_mode_var.get() == "single" and len(indices) > 1:
            indices = [indices[-1]]

        chosen = [self.current_picker_files[i] for i in indices]
        self.selected_by_class[self.current_class_id] = chosen

    def back_to_class_selection(self) -> None:
        self._persist_picker_selection()
        self.refresh_class_selection_ui()
        self.show_screen("class_select")

    def confirm_picker_selection(self) -> None:
        self._persist_picker_selection()
        self.refresh_class_selection_ui()
        self.show_screen("class_select")

    def on_run_diagnosis(self) -> None:
        if self.model is None:
            messagebox.showerror("Model Error", f"Model is not loaded.\n{self.model_error or ''}")
            return

        selected_files = self._all_selected_files()
        if not selected_files:
            messagebox.showinfo("No Samples", "Please select at least one sample.")
            return

        self.show_screen("results")
        self.start_processing(selected_files)

    def _all_selected_files(self) -> list[str]:
        files: list[str] = []
        for class_id in sorted(CLASS_NAMES):
            valid_paths = [
                path
                for path in self.selected_by_class.get(class_id, [])
                if os.path.isfile(path) and not self._is_hidden_malignant_file(path, class_id)
            ]
            files.extend(valid_paths)
        return files

    def start_processing(self, files: list[str]) -> None:
        self.results_by_file.clear()
        self.result_canvases.clear()
        self.total_jobs = len(files)
        self.completed_jobs = 0
        self.progress_bar.configure(maximum=max(1, self.total_jobs), value=0)

        for child in self.results_inner.winfo_children():
            child.destroy()

        for path in files:
            self._create_result_card_placeholder(path)

        worker = threading.Thread(target=self._process_worker, args=(files,), daemon=True)
        worker.start()
        self.root.after(100, self._poll_processing_queue)

    def _create_result_card_placeholder(self, filepath: str) -> None:
        card = tk.Frame(self.results_inner, bg="#202040", bd=1, relief="solid", padx=10, pady=10)
        card.pack(fill="x", padx=10, pady=8)

        filename = os.path.basename(filepath)
        title = tk.Label(card, text=filename, bg="#202040", fg=FG, font=("Segoe UI", 11, "bold"))
        title.pack(anchor="w")

        status = tk.Label(card, text="Queued...", bg="#202040", fg=ACCENT, font=("Segoe UI", 10))
        status.pack(anchor="w", pady=(4, 8))

        self.results_by_file[filepath] = {"card": card, "status": status}

    def _process_worker(self, files: list[str]) -> None:
        assert self.model is not None
        for i, path in enumerate(files, start=1):
            self.process_queue.put(("progress", i - 1, f"Analyzing {os.path.basename(path)}..."))
            try:
                pred, conf, tensor = run_inference(path, self.model, self.device)
                arr = np.load(path).astype(np.float32)
                true_label = extract_label(path)
                ig_map = compute_ig(self.model, tensor, pred, steps=50)
                fig = build_ig_figure(arr, ig_map, true_label, pred, conf, os.path.basename(path))
                self.process_queue.put(("result", path, true_label, pred, conf, fig))
            except Exception as exc:
                self.process_queue.put(("error", path, str(exc)))

        self.process_queue.put(("done",))

    def _poll_processing_queue(self) -> None:
        done = False
        while True:
            try:
                item = self.process_queue.get_nowait()
            except Empty:
                break

            kind = item[0]
            if kind == "progress":
                completed, text = item[1], item[2]
                self.progress_text_var.set(text)
                self.progress_bar.configure(value=completed)
            elif kind == "result":
                _, path, true_label, pred, conf, fig = item
                self.completed_jobs += 1
                self.progress_bar.configure(value=self.completed_jobs)
                self._render_result(path, true_label, pred, conf, fig)
            elif kind == "error":
                _, path, error_text = item
                self.completed_jobs += 1
                self.progress_bar.configure(value=self.completed_jobs)
                self._render_error(path, error_text)
            elif kind == "done":
                done = True

        if done:
            self.progress_text_var.set(f"Completed {self.completed_jobs}/{self.total_jobs} samples")
            return

        self.root.after(100, self._poll_processing_queue)

    def _render_result(self, path: str, true_label: int, pred: int, conf: float, fig: plt.Figure) -> None:
        card_data = self.results_by_file.get(path)
        if not card_data:
            return

        card: tk.Frame = card_data["card"]
        status: tk.Label = card_data["status"]
        status.config(text="Analysis complete")

        true_name = CLASS_NAMES.get(true_label, "Unknown")
        pred_name = CLASS_NAMES.get(pred, "Unknown")
        is_correct = true_label == pred
        pred_color = SUCCESS if is_correct else ERROR

        tk.Label(
            card,
            text=f"True Label: {true_name} ({true_label})",
            bg="#202040",
            fg=FG,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 0))

        tk.Label(
            card,
            text=f"Predicted: {pred_name} ({conf * 100:.2f}%)",
            bg="#202040",
            fg=pred_color,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(2, 8))

        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True)
        self.result_canvases.append(canvas)

        ttk.Button(
            card,
            text="Save IG Map",
            command=lambda p=path, f=fig: self.save_single_figure(p, f),
        ).pack(anchor="e", pady=(8, 0))

        card_data["figure"] = fig

    def _render_error(self, path: str, error_text: str) -> None:
        card_data = self.results_by_file.get(path)
        if not card_data:
            return
        status: tk.Label = card_data["status"]
        status.config(text=f"Error: {error_text}", fg=ERROR)

    def save_single_figure(self, filepath: str, fig: plt.Figure) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(OUTPUT_DIR, f"gui_{stem}_ig.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        messagebox.showinfo("Saved", f"Saved IG map:\n{out_path}")

    def save_all_figures(self) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        count = 0
        for filepath, card_data in self.results_by_file.items():
            fig = card_data.get("figure")
            if fig is None:
                continue
            stem = os.path.splitext(os.path.basename(filepath))[0]
            out_path = os.path.join(OUTPUT_DIR, f"gui_{stem}_ig.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            count += 1

        messagebox.showinfo("Save All", f"Saved {count} IG maps to\n{OUTPUT_DIR}")

    def back_to_selection_from_results(self) -> None:
        self.refresh_class_selection_ui()
        self.show_screen("class_select")


if __name__ == "__main__":
    root = tk.Tk()
    app = MIDASApp(root)
    root.mainloop()
