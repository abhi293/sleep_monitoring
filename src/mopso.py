"""
mopso.py
────────────────────────────────────────────────────────────────
Multi-Objective Particle Swarm Optimization (MOPSO) for joint
optimization of sleep classification model hyperparameters.

Objectives (simultaneously):
  1. Maximize accuracy (→ minimize 1 - accuracy)
  2. Minimize false alarm rate (for apnea / disturbance events)
  3. Minimize model parameter count (efficiency proxy)

MOPSO maintains a Pareto archive of non-dominated solutions.
Uses Python multiprocessing for parallel fitness evaluations
across all available CPU cores.

Search Space
────────────
  • cnn_filters_1   : [32, 64, 128]
  • cnn_filters_2   : [64, 128, 256]
  • gru_units       : [32, 64, 128]
  • lstm_units      : [32, 64, 128]
  • dense_units     : [64, 128, 256]
  • dropout         : [0.10 … 0.50] (continuous)
  • learning_rate   : [1e-4 … 1e-2] (log-scale continuous)
  • window_size     : [20, 25, 30, 35, 40] (discrete)
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Search space definition
# ────────────────────────────────────────────────────────────────

DISCRETE_OPTIONS = {
    "cnn_filters_1":  [32, 64, 128],
    "cnn_filters_2":  [64, 128, 256],
    "gru_units":      [32, 64, 128],
    "lstm_units":     [32, 64, 128],
    "dense_units":    [64, 128, 256],
    "window_size":    [20, 25, 30, 35, 40],
}

DIM_NAMES = [
    "cnn_filters_1",   # 0: discrete index
    "cnn_filters_2",   # 1: discrete index
    "gru_units",       # 2: discrete index
    "lstm_units",      # 3: discrete index
    "dense_units",     # 4: discrete index
    "dropout",         # 5: continuous [0,1] → [0.10, 0.50]
    "log_lr",          # 6: continuous [0,1] → log [1e-4, 1e-2]
    "window_size",     # 7: discrete index
]
N_DIM = len(DIM_NAMES)


def _decode_position(pos: np.ndarray) -> Dict[str, Any]:
    """Map raw PSO position vector [0,1]^N_DIM → hyper-parameter dict."""
    def idx(val: float, options: list) -> int:
        return int(np.clip(round(val * (len(options) - 1)), 0, len(options) - 1))

    return {
        "cnn_filters_1": DISCRETE_OPTIONS["cnn_filters_1"][idx(pos[0], DISCRETE_OPTIONS["cnn_filters_1"])],
        "cnn_filters_2": DISCRETE_OPTIONS["cnn_filters_2"][idx(pos[1], DISCRETE_OPTIONS["cnn_filters_2"])],
        "gru_units":     DISCRETE_OPTIONS["gru_units"][idx(pos[2], DISCRETE_OPTIONS["gru_units"])],
        "lstm_units":    DISCRETE_OPTIONS["lstm_units"][idx(pos[3], DISCRETE_OPTIONS["lstm_units"])],
        "dense_units":   DISCRETE_OPTIONS["dense_units"][idx(pos[4], DISCRETE_OPTIONS["dense_units"])],
        "dropout":       float(np.clip(0.10 + pos[5] * 0.40, 0.10, 0.50)),
        "learning_rate": float(10 ** (-4 + pos[6] * 2)),   # 1e-4 … 1e-2
        "window_size":   DISCRETE_OPTIONS["window_size"][idx(pos[7], DISCRETE_OPTIONS["window_size"])],
    }


# ────────────────────────────────────────────────────────────────
# Pareto helpers
# ────────────────────────────────────────────────────────────────

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution a dominates b (all objectives ≤, at least one <)."""
    return bool(np.all(a <= b) and np.any(a < b))


def _update_archive(archive: List[dict], candidate: dict) -> List[dict]:
    """
    Insert candidate into Pareto archive; remove solutions dominated by it;
    do not insert if dominated by any existing solution.
    """
    obj_c = np.array(candidate["objectives"])
    dominated_by_existing = any(_dominates(np.array(s["objectives"]), obj_c)
                                 for s in archive)
    if dominated_by_existing:
        return archive
    # Remove archive members dominated by candidate
    archive = [s for s in archive if not _dominates(obj_c, np.array(s["objectives"]))]
    archive.append(deepcopy(candidate))
    return archive


# ────────────────────────────────────────────────────────────────
# Fitness evaluation
# ────────────────────────────────────────────────────────────────

def _evaluate_candidate(
    hyperparams: Dict[str, Any],
    data: dict,
    quick_epochs: int = 3,
    n_features: int = 22,
) -> Tuple[float, float, float]:
    """
    Train a small model with the given hyperparams for quick_epochs and return
    three objectives: (1 - val_accuracy, false_alarm_rate, log10_param_count).

    NOTE: This runs in a forked worker process. TF is imported locally to
    avoid CUDA/fork issues – CPU-only training suffices for MOPSO trials.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import sys
    if "tensorflow" not in sys.modules:
        # Running in a subprocess worker — force CPU and suppress CUDA noise
        os.environ["CUDA_VISIBLE_DEVICES"]   = ""
        os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "1"
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    else:
        # Already in the main process with TF configured — just alias it
        import tensorflow as tf

    from src.model import build_from_config
    from src.preprocessing import create_windows, apply_scaler

    window_size = int(hyperparams["window_size"])

    # Re-window if window_size changed
    if data["X_train"].shape[1] != window_size:
        stride = max(1, window_size // 3)
        X_tr, y_tr = create_windows(
            data["df_train"], window_size, stride, data["feature_cols"], n_jobs=1)
        X_va, y_va = create_windows(
            data["df_val"], window_size, stride, data["feature_cols"], n_jobs=1)
    else:
        X_tr, y_tr = data["X_train"], data["y_train"]
        X_va, y_va = data["X_val"],   data["y_val"]

    if len(X_tr) == 0 or len(X_va) == 0:
        return 1.0, 1.0, 6.0

    cfg = {**hyperparams, "num_classes": 4}
    model = build_from_config(cfg, window_size, n_features,
                              class_weights=data.get("class_weights"))

    # Subsample for MOPSO fitness evaluation.
    # 2 500 train / 800 val balances signal quality vs per-worker memory.
    # batch_size=64 (not 256) is critical: BiGRU/BiLSTM backprop with large
    # batches in multiple concurrent workers causes CPU OOM.
    max_tr = min(len(X_tr), 2500)
    idx_tr = np.random.choice(len(X_tr), max_tr, replace=False)
    max_va = min(len(X_va), 800)
    idx_va = np.random.choice(len(X_va), max_va, replace=False)

    try:
        model.fit(
            X_tr[idx_tr], y_tr[idx_tr],
            validation_data=(X_va[idx_va], y_va[idx_va]),
            epochs=quick_epochs,
            batch_size=64,
            verbose=1,
        )

        loss, acc = model.evaluate(X_va[idx_va], y_va[idx_va], verbose=0)

        # False alarm rate: fraction of non-apnea windows flagged as Awake
        # when true label is Sleep (class 1,2,3).
        y_pred = np.argmax(model.predict(X_va[idx_va], verbose=0), axis=1)
        sleep_mask = y_va[idx_va] > 0                   # true sleep epochs
        if sleep_mask.sum() > 0:
            false_alarm_rate = float((y_pred[sleep_mask] == 0).sum() / sleep_mask.sum())
        else:
            false_alarm_rate = 0.0

        log_params = float(np.log10(max(model_param_count(model), 1)))
        result = (1.0 - float(acc)), false_alarm_rate, log_params
    except Exception as e:  # catches OOM and any other per-worker failure
        import logging as _log
        _log.getLogger(__name__).warning(
            "MOPSO candidate failed (%s: %s) — returning worst-case objectives.",
            type(e).__name__, str(e)[:120]
        )
        result = (1.0, 1.0, 6.0)  # worst-case: 0% acc, 100% FAR, 1M params
    finally:
        # Explicitly delete the model BEFORE clear_session to release all
        # TF tensors held by this worker.  Without this, loky reuses the
        # worker process and the TF graph grows unboundedly each iteration,
        # causing the per-iter wall time to blow up (370s → 1200s+).
        del model
        tf.keras.backend.clear_session()
        import gc; gc.collect()

    return result


def model_param_count(model) -> int:
    import numpy as np
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


# ────────────────────────────────────────────────────────────────
# MOPSO engine
# ────────────────────────────────────────────────────────────────

@dataclass
class Particle:
    position:  np.ndarray = field(default_factory=lambda: np.random.rand(N_DIM))
    velocity:  np.ndarray = field(default_factory=lambda: np.zeros(N_DIM))
    best_pos:  np.ndarray = field(default_factory=lambda: np.zeros(N_DIM))
    best_obj:  Optional[np.ndarray] = None


class MOPSO:
    """
    Multi-Objective PSO for sleep model hyperparameter tuning.

    Objectives are minimized: (1-accuracy, false_alarm_rate, log_param_count)
    Maintains a Pareto archive from which the global guide is sampled.
    """

    def __init__(
        self,
        n_particles: int = 10,
        max_iter: int = 10,
        n_jobs: int = 1,
        quick_epochs: int = 1,
        pareto_dir: str = "mopso_results",
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.n_jobs      = n_jobs
        self.quick_epochs = quick_epochs
        self.pareto_dir  = Path(pareto_dir)
        self.w  = w   # inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient

        self.pareto_archive: List[dict] = []
        self.history: List[dict] = []

        self.pareto_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialization ─────────────────────────────────────────

    def _init_swarm(self) -> List[Particle]:
        particles = []
        for _ in range(self.n_particles):
            p = Particle(
                position=np.random.rand(N_DIM),
                velocity=(np.random.rand(N_DIM) - 0.5) * 0.2,
            )
            p.best_pos = p.position.copy()
            particles.append(p)
        return particles

    # ── Guide selection from Pareto archive ───────────────────

    def _select_guide(self) -> np.ndarray:
        """Randomly select a guide from the archive (crowding-based preferred)."""
        if not self.pareto_archive:
            return np.random.rand(N_DIM)
        guide = self.pareto_archive[np.random.randint(len(self.pareto_archive))]
        return guide["position"].copy()

    # ── Velocity & position update ────────────────────────────

    def _update_particle(self, p: Particle, guide_pos: np.ndarray) -> Particle:
        r1, r2 = np.random.rand(N_DIM), np.random.rand(N_DIM)
        p.velocity = (
            self.w  * p.velocity
            + self.c1 * r1 * (p.best_pos - p.position)
            + self.c2 * r2 * (guide_pos  - p.position)
        )
        p.position = np.clip(p.position + p.velocity, 0.0, 1.0)
        return p

    # ── Evaluate swarm in parallel (or sequentially for n_jobs=1) ──────────

    def _evaluate_swarm(
        self,
        particles: List[Particle],
        data: dict,
        n_features: int,
    ) -> List[np.ndarray]:
        positions = [p.position for p in particles]
        hyperparams_list = [_decode_position(pos) for pos in positions]

        if self.n_jobs == 1:
            # Run in the main process — no forking, no data pickling, no OOM
            objectives_list = [
                _evaluate_candidate(hp, data, self.quick_epochs, n_features)
                for hp in hyperparams_list
            ]
        else:
            objectives_list = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(
                delayed(_evaluate_candidate)(hp, data, self.quick_epochs, n_features)
                for hp in hyperparams_list
            )
        return [np.array(obj) for obj in objectives_list]

    # ── Main optimization loop ────────────────────────────────

    def optimize(self, data: dict, n_features: int = 11) -> List[dict]:
        """
        Run MOPSO optimization.

        Parameters
        ----------
        data     : dict returned by preprocessing.full_pipeline() plus raw
                   df_train / df_val for re-windowing when window_size varies.
        n_features : number of input features

        Returns
        -------
        List of Pareto-optimal solutions (each is a dict with keys:
            'hyperparams', 'objectives', 'position')
        """
        logger.info("Starting MOPSO: %d particles × %d iterations on %d workers",
                    self.n_particles, self.max_iter, self.n_jobs)
        t0 = time.time()

        particles = self._init_swarm()

        for iteration in range(self.max_iter):
            t_iter = time.time()
            objectives = self._evaluate_swarm(particles, data, n_features)

            # Update personal bests & archive
            for p, obj in zip(particles, objectives):
                candidate = {
                    "position":    p.position.copy(),
                    "hyperparams": _decode_position(p.position),
                    "objectives":  obj.tolist(),
                }
                # Update personal best
                if p.best_obj is None or _dominates(obj, p.best_obj):
                    p.best_pos = p.position.copy()
                    p.best_obj = obj.copy()
                # Update archive
                self.pareto_archive = _update_archive(self.pareto_archive, candidate)

            # Update velocities & positions
            for p in particles:
                guide = self._select_guide()
                p = self._update_particle(p, guide)

            # Log with ETA
            best_acc = min(s["objectives"][0] for s in self.pareto_archive)
            elapsed = time.time() - t0
            avg_per_iter = elapsed / (iteration + 1)
            eta = avg_per_iter * (self.max_iter - iteration - 1)
            logger.info(
                "MOPSO iter %02d/%02d | archive=%d | best_1-acc=%.4f | "
                "%.1fs/iter | ETA %.0fs",
                iteration + 1, self.max_iter, len(self.pareto_archive),
                best_acc, time.time() - t_iter, eta,
            )
            self.history.append({
                "iteration": iteration + 1,
                "archive_size": len(self.pareto_archive),
                "best_accuracy": 1 - best_acc,
            })

        logger.info("MOPSO finished in %.1fs | Pareto archive: %d solutions",
                    time.time() - t0, len(self.pareto_archive))
        self._save_results()
        return self.pareto_archive

    # ── Best solution selector ────────────────────────────────

    def best_hyperparams(self, priority: str = "accuracy") -> Dict[str, Any]:
        """
        Select the best solution from Pareto archive.

        priority : 'accuracy' | 'efficiency' | 'balanced'
        """
        if not self.pareto_archive:
            raise ValueError("No solutions in archive. Run optimize() first.")

        objs = np.array([s["objectives"] for s in self.pareto_archive])
        # objectives: [1-acc, false_alarm, log_params]
        if priority == "accuracy":
            idx = int(np.argmin(objs[:, 0]))
        elif priority == "efficiency":
            # Weighted sum: accuracy × 0.5 + efficiency × 0.5
            norm = (objs - objs.min(0)) / (objs.max(0) - objs.min(0) + 1e-9)
            idx = int(np.argmin(0.5 * norm[:, 0] + 0.5 * norm[:, 2]))
        else:  # balanced
            norm = (objs - objs.min(0)) / (objs.max(0) - objs.min(0) + 1e-9)
            score = norm.mean(axis=1)
            idx = int(np.argmin(score))

        solution = self.pareto_archive[idx]
        logger.info("Selected hyperparams [%s]: %s | objectives: %s",
                    priority, solution["hyperparams"], solution["objectives"])
        return solution["hyperparams"]

    # ── Persistence ──────────────────────────────────────────

    def _save_results(self) -> None:
        out = {
            "pareto_archive": [
                {**s, "position": s["position"].tolist()} for s in self.pareto_archive
            ],
            "history": self.history,
        }
        path = self.pareto_dir / "mopso_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("MOPSO results saved -> %s", path)

    def load_results(self, path: Optional[str] = None) -> None:
        path = path or str(self.pareto_dir / "mopso_results.json")
        with open(path) as f:
            data = json.load(f)
        self.pareto_archive = [
            {**s, "position": np.array(s["position"])} for s in data["pareto_archive"]
        ]
        self.history = data.get("history", [])
        logger.info("Loaded %d Pareto solutions from %s",
                    len(self.pareto_archive), path)
