"""Map-independent models owned by the server's resident inference worker."""

import gc
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.demo_utils import CKPT_PATH, DEFAULT_BOXERNET_CKPT
from utils.taxonomy import load_text_labels


def inference_args(options, *, input=None, output_dir=None):
    """Adapt HTTP options without importing the CLI or initializing a map."""
    values = dict(options)
    values["labels"] = [s.strip() for s in values["labels"].split(",") if s.strip()]
    values["ckpt"] = str(
        Path(values.get("ckpt") or Path(CKPT_PATH) / DEFAULT_BOXERNET_CKPT)
        .expanduser()
        .resolve()
    )
    return SimpleNamespace(
        **values,
        input=input,
        output_dir=output_dir,
        validate_input=False,
        gt2d=False,
        cache2d=False,
        cache3d=False,
        skip_viz=True,
        write_name="boxer",
    )


def select_device(force_cpu):
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def inference_dtype(device, precision):
    if precision == "bfloat16" or (
        precision is None and device == "cuda" and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float32


def semantic_embeddings(owl, labels):
    values = owl.text_embeddings.detach().float().cpu().reshape(len(labels), -1)
    return {
        label.lower(): torch.nn.functional.normalize(values[i], dim=0)
        for i, label in enumerate(labels)
    }


def _checkpoint_key(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


class XNavRuntime:
    def __init__(self):
        self.boxer = self.owl = None
        self._boxer_key = self._owl_key = None
        self.labels = []
        self.embeddings = {}
        self.device = "initializing"
        self.ready = False

    @torch.inference_mode()
    def configure(self, args):
        """Load only changed resources; a failed update is never usable."""
        from boxernet.boxernet import BoxerNet
        from owl.owl_wrapper import OwlWrapper

        self.ready = False
        try:
            labels = load_text_labels(args.labels)
            if not labels:
                raise ValueError("At least one text label is required")
            device = select_device(args.force_cpu)
            dtype = inference_dtype(device, args.force_precision)
            boxer_key = (_checkpoint_key(args.ckpt), device)
            owl_key = (device, dtype if device == "cuda" else torch.float32)
            update_embeddings = False
            if boxer_key != self._boxer_key:
                self.boxer = None
                self._release_cache()
                self.boxer = BoxerNet.load_from_checkpoint(args.ckpt, device=device)
                self._boxer_key = boxer_key
            if owl_key != self._owl_key:
                self.owl = None
                self._release_cache()
                self.owl = OwlWrapper(
                    device,
                    text_prompts=labels,
                    min_confidence=args.thresh2d,
                    precision=args.force_precision,
                    warmup=True,
                )
                self._owl_key = owl_key
                update_embeddings = True
            elif labels != self.labels:
                self.owl.set_text_prompts(labels)
                self.owl._warmup()
                update_embeddings = True
            self.owl.min_confidence = args.thresh2d
            if update_embeddings:
                self.embeddings = semantic_embeddings(self.owl, labels)
            self.labels = list(labels)
            self.device = device
            self.dtype = dtype
            self.ready = True
        except BaseException:
            self.close()
            raise

    def validate(self, args):
        """Never perform lazy model loading when a job starts."""
        device = select_device(args.force_cpu)
        dtype = inference_dtype(device, args.force_precision)
        if (
            not self.ready
            or self._boxer_key != (_checkpoint_key(args.ckpt), device)
            or self._owl_key != (device, dtype if device == "cuda" else torch.float32)
            or self.labels != load_text_labels(args.labels)
            or self.owl.min_confidence != args.thresh2d
        ):
            raise ValueError(
                "Models or vocabulary changed; apply inference parameters again"
            )

    @staticmethod
    def _release_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self):
        self.ready = False
        self.boxer = self.owl = None
        self._boxer_key = self._owl_key = None
        self.labels = []
        self.embeddings = {}
        self._release_cache()
