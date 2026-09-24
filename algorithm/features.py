#!/usr/bin/env python3
"""Image feature extractors for CLIP and DINO models."""

import time
from pathlib import Path
from typing import List, Sequence, Union

import torch
from PIL import Image
from torch import Tensor

from transformers import CLIPModel, CLIPProcessor

DEFAULT_CLIP_CHECKPOINT: str = "openai/clip-vit-large-patch14"
DEFAULT_DINO_CHECKPOINT: str = "facebook/dinov2-base"
DEFAULT_DINOV3_CHECKPOINT: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
CLIP_EMBED_DIM: int = 768
DINO_EMBED_DIM: int = 768

ImageLike = Union[Tensor, Image.Image]


def _to_pil(image: ImageLike) -> Image.Image:
    """Convert one raw [0, 1] CHW float tensor (or pass-through PIL image) to
    an RGB uint8 ``PIL.Image`` -- the single format handed to the HF
    processors, so preprocessing is never ambiguous.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, Tensor):
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected a (3, H, W) CHW tensor, got shape {tuple(image.shape)}."
            )
        clamped = image.detach().clamp(0.0, 1.0)
        uint8_hwc = (clamped * 255.0).round().byte().permute(1, 2, 0).contiguous().numpy()
        return Image.fromarray(uint8_hwc)
    raise TypeError(f"Unsupported image type: {type(image)!r}")


class ClipFeatureExtractor:
    """Frozen CLIP image feature extractor."""

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_CHECKPOINT,
        device: str = "cpu",
    ) -> None:
        self.device: torch.device = torch.device(device)
        self.model_name: str = model_name

        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name)
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def extract(
        self,
        images: Sequence[ImageLike],
        batch_size: int = 32,
    ) -> Tensor:
        """Extract 768-dim frozen CLIP image embeddings for a list of images."""
        if len(images) == 0:
            return torch.empty((0, CLIP_EMBED_DIM), dtype=torch.float32, device=self.device)

        all_features: List[Tensor] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            pil_chunk = [_to_pil(img) for img in chunk]
            inputs = self.processor(images=pil_chunk, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            chunk_out = self.model.get_image_features(pixel_values=pixel_values)
            # Older transformers return a tensor; 5.x returns an output object
            # whose pooler_output holds the same projected embedding.
            chunk_features = chunk_out if isinstance(chunk_out, Tensor) else chunk_out.pooler_output
            all_features.append(chunk_features.to(torch.float32))

        return torch.cat(all_features, dim=0)

    def __call__(self, images: Sequence[ImageLike], batch_size: int = 32) -> Tensor:
        return self.extract(images, batch_size=batch_size)


class DinoFeatureExtractor:
    """Frozen DINO (DINOv2 / DINOv3) image encoder: images -> R^768 embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_DINO_CHECKPOINT,
        device: str = "cpu",
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device: torch.device = torch.device(device)
        self.model_name: str = model_name

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as err:
            err_msg = str(err)
            if "gated repo" in err_msg.lower() or "restricted" in err_msg.lower() or "401" in err_msg:
                raise RuntimeError(
                    f"Failed to load gated model '{model_name}'. "
                    f"Please request access on HuggingFace (https://huggingface.co/{model_name}) "
                    f"and authenticate via `huggingface-cli login` or set HF_TOKEN env variable."
                ) from err
            raise

        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def extract(
        self,
        images: Sequence[ImageLike],
        batch_size: int = 32,
    ) -> Tensor:
        if len(images) == 0:
            return torch.empty((0, DINO_EMBED_DIM), dtype=torch.float32, device=self.device)

        all_features: List[Tensor] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            pil_chunk = [_to_pil(img) for img in chunk]
            inputs = self.processor(images=pil_chunk, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            outputs = self.model(pixel_values=pixel_values)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0, :]
            all_features.append(feats.to(torch.float32))

        return torch.cat(all_features, dim=0)

    def __call__(self, images: Sequence[ImageLike], batch_size: int = 32) -> Tensor:
        return self.extract(images, batch_size=batch_size)


def get_feature_extractor(
    backbone: str = "clip",
    device: str = "cpu", # we recommend cuda
):
    b_lower = backbone.lower().strip()
    if b_lower in ("clip", "clip-vit-l14", "openai/clip-vit-large-patch14"):
        return ClipFeatureExtractor(model_name=DEFAULT_CLIP_CHECKPOINT, device=device)
    elif b_lower in ("dinov3", "dino-v3", "dino3", "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        return DinoFeatureExtractor(model_name=DEFAULT_DINOV3_CHECKPOINT, device=device)
    elif b_lower in ("dino", "dinov2", "dino-v2", "dinov2-base", "facebook/dinov2-base"):
        return DinoFeatureExtractor(model_name=DEFAULT_DINO_CHECKPOINT, device=device)
    elif "clip" in b_lower:
        return ClipFeatureExtractor(model_name=backbone, device=device)
    else:
        return DinoFeatureExtractor(model_name=backbone, device=device)


if __name__ == "__main__":
    import argparse
    import sys

    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))
    from dataset.ctx_uxo import DEFAULT_INSTANCES_ROOT, list_class_instances

    parser = argparse.ArgumentParser(description="Feature extractor timing check.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="clip",
        help="Backbone: 'clip', 'dino', or HuggingFace checkpoint ID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    sample_classes = ["Aviation_Bomb", "Grenade", "Mortar_Bomb"]
    sample_paths: List[Path] = []
    for class_name in sample_classes:
        sample_paths.extend(list_class_instances("train", class_name, root=DEFAULT_INSTANCES_ROOT)[:3])
    images = [Image.open(p).convert("RGB") for p in sample_paths]

    t0 = time.perf_counter()
    extractor = get_feature_extractor(backbone=args.backbone, device=args.device)
    print(f"{args.backbone} loaded in {time.perf_counter() - t0:.2f} s")

    t0 = time.perf_counter()
    unbatched_feats = torch.cat([extractor.extract([img]) for img in images], dim=0)
    unbatched_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    batched_feats = extractor.extract(images, batch_size=len(images))
    batched_time = time.perf_counter() - t0

    n = len(images)
    print(f"shape={tuple(batched_feats.shape)} dtype={batched_feats.dtype}")
    print(f"batched vs unbatched max abs diff: {(batched_feats - unbatched_feats).abs().max().item():.2e}")
    print(f"unbatched: {unbatched_time / n * 1000:.1f} ms/image, batched: {batched_time / n * 1000:.1f} ms/image")
    assert torch.isfinite(batched_feats).all()
    assert not (batched_feats == 0).all()
