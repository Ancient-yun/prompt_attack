"""FLUX.2 generator adapter."""

from __future__ import annotations

from functools import lru_cache

from PIL import Image

from prompt_attack.config import GeneratorConfig
from prompt_attack.generators.base import GenerationResult
from prompt_attack.utils.image import pil_to_tensor, tensor_to_pil


class Flux2Adapter:
    """FLUX.2-klein-4B image-editing adapter with soft prompt embeddings."""

    supports_gradient = True

    def __init__(self, config: GeneratorConfig, *, device: str) -> None:
        self.config = config
        self.device = device
        self._pipe = None

    def _dtype(self):
        import torch

        if self.config.precision in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if self.config.precision in {"fp16", "float16"}:
            return torch.float16
        return torch.float32

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe

        try:
            from diffusers import Flux2KleinPipeline

            pipe_cls = Flux2KleinPipeline
        except ImportError:
            from diffusers import DiffusionPipeline

            pipe_cls = DiffusionPipeline
        pipe = pipe_cls.from_pretrained(self.config.model_id, torch_dtype=self._dtype())
        if self.config.use_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        self._freeze_pipe(pipe)
        self._enable_memory_options(pipe)
        self._pipe = pipe
        return pipe

    @staticmethod
    def _freeze_pipe(pipe) -> None:
        """Freeze generator weights while keeping gradients through activations."""
        for module_name in ("text_encoder", "transformer", "vae"):
            module = getattr(pipe, module_name, None)
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def _enable_memory_options(self, pipe) -> None:
        if not self.config.gradient_checkpointing:
            return
        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            return
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        elif hasattr(transformer, "gradient_checkpointing"):
            transformer.gradient_checkpointing = True

    @staticmethod
    def _hard_prompt_from_prompt(prompt: str) -> str:
        pieces = [part for part in prompt.split() if not (part.startswith("[V") and part.endswith("]"))]
        return " ".join(pieces).strip() or prompt

    @lru_cache(maxsize=128)
    def _cached_hard_prompt_embeds(self, hard_prompt: str):
        """Encode the hard text prompt once and keep it detached."""
        import torch

        pipe = self._load_pipe()
        with torch.no_grad():
            prompt_embeds, _ = pipe.encode_prompt(
                prompt=hard_prompt,
                device=getattr(pipe, "_execution_device", self.device),
            )
        return prompt_embeds.detach().to("cpu")

    def soft_token_dim(self, hard_prompt: str) -> int:
        """Return FLUX.2 prompt embedding width for this hard prompt."""
        embeds = self._cached_hard_prompt_embeds(hard_prompt)
        return int(embeds.shape[-1])

    def _compose_prompt_embeds(self, *, hard_prompt: str, soft_tokens):
        """Prepend learnable continuous tokens to hard prompt embeddings."""
        import torch

        if not isinstance(soft_tokens, torch.Tensor):
            raise TypeError("FLUX.2 soft-token attack expects a torch.Tensor parameter.")
        hard_prompt_embeds = self._cached_hard_prompt_embeds(hard_prompt).to(
            device=soft_tokens.device,
            dtype=self._dtype(),
        )
        if soft_tokens.ndim != 2:
            raise ValueError(f"Expected soft tokens with shape [N, D], got {tuple(soft_tokens.shape)}")
        if soft_tokens.shape[-1] != hard_prompt_embeds.shape[-1]:
            raise ValueError(
                f"Soft-token dim {soft_tokens.shape[-1]} does not match FLUX.2 prompt "
                f"embedding dim {hard_prompt_embeds.shape[-1]}."
            )
        if soft_tokens.shape[0] >= hard_prompt_embeds.shape[1]:
            raise ValueError("Number of soft tokens must be smaller than max prompt sequence length.")
        soft = soft_tokens.unsqueeze(0).to(dtype=hard_prompt_embeds.dtype)
        hard_prompt_tail = hard_prompt_embeds[:, : hard_prompt_embeds.shape[1] - soft.shape[1], :]
        return torch.cat([soft, hard_prompt_tail], dim=1)

    @staticmethod
    def _differentiable_call(pipe, **kwargs):
        """Call a Diffusers pipeline while bypassing its @torch.no_grad wrapper."""
        import torch

        wrapped = getattr(pipe.__call__, "__wrapped__", None)
        if wrapped is None:
            raise RuntimeError("Cannot find undecorated FLUX.2 pipeline call for gradient mode.")
        with torch.enable_grad():
            return wrapped(pipe, **kwargs)

    def generate(
        self,
        *,
        input_image: Image.Image,
        input_tensor: object,
        prompt: str,
        soft_tokens: object,
        seed: int,
        require_grad: bool,
    ) -> GenerationResult:
        """Run FLUX.2 editing with either hard text or learnable prompt embeddings."""
        import torch

        pipe = self._load_pipe()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        hard_prompt = self._hard_prompt_from_prompt(prompt)

        if require_grad:
            prompt_embeds = self._compose_prompt_embeds(
                hard_prompt=hard_prompt,
                soft_tokens=soft_tokens,
            )
            output = self._differentiable_call(
                pipe,
                image=input_image,
                prompt=None,
                prompt_embeds=prompt_embeds,
                height=self.config.height,
                width=self.config.width,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator,
                output_type="pt",
                return_dict=True,
            )
            image_tensor = output.images
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(dtype=torch.float32).clamp(0, 1)
            return GenerationResult(
                image_tensor=image_tensor,
                pil_image=tensor_to_pil(image_tensor),
            )

        with torch.no_grad():
            image = pipe(
                image=input_image,
                prompt=hard_prompt,
                height=self.config.height,
                width=self.config.width,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator,
            ).images[0]
        return GenerationResult(image_tensor=pil_to_tensor(image, device=self.device), pil_image=image)
