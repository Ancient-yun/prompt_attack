"""FLUX.2 generator adapter."""

from __future__ import annotations

import random
from collections.abc import Sequence
from types import MethodType

from PIL import Image

from prompt_attack.attacks.learnable_tokens import (
    build_prompt,
    build_token_texts,
    validate_token_init_std,
)
from prompt_attack.config import GeneratorConfig
from prompt_attack.generators.base import (
    GenerationBatchResult,
    GenerationResult,
    LearnablePrompt,
    LearnablePromptBatch,
)
from prompt_attack.data.imagenet import FIXED_10_CLASSES
from prompt_attack.utils.image import pil_to_tensor, tensor_to_pil


class Flux2Adapter:
    """FLUX.2-klein-4B image-editing adapter with learnable tokenizer tokens."""

    supports_gradient = True

    def __init__(self, config: GeneratorConfig, *, device: str) -> None:
        self.config = config
        self.device = device
        self._pipe: object | None = None

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

    def _ensure_learnable_tokens(self, token_texts: tuple[str, ...]) -> tuple[int, ...]:
        """Add learnable token strings to the tokenizer once and return their ids."""
        pipe = self._load_pipe()
        tokenizer = pipe.tokenizer
        added = tokenizer.add_tokens(list(token_texts))
        if added:
            pipe.text_encoder.resize_token_embeddings(len(tokenizer))
            self._freeze_pipe(pipe)

        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        token_ids: list[int] = []
        for token_text in token_texts:
            token_id = tokenizer.convert_tokens_to_ids(token_text)
            if token_id is None or (unk_token_id is not None and token_id == unk_token_id):
                raise RuntimeError(f"Tokenizer did not register learnable token {token_text!r}.")
            token_ids.append(int(token_id))
        return tuple(token_ids)

    def _initializer_embedding(self, initializer: str):
        """Return the mean text-encoder embedding for the initializer text."""
        import torch

        pipe = self._load_pipe()
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        token_ids = tokenizer.encode(initializer, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Initializer {initializer!r} produced no tokenizer ids.")
        embedding_layer = text_encoder.get_input_embeddings()
        weight = embedding_layer.weight
        ids = torch.tensor(token_ids, device=weight.device, dtype=torch.long)
        with torch.no_grad():
            return weight.index_select(0, ids).float().mean(dim=0)

    def _fixed10_class_average_embedding(self):
        """Return the mean embedding across configured fixed-10 class labels."""
        import torch

        embeddings = [self._initializer_embedding(cls.label) for cls in FIXED_10_CLASSES]
        return torch.stack(embeddings, dim=0).mean(dim=0)

    def _random_real_token_embeddings(
        self,
        *,
        num_tokens: int,
        exclude_token_ids: tuple[int, ...],
        init_seed: int,
    ):
        """Return deterministic random existing-token embedding rows."""
        import torch

        pipe = self._load_pipe()
        tokenizer = pipe.tokenizer
        embedding_layer = pipe.text_encoder.get_input_embeddings()
        weight = embedding_layer.weight
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        for token in getattr(tokenizer, "all_special_tokens", []) or []:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None:
                special_ids.add(int(token_id))
        excluded = special_ids | set(exclude_token_ids)
        upper = min(len(tokenizer), weight.shape[0])
        candidates = [token_id for token_id in range(upper) if token_id not in excluded]
        if not candidates:
            raise ValueError("No tokenizer rows are available for random_real_tokens initialization.")
        rng = random.Random(init_seed)
        if len(candidates) >= num_tokens:
            sampled = rng.sample(candidates, num_tokens)
        else:
            sampled = [rng.choice(candidates) for _ in range(num_tokens)]
        ids = torch.tensor(sampled, device=weight.device, dtype=torch.long)
        with torch.no_grad():
            return weight.index_select(0, ids).float()

    def _initial_values(
        self,
        *,
        initializer: str,
        num_tokens: int,
        token_ids: tuple[int, ...],
        init_std: float,
        init_seed: int,
    ):
        """Create learnable-token initial values for one prompt state."""
        import torch

        normalized = initializer.lower().replace("-", "_")
        if normalized == "random_real_tokens":
            values = self._random_real_token_embeddings(
                num_tokens=num_tokens,
                exclude_token_ids=token_ids,
                init_seed=init_seed,
            )
        else:
            if normalized == "fixed10_class_average":
                initializer_embedding = self._fixed10_class_average_embedding()
            else:
                initializer_embedding = self._initializer_embedding(initializer)
            values = initializer_embedding.repeat(num_tokens, 1)
        values = values.to(device=torch.device(self.device), dtype=torch.float32)
        generator = torch.Generator(device=values.device).manual_seed(init_seed)
        noise = torch.randn(
            values.shape,
            generator=generator,
            device=values.device,
            dtype=values.dtype,
        )
        return values + noise * init_std

    def create_learnable_prompt(
        self,
        *,
        class_label: str,
        num_tokens: int,
        initializer: str,
        init_std: float,
        init_seed: int = 0,
    ) -> LearnablePrompt:
        """Create textual-inversion tokens initialized from an existing token."""
        import torch

        validate_token_init_std(init_std)
        token_texts = build_token_texts(num_tokens)
        token_ids = self._ensure_learnable_tokens(token_texts)
        values = self._initial_values(
            initializer=initializer,
            num_tokens=num_tokens,
            token_ids=token_ids,
            init_std=init_std,
            init_seed=init_seed,
        )
        prompt_state = LearnablePrompt(
            prompt_text=build_prompt(class_label, num_tokens),
            token_texts=token_texts,
            token_ids=token_ids,
            learnable_embeddings=torch.nn.Parameter(values),
        )
        self.sync_learnable_prompt(prompt_state)
        return prompt_state

    def create_axis_prompt(
        self,
        *,
        class_label: str,
        num_tokens: int,
        num_anchor_tokens: int,
        initializer: str,
        init_std: float,
        init_seed: int = 0,
    ) -> "AxisPromptState":
        """Create anchor/axis textual-inversion tokens for a strength-scheduled attack."""
        import torch

        from prompt_attack.attacks.axis_tokens import AxisPromptState, embeddings_at

        del class_label  # universal attacks share one class-agnostic token set
        validate_token_init_std(init_std)
        token_texts = build_token_texts(num_tokens)
        token_ids = self._ensure_learnable_tokens(token_texts)
        full_initial = self._initial_values(
            initializer=initializer,
            num_tokens=num_tokens,
            token_ids=token_ids,
            init_std=init_std,
            init_seed=init_seed,
        )
        anchor_values = full_initial[:num_anchor_tokens].clone()
        axis_base_values = full_initial[num_anchor_tokens:].clone().detach()
        axis_direction_values = torch.zeros_like(axis_base_values)
        state = AxisPromptState(
            token_texts=token_texts,
            token_ids=token_ids,
            num_anchor_tokens=num_anchor_tokens,
            num_axis_tokens=num_tokens - num_anchor_tokens,
            anchor_embeddings=torch.nn.Parameter(anchor_values),
            axis_base=axis_base_values,
            axis_direction=torch.nn.Parameter(axis_direction_values),
        )
        self.sync_axis_prompt(state, t=1.0)
        return state

    def sync_axis_prompt(self, state: "AxisPromptState", *, t: float = 1.0) -> None:
        """Synchronize generator-owned token rows for an axis prompt at strength ``t``."""
        from prompt_attack.attacks.axis_tokens import embeddings_at

        temp_prompt = LearnablePrompt(
            prompt_text="",
            token_texts=state.token_texts,
            token_ids=state.token_ids,
            learnable_embeddings=embeddings_at(state, t),
        )
        self.sync_learnable_prompt(temp_prompt)

    def create_learnable_prompt_batch(
        self,
        *,
        class_labels: Sequence[str],
        num_tokens: int,
        initializer: str,
        init_std: float,
        init_seed: int = 0,
    ) -> LearnablePromptBatch:
        """Create per-sample textual-inversion tokens for a batch."""
        import torch

        if not class_labels:
            raise ValueError("class_labels must not be empty.")
        validate_token_init_std(init_std)
        token_texts = build_token_texts(num_tokens)
        token_ids = self._ensure_learnable_tokens(token_texts)
        values = torch.stack(
            [
                self._initial_values(
                    initializer=initializer,
                    num_tokens=num_tokens,
                    token_ids=token_ids,
                    init_std=init_std,
                    init_seed=init_seed + index,
                )
                for index in range(len(class_labels))
            ],
            dim=0,
        )
        prompt_state = LearnablePromptBatch(
            prompt_texts=tuple(build_prompt(label, num_tokens) for label in class_labels),
            token_texts=token_texts,
            token_ids=token_ids,
            learnable_embeddings=torch.nn.Parameter(values),
        )
        self.sync_learnable_prompt_batch(prompt_state)
        return prompt_state

    def sync_learnable_prompt(self, prompt_state: LearnablePrompt) -> None:
        """Copy optimized token embeddings into the text encoder embedding rows."""
        import torch

        pipe = self._load_pipe()
        embedding_layer = pipe.text_encoder.get_input_embeddings()
        learnable_embeddings = prompt_state.learnable_embeddings
        if not isinstance(learnable_embeddings, torch.Tensor):
            raise TypeError("FLUX.2 prompt state must contain a torch.Tensor parameter.")
        ids = torch.tensor(prompt_state.token_ids, device=embedding_layer.weight.device, dtype=torch.long)
        values = learnable_embeddings.detach().to(
            device=embedding_layer.weight.device,
            dtype=embedding_layer.weight.dtype,
        )
        with torch.no_grad():
            embedding_layer.weight.index_copy_(0, ids, values)

    def sync_learnable_prompt_batch(self, prompt_state: LearnablePromptBatch) -> None:
        """Validate batch prompt values.

        Per-sample batch values are injected row-wise by the text embedding hook. Shared
        universal values are synchronized through ``sync_learnable_prompt``.
        """
        learnable_embeddings = prompt_state.learnable_embeddings
        import torch

        if not isinstance(learnable_embeddings, torch.Tensor):
            raise TypeError("FLUX.2 prompt state must contain a torch.Tensor parameter.")
        if learnable_embeddings.ndim not in {2, 3}:
            raise ValueError("Batch prompt embeddings must have shape [N, D] or [B, N, D].")

    def _register_learnable_embedding_hook(
        self,
        prompt_state: LearnablePrompt | LearnablePromptBatch,
    ):
        """Replace only the learnable token positions in text-encoder embeddings."""
        import torch

        pipe = self._load_pipe()
        embedding_layer = pipe.text_encoder.get_input_embeddings()
        learnable_embeddings = prompt_state.learnable_embeddings
        if not isinstance(learnable_embeddings, torch.Tensor):
            raise TypeError("FLUX.2 prompt state must contain a torch.Tensor parameter.")

        def replace_learnable_tokens(_module, inputs, output):
            input_ids = inputs[0] if inputs else None
            if input_ids is None or not torch.is_tensor(input_ids):
                return output
            edited = output
            replacements = learnable_embeddings.to(device=output.device, dtype=output.dtype)
            for index, token_id in enumerate(prompt_state.token_ids):
                mask = (input_ids == token_id).unsqueeze(-1)
                if mask.any():
                    if replacements.ndim == 2:
                        value = replacements[index].view(*([1] * (output.ndim - 1)), output.shape[-1])
                    elif replacements.ndim == 3:
                        row_replacements = replacements
                        if row_replacements.shape[0] != output.shape[0]:
                            if output.shape[0] % row_replacements.shape[0] != 0:
                                raise RuntimeError(
                                    "Cannot align batch learnable embeddings with text encoder output."
                                )
                            repeats = output.shape[0] // row_replacements.shape[0]
                            row_replacements = row_replacements.repeat_interleave(repeats, dim=0)
                        value = row_replacements[:, index, :].unsqueeze(1)
                    else:
                        raise ValueError("Learnable embeddings must have shape [N, D] or [B, N, D].")
                    edited = torch.where(mask, value, edited)
            return edited

        return embedding_layer.register_forward_hook(replace_learnable_tokens)

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
        prompt_state: LearnablePrompt,
        seed: int,
        require_grad: bool,
    ) -> GenerationResult:
        """Run FLUX.2 editing through the tokenizer/text-encoder prompt path."""
        import torch

        del input_tensor
        pipe = self._load_pipe()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if require_grad:
            hook = self._register_learnable_embedding_hook(prompt_state)
            try:
                output = self._differentiable_call(
                    pipe,
                    image=input_image,
                    prompt=prompt_state.prompt_text,
                    height=self.config.height,
                    width=self.config.width,
                    guidance_scale=self.config.guidance_scale,
                    num_inference_steps=self.config.num_inference_steps,
                    generator=generator,
                    output_type="pt",
                    return_dict=True,
                )
            finally:
                hook.remove()
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
                prompt=prompt_state.prompt_text,
                height=self.config.height,
                width=self.config.width,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator,
            ).images[0]
        return GenerationResult(image_tensor=pil_to_tensor(image, device=self.device), pil_image=image)

    @staticmethod
    def _prepare_per_sample_image_latents(pipe, images, batch_size, generator, device, dtype):
        """Prepare one independent conditioning image per batch row.

        FLUX.2 Klein treats `image=[...]` as multiple shared reference images for every prompt.
        For attack batches we need row `i` to see only image `i`, so this method replaces the
        pipeline's shared-reference implementation during `generate_batch`.
        """
        import torch

        if len(images) != batch_size:
            raise ValueError(
                f"Expected {batch_size} conditioning images for per-sample batching, got {len(images)}."
            )
        image_tensors = [image.to(device=device, dtype=dtype) for image in images]
        spatial_shapes = {tuple(image.shape[-2:]) for image in image_tensors}
        if len(spatial_shapes) != 1:
            raise ValueError("Per-sample FLUX.2 batching requires equal-sized conditioning images.")

        image_batch = torch.cat(image_tensors, dim=0)
        image_latents = pipe._encode_vae_image(image=image_batch, generator=generator)
        batch_size, _, height, width = image_latents.shape
        packed_latents = pipe._pack_latents(image_latents)

        t = torch.tensor([10])
        h = torch.arange(height)
        w = torch.arange(width)
        seq = torch.arange(1)
        image_latent_ids = torch.cartesian_prod(t, h, w, seq)
        image_latent_ids = image_latent_ids.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        return packed_latents, image_latent_ids

    def _per_sample_condition_images(self, input_images: Sequence[Image.Image]) -> list[Image.Image]:
        """Resize batch conditioning images to a common FLUX size."""
        return [
            image.resize(
                (self.config.width, self.config.height),
                Image.Resampling.LANCZOS,
            )
            for image in input_images
        ]

    def generate_batch(
        self,
        *,
        input_images: Sequence[Image.Image],
        input_tensor: object,
        prompt_state: LearnablePromptBatch,
        seeds: Sequence[int],
        require_grad: bool,
    ) -> GenerationBatchResult:
        """Run FLUX.2 editing for a prompt/image batch in one pipeline call."""
        import torch

        del input_tensor
        if len(input_images) != len(prompt_state.prompt_texts):
            raise ValueError("input_images and prompt_texts must have the same length.")
        if len(seeds) != len(input_images):
            raise ValueError("seeds and input_images must have the same length.")
        pipe = self._load_pipe()
        generators = [torch.Generator(device=self.device).manual_seed(seed) for seed in seeds]
        condition_images = self._per_sample_condition_images(input_images)
        original_prepare_image_latents = pipe.prepare_image_latents

        def prepare_per_sample_image_latents(pipe_self, images, batch_size, generator, device, dtype):
            return self._prepare_per_sample_image_latents(
                pipe_self,
                images,
                batch_size,
                generator,
                device,
                dtype,
            )

        if require_grad:
            hook = self._register_learnable_embedding_hook(prompt_state)
            try:
                pipe.prepare_image_latents = MethodType(prepare_per_sample_image_latents, pipe)
                output = self._differentiable_call(
                    pipe,
                    image=condition_images,
                    prompt=list(prompt_state.prompt_texts),
                    height=self.config.height,
                    width=self.config.width,
                    guidance_scale=self.config.guidance_scale,
                    num_inference_steps=self.config.num_inference_steps,
                    generator=generators,
                    output_type="pt",
                    return_dict=True,
                )
            finally:
                hook.remove()
                pipe.prepare_image_latents = original_prepare_image_latents
            image_tensor = output.images
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(dtype=torch.float32).clamp(0, 1)
            return GenerationBatchResult(
                image_tensor=image_tensor,
                pil_images=[],
            )

        hook = self._register_learnable_embedding_hook(prompt_state)
        try:
            pipe.prepare_image_latents = MethodType(prepare_per_sample_image_latents, pipe)
            with torch.no_grad():
                output = pipe(
                    image=condition_images,
                    prompt=list(prompt_state.prompt_texts),
                    height=self.config.height,
                    width=self.config.width,
                    guidance_scale=self.config.guidance_scale,
                    num_inference_steps=self.config.num_inference_steps,
                    generator=generators,
                    output_type="pt",
                    return_dict=True,
                )
        finally:
            hook.remove()
            pipe.prepare_image_latents = original_prepare_image_latents
        image_tensor = output.images
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float32).clamp(0, 1)
        return GenerationBatchResult(
            image_tensor=image_tensor,
            pil_images=[tensor_to_pil(image_tensor[index]) for index in range(image_tensor.shape[0])],
        )
