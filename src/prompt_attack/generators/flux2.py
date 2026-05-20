"""FLUX.2 generator adapter."""

from __future__ import annotations

from PIL import Image

from prompt_attack.attacks.soft_tokens import build_prompt, build_token_texts, validate_token_init_std
from prompt_attack.config import GeneratorConfig
from prompt_attack.generators.base import GenerationResult, LearnablePrompt
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

    def create_learnable_prompt(
        self,
        *,
        class_label: str,
        num_tokens: int,
        initializer: str,
        init_std: float,
    ) -> LearnablePrompt:
        """Create textual-inversion tokens initialized from an existing token."""
        import torch

        validate_token_init_std(init_std)
        token_texts = build_token_texts(num_tokens)
        token_ids = self._ensure_learnable_tokens(token_texts)
        initializer_embedding = self._initializer_embedding(initializer)
        values = initializer_embedding.to(device=torch.device(self.device), dtype=torch.float32).repeat(
            num_tokens,
            1,
        )
        values = values + torch.randn_like(values) * init_std
        prompt_state = LearnablePrompt(
            prompt_text=build_prompt(class_label, num_tokens),
            token_texts=token_texts,
            token_ids=token_ids,
            learnable_embeddings=torch.nn.Parameter(values),
        )
        self.sync_learnable_prompt(prompt_state)
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

    def _register_learnable_embedding_hook(self, prompt_state: LearnablePrompt):
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
                    value = replacements[index].view(*([1] * (output.ndim - 1)), output.shape[-1])
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
