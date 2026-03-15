from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import ModelBackendConfig
from ..exceptions import ModelBackendError
from ..integrations import import_langchain_core, import_torch, import_transformers
from ..utils import tokenize


@dataclass(slots=True)
class TokenizationResult:
    token_ids: list[int]
    tokens: list[str]
    token_count: int
    backend: str


@dataclass(slots=True)
class GenerationResult:
    text: str
    prompt: str
    generated_tokens: int
    backend: str
    model_family: str
    metadata: dict[str, Any]


class TransformersTextBackend:
    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config
        self._transformers = import_transformers() if config.use_transformers else None
        self._torch = import_torch() if config.use_torch else None
        self._langchain = import_langchain_core() if config.use_langchain else None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._pipeline: Any | None = None
        self._generation_config: Any | None = None
        self._model_family = "fallback"
        self._load()

    @property
    def available(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def model_family(self) -> str:
        return self._model_family

    def _load(self) -> None:
        if self._transformers is None:
            return
        try:
            self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(
                self.config.tokenizer_name_or_path,
                local_files_only=self.config.local_files_only,
                trust_remote_code=self.config.trust_remote_code,
            )
        except Exception:
            self._tokenizer = None
            return
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._generation_config = self._transformers.GenerationConfig(**self.config.generation)
        self._model, self._model_family = self._load_model()
        if self._model is not None:
            self._pipeline = self._build_pipeline()

    def _resolve_dtype(self):
        if self._torch is None:
            return None
        dtype_name = self.config.dtype.lower()
        mapping = {
            "float16": self._torch.float16,
            "fp16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "bf16": self._torch.bfloat16,
            "float32": self._torch.float32,
            "fp32": self._torch.float32,
        }
        return mapping.get(dtype_name, self._torch.float32)

    def _load_model(self) -> tuple[Any | None, str]:
        if self._transformers is None:
            return None, "fallback"
        torch_dtype = self._resolve_dtype()
        model_loaders = [
            ("seq2seq", self._transformers.AutoModelForSeq2SeqLM),
            ("causal_lm", self._transformers.AutoModelForCausalLM),
            ("encoder", self._transformers.AutoModel),
        ]
        for family, loader in model_loaders:
            try:
                model = loader.from_pretrained(
                    self.config.model_name_or_path,
                    local_files_only=self.config.local_files_only,
                    trust_remote_code=self.config.trust_remote_code,
                    torch_dtype=torch_dtype,
                )
                if hasattr(model, "eval"):
                    model.eval()
                return model, family
            except Exception:
                continue
        return None, "fallback"

    def _build_pipeline(self):
        if self._transformers is None or self._model is None or self._tokenizer is None:
            return None
        task_name = "text2text-generation" if self._model_family == "seq2seq" else "text-generation"
        try:
            return self._transformers.pipeline(
                task_name,
                model=self._model,
                tokenizer=self._tokenizer,
                device=self._resolve_pipeline_device(),
                trust_remote_code=self.config.trust_remote_code,
            )
        except Exception:
            return None

    def _resolve_pipeline_device(self) -> int:
        if self.config.device.startswith("cuda"):
            try:
                return int(self.config.device.split(":")[1])
            except Exception:
                return 0
        return -1

    def tokenize(self, text: str) -> TokenizationResult:
        if self._tokenizer is not None:
            encoded = self._tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.config.max_prompt_tokens,
                return_attention_mask=False,
            )
            token_ids = list(encoded["input_ids"])
            tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
            return TokenizationResult(
                token_ids=token_ids,
                tokens=tokens,
                token_count=len(token_ids),
                backend="transformers",
            )
        fallback_tokens = tokenize(text)
        fallback_ids = [sum(ord(char) for char in token) % 997 for token in fallback_tokens[: self.config.max_prompt_tokens]]
        return TokenizationResult(
            token_ids=fallback_ids,
            tokens=fallback_tokens[: self.config.max_prompt_tokens],
            token_count=min(len(fallback_tokens), self.config.max_prompt_tokens),
            backend="fallback",
        )

    def encode_hidden(self, text: str) -> list[float]:
        tokenized = self.tokenize(text)
        if not self.available or self._torch is None or self._model_family not in {"seq2seq", "causal_lm", "encoder"}:
            return [float(token_id % 97) / 96.0 for token_id in tokenized.token_ids[: self.config.hidden_size]]
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_tokens,
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1][0]
        pooled = hidden_state.mean(dim=0)
        values = pooled.tolist()
        if len(values) >= self.config.hidden_size:
            return values[: self.config.hidden_size]
        return values + [0.0 for _ in range(self.config.hidden_size - len(values))]

    def generate(self, prompt: str, max_new_tokens: int | None = None, stop_strings: list[str] | None = None) -> GenerationResult:
        stop_strings = stop_strings or []
        if self._pipeline is not None:
            try:
                outputs = self._pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens or self.config.generation.get("max_new_tokens", 96),
                    temperature=self.config.generation.get("temperature", 0.2),
                    top_p=self.config.generation.get("top_p", 0.9),
                    return_full_text=self._model_family == "causal_lm",
                    pad_token_id=getattr(self._tokenizer, "pad_token_id", None),
                )
                raw_text = outputs[0].get("generated_text") or outputs[0].get("summary_text") or ""
                generated = raw_text if self._model_family == "seq2seq" else raw_text[len(prompt) :].strip()
                text = self._apply_stop_strings(generated.strip(), stop_strings)
                return GenerationResult(
                    text=text,
                    prompt=prompt,
                    generated_tokens=len(self.tokenize(text).token_ids),
                    backend="transformers",
                    model_family=self._model_family,
                    metadata={"used_pipeline": True},
                )
            except Exception as exc:
                return self._fallback_generation(prompt, reason=type(exc).__name__)
        return self._fallback_generation(prompt, reason="pipeline_unavailable")

    def build_langchain_runnable(self):
        if self._pipeline is None or self._langchain is None:
            return None
        try:
            providers = __import__("langchain_community.llms", fromlist=["HuggingFacePipeline"])
            return providers.HuggingFacePipeline(pipeline=self._pipeline)
        except Exception:
            return None

    def _apply_stop_strings(self, text: str, stop_strings: list[str]) -> str:
        clipped = text
        for marker in stop_strings:
            if marker and marker in clipped:
                clipped = clipped.split(marker, 1)[0]
        return clipped.strip()

    def _fallback_generation(self, prompt: str, reason: str) -> GenerationResult:
        fallback_text = prompt.splitlines()[-1] if prompt.splitlines() else prompt
        fallback_text = fallback_text[: self.config.generation.get("max_new_tokens", 96)]
        return GenerationResult(
            text=fallback_text,
            prompt=prompt,
            generated_tokens=len(tokenize(fallback_text)),
            backend="fallback",
            model_family="fallback",
            metadata={"reason": reason},
        )

    def require_available(self) -> None:
        if not self.available:
            raise ModelBackendError(
                f"Unable to load transformers model {self.config.model_name_or_path}. "
                f"Check local weights or disable local_files_only."
            )
