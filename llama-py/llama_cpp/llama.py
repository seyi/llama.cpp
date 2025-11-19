"""
Core llama.cpp Python bindings using ctypes.

This module follows the pattern established in gguf-py/tests/test_quants.py,
using ctypes to directly call C functions from the llama.cpp shared library.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, List, Union
from dataclasses import dataclass


# Type aliases matching llama.h
llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32


def _find_library() -> Optional[Path]:
    """Find the llama shared library following the pattern from test_quants.py"""
    # Common library names
    lib_names = ["libllama.so", "libllama.dylib", "llama.dll", "libllama.dll"]

    # Search paths
    search_paths = [
        Path(__file__).parent.parent.parent / "build" / "bin",
        Path(__file__).parent.parent.parent / "build" / "lib",
        Path(__file__).parent.parent.parent / "build",
        Path.cwd() / "build" / "bin",
        Path.cwd() / "build" / "lib",
    ]

    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return lib_path

    return None


# C structure definitions matching llama.h
class llama_model_params(ctypes.Structure):
    """Model parameters structure matching llama.h:llama_model_params"""
    _fields_ = [
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("rpc_servers", ctypes.c_char_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
    ]


class llama_context_params(ctypes.Structure):
    """Context parameters structure matching llama.h:llama_context_params"""
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("logits_all", ctypes.c_bool),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("flash_attn", ctypes.c_int),
        ("no_perf", ctypes.c_bool),
    ]


class llama_batch(ctypes.Structure):
    """Batch structure for input tokens matching llama.h:llama_batch"""
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


class LlamaLibrary:
    """
    Wrapper for the llama.cpp shared library.

    This class follows the pattern from gguf-py/tests/test_quants.py:GGMLQuants,
    loading the shared library and defining function signatures.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        if lib_path is None:
            lib_path = _find_library()
            if lib_path is None:
                raise RuntimeError(
                    "Could not find llama shared library. "
                    "Please build llama.cpp with BUILD_SHARED_LIBS=ON"
                )

        self.lib = ctypes.CDLL(str(lib_path))
        self._setup_functions()

    def _setup_functions(self):
        """Setup function signatures following the ctypes pattern from test_quants.py"""

        # Backend initialization
        self.lib.ggml_backend_load_all.restype = None
        self.lib.ggml_backend_load_all.argtypes = []

        # Model functions
        self.lib.llama_model_default_params.restype = llama_model_params
        self.lib.llama_model_default_params.argtypes = []

        self.lib.llama_model_load_from_file.restype = ctypes.c_void_p
        self.lib.llama_model_load_from_file.argtypes = [
            ctypes.c_char_p,
            llama_model_params,
        ]

        self.lib.llama_model_free.restype = None
        self.lib.llama_model_free.argtypes = [ctypes.c_void_p]

        # Vocab functions
        self.lib.llama_model_get_vocab.restype = ctypes.c_void_p
        self.lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]

        # Tokenization functions
        self.lib.llama_tokenize.restype = ctypes.c_int32
        self.lib.llama_tokenize.argtypes = [
            ctypes.c_void_p,  # vocab
            ctypes.c_char_p,  # text
            ctypes.c_int32,   # text_len
            ctypes.POINTER(llama_token),  # tokens
            ctypes.c_int32,   # n_tokens_max
            ctypes.c_bool,    # add_special
            ctypes.c_bool,    # parse_special
        ]

        self.lib.llama_token_to_piece.restype = ctypes.c_int32
        self.lib.llama_token_to_piece.argtypes = [
            ctypes.c_void_p,  # vocab
            llama_token,      # token
            ctypes.c_char_p,  # buf
            ctypes.c_int32,   # length
            ctypes.c_int32,   # lstrip
            ctypes.c_bool,    # special
        ]

        # Context functions
        self.lib.llama_context_default_params.restype = llama_context_params
        self.lib.llama_context_default_params.argtypes = []

        self.lib.llama_init_from_model.restype = ctypes.c_void_p
        self.lib.llama_init_from_model.argtypes = [
            ctypes.c_void_p,
            llama_context_params,
        ]

        self.lib.llama_free.restype = None
        self.lib.llama_free.argtypes = [ctypes.c_void_p]

        # Batch functions
        self.lib.llama_batch_get_one.restype = llama_batch
        self.lib.llama_batch_get_one.argtypes = [
            ctypes.POINTER(llama_token),
            ctypes.c_int32,
        ]

        self.lib.llama_batch_free.restype = None
        self.lib.llama_batch_free.argtypes = [llama_batch]

        # Decode function
        self.lib.llama_decode.restype = ctypes.c_int32
        self.lib.llama_decode.argtypes = [
            ctypes.c_void_p,
            llama_batch,
        ]

        # Encode function (for encoder-decoder models)
        self.lib.llama_encode.restype = ctypes.c_int32
        self.lib.llama_encode.argtypes = [
            ctypes.c_void_p,
            llama_batch,
        ]

        # Model has encoder
        self.lib.llama_model_has_encoder.restype = ctypes.c_bool
        self.lib.llama_model_has_encoder.argtypes = [ctypes.c_void_p]

        # Logits functions
        self.lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.llama_get_logits_ith.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
        ]

        # Sampler functions
        self.lib.llama_sampler_chain_init.restype = ctypes.c_void_p
        self.lib.llama_sampler_chain_init.argtypes = [ctypes.c_void_p]

        self.lib.llama_sampler_chain_add.restype = None
        self.lib.llama_sampler_chain_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]

        self.lib.llama_sampler_init_greedy.restype = ctypes.c_void_p
        self.lib.llama_sampler_init_greedy.argtypes = []

        self.lib.llama_sampler_sample.restype = llama_token
        self.lib.llama_sampler_sample.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
        ]

        self.lib.llama_sampler_free.restype = None
        self.lib.llama_sampler_free.argtypes = [ctypes.c_void_p]

        # Model info
        self.lib.llama_n_vocab.restype = ctypes.c_int32
        self.lib.llama_n_vocab.argtypes = [ctypes.c_void_p]


# Global library instance (following the singleton pattern)
_llama_lib: Optional[LlamaLibrary] = None


def get_library(lib_path: Optional[Path] = None) -> LlamaLibrary:
    """Get or create the global library instance"""
    global _llama_lib
    if _llama_lib is None:
        _llama_lib = LlamaLibrary(lib_path)
        # Initialize backends
        _llama_lib.lib.ggml_backend_load_all()
    return _llama_lib


class LlamaModel:
    """
    Python wrapper for llama_model.

    Follows the pattern from the simple.cpp example and provides a Pythonic
    interface to the C API.

    Example:
        >>> model = LlamaModel.from_file("model.gguf", n_gpu_layers=32)
    """

    def __init__(self, model_ptr: int, lib: LlamaLibrary):
        self._model = model_ptr
        self._lib = lib

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        n_gpu_layers: int = 0,
        use_mmap: bool = True,
        use_mlock: bool = False,
        vocab_only: bool = False,
        lib_path: Optional[Path] = None,
    ) -> "LlamaModel":
        """
        Load a model from a GGUF file.

        Args:
            path: Path to the GGUF model file
            n_gpu_layers: Number of layers to offload to GPU
            use_mmap: Use memory mapping for model loading
            use_mlock: Lock model in RAM
            vocab_only: Only load vocabulary
            lib_path: Path to llama shared library (auto-detected if None)

        Returns:
            LlamaModel instance
        """
        lib = get_library(lib_path)

        # Get default parameters
        params = lib.lib.llama_model_default_params()
        params.n_gpu_layers = n_gpu_layers
        params.use_mmap = use_mmap
        params.use_mlock = use_mlock
        params.vocab_only = vocab_only

        # Load model
        model_ptr = lib.lib.llama_model_load_from_file(
            str(path).encode("utf-8"),
            params,
        )

        if model_ptr is None or model_ptr == 0:
            raise RuntimeError(f"Failed to load model from {path}")

        return cls(model_ptr, lib)

    def get_vocab(self) -> int:
        """Get the vocabulary pointer"""
        return self._lib.lib.llama_model_get_vocab(self._model)

    def tokenize(
        self,
        text: str,
        add_special: bool = True,
        parse_special: bool = True,
    ) -> List[int]:
        """
        Tokenize text.

        Args:
            text: Text to tokenize
            add_special: Add special tokens (BOS, etc.)
            parse_special: Parse special tokens in text

        Returns:
            List of token IDs
        """
        vocab = self.get_vocab()
        text_bytes = text.encode("utf-8")

        # First call to get token count (negative return value)
        n_tokens = -self._lib.lib.llama_tokenize(
            vocab,
            text_bytes,
            len(text_bytes),
            None,
            0,
            add_special,
            parse_special,
        )

        # Allocate buffer and tokenize
        tokens = (llama_token * n_tokens)()
        result = self._lib.lib.llama_tokenize(
            vocab,
            text_bytes,
            len(text_bytes),
            tokens,
            n_tokens,
            add_special,
            parse_special,
        )

        if result < 0:
            raise RuntimeError("Tokenization failed")

        return list(tokens[:result])

    def token_to_piece(self, token: int) -> str:
        """Convert token ID to text piece"""
        vocab = self.get_vocab()
        buf = ctypes.create_string_buffer(256)

        length = self._lib.lib.llama_token_to_piece(
            vocab,
            token,
            buf,
            len(buf),
            0,
            True,
        )

        if length < 0:
            raise RuntimeError("Token conversion failed")

        return buf.value[:length].decode("utf-8", errors="replace")

    def has_encoder(self) -> bool:
        """Check if model has an encoder (encoder-decoder architecture)"""
        return self._lib.lib.llama_model_has_encoder(self._model)

    def n_vocab(self) -> int:
        """Get vocabulary size"""
        return self._lib.lib.llama_n_vocab(self._model)

    def __del__(self):
        """Free model resources"""
        if hasattr(self, "_model") and self._model:
            self._lib.lib.llama_model_free(self._model)


class LlamaContext:
    """
    Python wrapper for llama_context.

    Example:
        >>> model = LlamaModel.from_file("model.gguf")
        >>> ctx = LlamaContext(model, n_ctx=2048)
    """

    def __init__(
        self,
        model: LlamaModel,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_threads: int = -1,
        flash_attn: bool = False,
    ):
        self._model = model
        self._lib = model._lib

        # Get default parameters
        params = self._lib.lib.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch

        if n_threads > 0:
            params.n_threads = n_threads
            params.n_threads_batch = n_threads

        params.flash_attn = 1 if flash_attn else 0
        params.no_perf = False

        # Initialize context
        self._ctx = self._lib.lib.llama_init_from_model(
            model._model,
            params,
        )

        if self._ctx is None or self._ctx == 0:
            raise RuntimeError("Failed to create context")

    def decode(self, batch: "LlamaBatch") -> None:
        """Decode a batch of tokens"""
        result = self._lib.lib.llama_decode(self._ctx, batch._batch)
        if result != 0:
            raise RuntimeError(f"Decode failed with code {result}")

    def encode(self, batch: "LlamaBatch") -> None:
        """Encode a batch (for encoder-decoder models)"""
        result = self._lib.lib.llama_encode(self._ctx, batch._batch)
        if result != 0:
            raise RuntimeError(f"Encode failed with code {result}")

    def get_logits(self, idx: int = -1) -> List[float]:
        """Get output logits for token at index"""
        logits_ptr = self._lib.lib.llama_get_logits_ith(self._ctx, idx)
        n_vocab = self._model.n_vocab()
        return [logits_ptr[i] for i in range(n_vocab)]

    def __del__(self):
        """Free context resources"""
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.lib.llama_free(self._ctx)


class LlamaBatch:
    """
    Python wrapper for llama_batch.

    Example:
        >>> tokens = [1, 2, 3, 4]
        >>> batch = LlamaBatch.from_tokens(tokens, model._lib)
    """

    def __init__(self, batch: llama_batch, lib: LlamaLibrary):
        self._batch = batch
        self._lib = lib

    @classmethod
    def from_tokens(cls, tokens: List[int], lib: LlamaLibrary) -> "LlamaBatch":
        """Create a batch from a list of tokens"""
        n_tokens = len(tokens)
        tokens_array = (llama_token * n_tokens)(*tokens)
        batch = lib.lib.llama_batch_get_one(tokens_array, n_tokens)
        return cls(batch, lib)


class LlamaSampler:
    """
    Python wrapper for llama_sampler.

    Example:
        >>> sampler = LlamaSampler.greedy(model._lib)
        >>> token = sampler.sample(ctx, -1)
    """

    def __init__(self, sampler_ptr: int, lib: LlamaLibrary):
        self._sampler = sampler_ptr
        self._lib = lib

    @classmethod
    def greedy(cls, lib: LlamaLibrary) -> "LlamaSampler":
        """Create a greedy sampler"""
        # Create chain
        chain_ptr = lib.lib.llama_sampler_chain_init(None)

        # Add greedy sampler
        greedy_ptr = lib.lib.llama_sampler_init_greedy()
        lib.lib.llama_sampler_chain_add(chain_ptr, greedy_ptr)

        return cls(chain_ptr, lib)

    def sample(self, ctx: LlamaContext, idx: int = -1) -> int:
        """Sample a token from the context logits"""
        token = self._lib.lib.llama_sampler_sample(
            self._sampler,
            ctx._ctx,
            idx,
        )
        return token

    def __del__(self):
        """Free sampler resources"""
        if hasattr(self, "_sampler") and self._sampler:
            self._lib.lib.llama_sampler_free(self._sampler)
