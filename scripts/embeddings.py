"""Local text embeddings through onnxruntime — no network, no API key.

The descriptions written by the vision model are embedded locally, so searching
an archive costs nothing and works offline. Any sentence-embedding model exported
to ONNX with a `sentence_embedding` output will do; EMBEDDING_MODEL_DIR must
contain `model_quantized.onnx` and `tokenizer.json` (see README).

Two things are easy to get wrong here:

* pooling — the exported graph already returns a pooled sentence vector, so no
  manual mean pooling over token states is needed;
* asymmetry — retrieval models expect different text prefixes for a query and
  for a document. Embedding a query as if it were a document measurably degrades
  results, which is what `kind` is for.
"""
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

import config

MODEL_DIR = config.EMBEDDING_MODEL_DIR
MODEL_FILE = "model_quantized.onnx"
TOKENIZER_FILE = "tokenizer.json"
BATCH_SIZE = 8

_session = None
_tokenizer = None
_dimensions = None


def _load():
    global _session, _tokenizer
    if _session is None:
        if not model_available():
            raise FileNotFoundError(
                f"embedding model not found in {MODEL_DIR} - expected "
                f"{MODEL_FILE} and {TOKENIZER_FILE}; see README, 'Semantic search'"
            )
        _session = ort.InferenceSession(
            str(MODEL_DIR / MODEL_FILE), providers=["CPUExecutionProvider"]
        )
        _tokenizer = Tokenizer.from_file(str(MODEL_DIR / TOKENIZER_FILE))
    return _session, _tokenizer


def apply_prompt(text, kind):
    """Prefix a string the way the retrieval model expects it."""
    if kind == "query":
        return f"task: search result | query: {text}"
    return f"title: none | text: {text}"


def dimensions():
    """Embedding width, read from the loaded model rather than assumed."""
    global _dimensions
    if _dimensions is None:
        session, _ = _load()
        for output in session.get_outputs():
            if output.name == "sentence_embedding":
                shape = output.shape
                if len(shape) >= 2 and isinstance(shape[-1], int):
                    _dimensions = shape[-1]
                    break
        if _dimensions is None:
            _dimensions = int(embed(["probe"]).shape[1])
    return _dimensions


def embed(texts, kind="document", batch_size=BATCH_SIZE):
    """Embed a list of strings -> np.ndarray of shape (len(texts), dimensions).

    `kind` is "document" when indexing and "query" when searching.
    """
    if not texts:
        return np.zeros((0, dimensions()), dtype=np.float32)

    session, tokenizer = _load()
    prompted = [apply_prompt(t, kind) for t in texts]

    vectors = []
    for i in range(0, len(prompted), batch_size):
        chunk = prompted[i : i + batch_size]
        encodings = tokenizer.encode_batch(chunk)
        max_len = max(len(e.ids) for e in encodings)
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        for j, e in enumerate(encodings):
            n = len(e.ids)
            input_ids[j, :n] = e.ids
            attention_mask[j, :n] = e.attention_mask
        out = session.run(
            ["sentence_embedding"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        vectors.append(out[0])
    return np.concatenate(vectors, axis=0)


def model_available():
    return (MODEL_DIR / MODEL_FILE).exists() and (MODEL_DIR / TOKENIZER_FILE).exists()
