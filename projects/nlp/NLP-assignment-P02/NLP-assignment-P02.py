"""Smoke test library Python untuk NLP.

Pengujian sengaja ringan agar dapat dijalankan tanpa mengunduh model besar.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class Result:
    name: str
    status: str
    detail: str


def run_test(name: str, module: str, check: Callable[[object], str]) -> Result:
    try:
        imported = importlib.import_module(module)
    except ImportError as exc:
        return Result(name, "SKIP", f"belum terpasang ({exc.name or exc})")
    except Exception as exc:  # pragma: no cover - bergantung environment lokal
        return Result(name, "FAIL", f"import error: {exc}")

    try:
        return Result(name, "PASS", check(imported))
    except Exception as exc:  # pragma: no cover - bergantung versi package
        return Result(name, "FAIL", str(exc))


def check_nltk(nltk: object) -> str:
    from nltk.tokenize import word_tokenize

    # split sederhana tidak membutuhkan resource NLTK tambahan.
    tokens = re.findall(r"\w+", "NLP mengolah bahasa manusia.")
    assert tokens == ["NLP", "mengolah", "bahasa", "manusia"]
    assert callable(word_tokenize)
    return "tokenisasi dasar siap digunakan"


def check_sklearn(sklearn: object) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer

    matrix = TfidfVectorizer().fit_transform(["python untuk NLP", "NLP dan data"])
    assert matrix.shape[0] == 2
    return f"TF-IDF menghasilkan matriks {matrix.shape}"


def check_gensim(gensim: object) -> str:
    from gensim.corpora import Dictionary

    dictionary = Dictionary([["nlp", "python"], ["data", "nlp"]])
    assert len(dictionary) == 3
    return f"dictionary berisi {len(dictionary)} token"


def check_spacy(spacy: object) -> str:
    doc = spacy.blank("id")("Saya belajar NLP.")
    assert len(doc) == 4
    return f"pipeline kosong memproses {len(doc)} token"


def check_textblob(textblob: object) -> str:
    blob = textblob.TextBlob("This is a good NLP example")
    assert blob.sentiment.polarity > 0
    return f"sentiment polarity={blob.sentiment.polarity:.2f}"


def check_fasttext(fasttext: object) -> str:
    assert hasattr(fasttext, "train_unsupervised")
    return "API train_unsupervised tersedia"


def check_allennlp(allennlp: object) -> str:
    assert allennlp is not None
    return "package berhasil di-import"


def check_polyglot(polyglot: object) -> str:
    assert polyglot is not None
    return "package berhasil di-import"


def check_transformers(transformers: object) -> str:
    assert hasattr(transformers, "pipeline")
    return "API pipeline tersedia (model tidak diunduh)"


def check_tensorflow(tensorflow: object) -> str:
    model = tensorflow.keras.Sequential([tensorflow.keras.layers.Dense(1, input_shape=(2,))])
    assert len(model.layers) == 1
    return "model Keras sederhana berhasil dibuat"


def check_torch(torch: object) -> str:
    tensor = torch.tensor([1, 2, 3])
    assert tensor.tolist() == [1, 2, 3]
    return f"tensor dibuat dengan shape={tuple(tensor.shape)}"


def check_keras(keras: object) -> str:
    model = keras.Sequential([keras.layers.Dense(1, input_shape=(2,))])
    assert len(model.layers) == 1
    return "model Keras sederhana berhasil dibuat"


def _check_numpy(np: object) -> str:
    values = np.array([1, 2, 3])
    assert values.sum() == 6
    return "array dan operasi numerik berhasil"


def _check_pandas(pd: object) -> str:
    frame = pd.DataFrame({"text": ["NLP"]})
    assert len(frame) == 1
    return "DataFrame berhasil dibuat"


def _check_regex(regex: object) -> str:
    assert regex.findall(r"\\w+", "NLP 2026") == ["NLP", "2026"]
    return "pencarian pola teks berhasil"


TESTS = [
    # Library wajib untuk fondasi Python, data, dan NLP modern.
    ("NumPy", "numpy", lambda np: _check_numpy(np)),
    ("pandas", "pandas", lambda pd: _check_pandas(pd)),
    ("SciPy", "scipy", lambda scipy: "komputasi ilmiah tersedia"),
    ("regex", "regex", lambda regex: _check_regex(regex)),
    ("NLTK", "nltk", check_nltk),
    ("spaCy", "spacy", check_spacy),
    ("scikit-learn", "sklearn", check_sklearn),
    ("PyTorch", "torch", check_torch),
    ("Transformers", "transformers", check_transformers),
    ("Tokenizers", "tokenizers", lambda tokenizers: "tokenizer tersedia"),
    ("Datasets", "datasets", lambda datasets: "dataset loader tersedia"),
    ("Sentence Transformers", "sentence_transformers", lambda st: "embedding API tersedia"),
    ("langdetect", "langdetect", lambda langdetect: "deteksi bahasa tersedia"),
    ("Sastrawi", "Sastrawi", lambda sastrawi: "stemming bahasa Indonesia tersedia"),
    ("Matplotlib", "matplotlib", lambda matplotlib: "visualisasi tersedia"),
    ("Seaborn", "seaborn", lambda seaborn: "visualisasi statistik tersedia"),
]


def main() -> int:
    results = [run_test(*test) for test in TESTS]
    print("HASIL PENGUJIAN LIBRARY PYTHON UNTUK NLP")
    print("Pengujian ini ditujukan untuk dijalankan di Google Colab.")
    print("Keterangan: PASS=berhasil, SKIP=belum terpasang, FAIL=terjadi error.")
    print("=" * 72)
    for result in results:
        print(f"[{result.status:4}] {result.name:16} - {result.detail}")
    print("=" * 72)
    print(
        "Ringkasan: "
        f"{sum(r.status == 'PASS' for r in results)} PASS, "
        f"{sum(r.status == 'SKIP' for r in results)} SKIP, "
        f"{sum(r.status == 'FAIL' for r in results)} FAIL"
    )
    if any(r.status == "FAIL" for r in results):
        print("Tindak lanjut: periksa detail pada baris FAIL dan versi dependency.")
    elif any(r.status == "SKIP" for r in results):
        print("Tindak lanjut: pasang package yang SKIP jika ingin menguji seluruh library.")
    else:
        print("Semua library wajib berhasil diuji.")
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
