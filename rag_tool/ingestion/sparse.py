import io

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_tool.gcs_store import GCSStore

_VECTORIZER_BLOB = "sparse_vectorizer.joblib"


class SparseVectorizer:
    def __init__(self, vectorizer: TfidfVectorizer) -> None:
        self.vectorizer = vectorizer

    @classmethod
    def fit(cls, texts: list[str]) -> "SparseVectorizer":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
        )
        vectorizer.fit(texts or [""])
        return cls(vectorizer)

    @classmethod
    def load(cls, store: GCSStore) -> "SparseVectorizer":
        data = store.read_bytes(_VECTORIZER_BLOB)
        if data is None:
            raise FileNotFoundError("sparse_vectorizer.joblib not found in GCS.")
        return cls(joblib.load(io.BytesIO(data)))

    def save(self, store: GCSStore) -> None:
        buf = io.BytesIO()
        joblib.dump(self.vectorizer, buf)
        store.write_bytes(_VECTORIZER_BLOB, buf.getvalue())

    def transform(self, text: str) -> dict[str, list[float] | list[int]]:
        sparse_matrix = self.vectorizer.transform([text])
        row = sparse_matrix.tocoo()
        return {
            "values": [float(value) for value in row.data],
            "dimensions": [int(index) for index in row.col],
        }

