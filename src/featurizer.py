import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer



class CryptogramFeaturizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = []

        for text in X:
            text = text.upper()
            words = re.findall(r"[A-Z']+", text)
            n_chars = len(re.findall(r"[A-Z]", text))

            # --- Letter frequencies ---
            counts = Counter(text)
            freqs = np.array([counts.get(chr(ord('A')+i), 0) for i in range(26)])
            freqs = freqs / (n_chars if n_chars > 0 else 1)

            # Index of Coincidence
            ic = 0
            if n_chars > 1:
                ic = sum(v*(v-1) for v in freqs*n_chars) / (n_chars*(n_chars-1))

            # --- Word-level stats ---
            lengths = [len(w) for w in words]
            avg_len = np.mean(lengths) if lengths else 0
            var_len = np.var(lengths) if lengths else 0

            one_letter = sum(len(w) == 1 for w in words)
            two_letter = sum(len(w) == 2 for w in words)
            three_letter = sum(len(w) == 3 for w in words)

            # Ratios of short words
            n_words = len(words)
            one_ratio = one_letter / (n_words if n_words else 1)
            two_ratio = two_letter / (n_words if n_words else 1)
            three_ratio = three_letter / (n_words if n_words else 1)

            # Repetition and patterns
            unique_words = len(set(words))
            repetition_ratio = 1 - unique_words / (n_words if n_words else 1)
            patterned_ratio = sum(len(set(w)) < len(w) for w in words) / (n_words if n_words else 1)

            # Start / end duplication
            starts = Counter([w[0] for w in words if w])
            ends = Counter([w[-1] for w in words if w])
            start_dup = sum(v-1 for v in starts.values() if v > 1) / (n_words if n_words else 1)
            end_dup = sum(v-1 for v in ends.values() if v > 1) / (n_words if n_words else 1)

            # Bigram repetition
            bigrams = [text[i:i+2] for i in range(len(text)-1)]
            bigram_repeat_ratio = (len(bigrams) - len(set(bigrams))) / (len(bigrams) if bigrams else 1)

            # Trigram repetition
            trigrams = [text[i:i+3] for i in range(len(text)-2)]
            trigram_repeat_ratio = (len(trigrams) - len(set(trigrams))) / (len(trigrams) if trigrams else 1)

            features.append({
                "n_chars": n_chars,
                "n_words": n_words,
                "avg_word_len": avg_len,
                "var_word_len": var_len,

                # counts
                "one_letter_words": one_letter,
                "two_letter_words": two_letter,
                "three_letter_words": three_letter,

                # ratios
                "one_letter_ratio": one_ratio,
                "two_letter_ratio": two_ratio,
                "three_letter_ratio": three_ratio,

                "repetition_ratio": repetition_ratio,
                "patterned_words_ratio": patterned_ratio,
                "index_of_coincidence": ic,
                "start_dup_ratio": start_dup,
                "end_dup_ratio": end_dup,
                "bigram_repeat_ratio": bigram_repeat_ratio,
                "trigram_repeat_ratio": trigram_repeat_ratio
            })

        return pd.DataFrame(features)


class RepeatNgramCounter(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_ranges=[(1, 1), (2, 2), (3, 3)]):
        self.ngram_ranges = ngram_ranges
        self.vectorizers = [CountVectorizer(ngram_range=r) for r in ngram_ranges]

    def fit(self, X, y=None):
        X = [str(doc) for doc in X]  # ensure all inputs are strings
        for vec in self.vectorizers:
            vec.fit(X)
        return self

    def transform(self, X):
        X = [str(doc) for doc in X]  # ensure all inputs are strings
        features = []
        for vec in self.vectorizers:
            X_counts = vec.transform(X)
            # Count repeated n-grams in each document
            repeats = np.array((X_counts > 1).sum(axis=1)).flatten()
            features.append(repeats)
        # Return a dense 2D array of shape (n_docs, n_features)
        return np.column_stack(features)