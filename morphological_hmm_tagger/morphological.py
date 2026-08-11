"""Suffix-based morphological priors for OOV handling."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

try:
    from .data_loader import BOS_TOKEN, EOS_TOKEN, BOS_TAG, EOS_TAG
except ImportError:
    from data_loader import BOS_TOKEN, EOS_TOKEN, BOS_TAG, EOS_TAG


def normalize_word(word, lowercase=True):
    return word.lower() if lowercase else word


@dataclass
class SuffixMorphologyModel:
    """Learn tag distributions conditioned on word suffixes."""

    min_suffix_length: int = 1
    max_suffix_length: int = 4
    min_support: int = 2
    lowercase: bool = True
    prior_weight: float = 1.0
    suffix_tag_counts: dict = field(default_factory=dict)
    tag_totals: Counter = field(default_factory=Counter)
    tag_priors: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)

    def fit(self, sentences):
        """Fit suffix statistics from tagged sentences."""
        suffix_tag_counts = defaultdict(lambda: defaultdict(Counter))
        tag_totals = Counter()

        for sentence in sentences:
            for word, tag in sentence:
                if word in (BOS_TOKEN, EOS_TOKEN) or tag in (BOS_TAG, EOS_TAG):
                    continue
                normalized = normalize_word(word, self.lowercase)
                tag_totals[tag] += 1
                for suffix_length in range(self.min_suffix_length, self.max_suffix_length + 1):
                    if len(normalized) < suffix_length:
                        continue
                    suffix = normalized[-suffix_length:]
                    suffix_tag_counts[suffix_length][suffix][tag] += 1

        cleaned_suffix_counts = defaultdict(dict)
        for suffix_length, suffix_map in suffix_tag_counts.items():
            for suffix, counts in suffix_map.items():
                if sum(counts.values()) >= self.min_support:
                    cleaned_suffix_counts[suffix_length][suffix] = counts

        self.suffix_tag_counts = cleaned_suffix_counts
        self.tag_totals = tag_totals
        self.tags = sorted(tag_totals.keys())
        total_tag_count = sum(tag_totals.values())
        if total_tag_count > 0:
            self.tag_priors = {tag: tag_totals[tag] / total_tag_count for tag in self.tags}
        else:
            self.tag_priors = {tag: 0.0 for tag in self.tags}
        return self

    def suffix_counts_for_word(self, word):
        normalized = normalize_word(word, self.lowercase)
        for suffix_length in range(self.max_suffix_length, self.min_suffix_length - 1, -1):
            if len(normalized) < suffix_length:
                continue
            suffix = normalized[-suffix_length:]
            counts = self.suffix_tag_counts.get(suffix_length, {}).get(suffix)
            if counts:
                return counts
        return None

    def tag_distribution(self, word, tags=None):
        """Return a normalized P(tag | suffix) distribution for a token."""
        candidate_tags = list(tags) if tags is not None else list(self.tags)
        if not candidate_tags:
            return {}

        counts = self.suffix_counts_for_word(word)
        if counts is None:
            priors = {tag: self.tag_priors.get(tag, 1.0 / len(candidate_tags)) for tag in candidate_tags}
            total = sum(priors.values())
            if total <= 0:
                return {tag: 1.0 / len(candidate_tags) for tag in candidate_tags}
            return {tag: value / total for tag, value in priors.items()}

        scores = {}
        for tag in candidate_tags:
            base_count = counts.get(tag, 0)
            scores[tag] = base_count + self.prior_weight * self.tag_priors.get(tag, 0.0)

        total = sum(scores.values())
        if total <= 0:
            return {tag: 1.0 / len(candidate_tags) for tag in candidate_tags}
        return {tag: score / total for tag, score in scores.items()}