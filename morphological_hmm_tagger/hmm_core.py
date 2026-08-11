"""First-order HMM with smoothing and suffix-based OOV handling."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import log

try:
    from .data_loader import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, BOS_TAG, EOS_TAG
    from .morphological import SuffixMorphologyModel
    from .smoothing import add_alpha_probability, good_turing_probability
except ImportError:
    from data_loader import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, BOS_TAG, EOS_TAG
    from morphological import SuffixMorphologyModel
    from smoothing import add_alpha_probability, good_turing_probability


NEG_INF = float("-inf")


def safe_log(value):
    return log(value) if value > 0 else NEG_INF


def strip_boundaries(sentence):
    if not sentence:
        return []

    first_item = sentence[0]
    if isinstance(first_item, tuple):
        words = [word for word, _ in sentence]
    else:
        words = list(sentence)

    return [word for word in words if word not in (BOS_TOKEN, EOS_TOKEN)]


@dataclass
class FirstOrderHMMTagger:
    """Bigram HMM tagger using configurable smoothing and OOV strategies."""

    smoothing: str = "add_alpha"
    alpha: float = 0.1
    oov_strategy: str = "uniform"
    suffix_min_length: int = 1
    suffix_max_length: int = 4
    suffix_min_support: int = 2
    suffix_prior_weight: float = 1.0
    tags: list = field(default_factory=list)
    state_tags: list = field(default_factory=list)
    vocabulary: set = field(default_factory=set)
    tag_totals: Counter = field(default_factory=Counter)
    transition_counts: Counter = field(default_factory=Counter)
    transition_totals: Counter = field(default_factory=Counter)
    emission_counts: dict = field(default_factory=lambda: defaultdict(Counter))
    tag_priors: dict = field(default_factory=dict)
    morphology_model: object = None

    def fit(self, train_sents, train_vocab=None):
        """Estimate transition and emission statistics from tagged sentences."""
        self.transition_counts = Counter()
        self.transition_totals = Counter()
        self.emission_counts = defaultdict(Counter)
        self.tag_totals = Counter()

        vocabulary = set(train_vocab) if train_vocab is not None else set()

        for sentence in train_sents:
            if not sentence:
                continue
            previous_tag = BOS_TAG
            for word, tag in sentence:
                if word in (BOS_TOKEN, EOS_TOKEN):
                    continue
                vocabulary.add(word)
                self.emission_counts[tag][word] += 1
                self.tag_totals[tag] += 1
                self.transition_counts[(previous_tag, tag)] += 1
                self.transition_totals[previous_tag] += 1
                previous_tag = tag

            self.transition_counts[(previous_tag, EOS_TAG)] += 1
            self.transition_totals[previous_tag] += 1

        self.vocabulary = set(vocabulary)
        self.vocabulary.add(UNK_TOKEN)
        self.vocabulary.update({BOS_TOKEN, EOS_TOKEN})

        self.state_tags = sorted(self.tag_totals.keys())
        self.tags = [BOS_TAG] + self.state_tags + [EOS_TAG]

        total_tag_count = sum(self.tag_totals.values())
        if total_tag_count > 0:
            self.tag_priors = {tag: self.tag_totals[tag] / total_tag_count for tag in self.state_tags}
        else:
            if self.state_tags:
                self.tag_priors = {tag: 1.0 / len(self.state_tags) for tag in self.state_tags}
            else:
                self.tag_priors = {}

        if self.oov_strategy == "morphological":
            self.morphology_model = SuffixMorphologyModel(
                min_suffix_length=self.suffix_min_length,
                max_suffix_length=self.suffix_max_length,
                min_support=self.suffix_min_support,
                prior_weight=self.suffix_prior_weight,
            ).fit(train_sents)
        else:
            self.morphology_model = None

        return self

    def transition_probability(self, previous_tag, next_tag):
        support_size = len(self.state_tags) + 1
        count = self.transition_counts.get((previous_tag, next_tag), 0)
        total = self.transition_totals.get(previous_tag, 0)

        if self.smoothing == "good_turing":
            distribution = Counter({
                next_state_tag: self.transition_counts.get((previous_tag, next_state_tag), 0)
                for next_state_tag in self.state_tags + [EOS_TAG]
            })
            return good_turing_probability(count, distribution, support_size)

        return add_alpha_probability(count, total, support_size, self.alpha)

    def known_emission_probability(self, tag, word):
        count = self.emission_counts[tag].get(word, 0)
        total = self.tag_totals.get(tag, 0)
        support_size = max(len(self.vocabulary) - 2, 1)

        if self.smoothing == "good_turing":
            distribution = Counter(self.emission_counts[tag])
            distribution[UNK_TOKEN] = distribution.get(UNK_TOKEN, 0)
            return good_turing_probability(count, distribution, support_size)

        return add_alpha_probability(count, total, support_size, self.alpha)

    def oov_emission_probability(self, tag, word):
        base_unknown_probability = self.known_emission_probability(tag, UNK_TOKEN)
        if self.oov_strategy != "morphological" or self.morphology_model is None:
            return base_unknown_probability

        suffix_distribution = self.morphology_model.tag_distribution(word, self.state_tags)
        morphology_probability = suffix_distribution.get(tag, 0.0)
        interpolated = 0.75 * morphology_probability + 0.25 * base_unknown_probability
        return max(interpolated, 1e-12)

    def emission_probability(self, tag, word):
        if tag not in self.state_tags:
            return 0.0

        if word in (BOS_TOKEN, EOS_TOKEN):
            return 0.0

        if word in self.vocabulary and word != UNK_TOKEN:
            return max(self.known_emission_probability(tag, word), 1e-12)

        return max(self.oov_emission_probability(tag, word), 1e-12)

    def decode(self, sentence):
        """Tag a sentence using log-space Viterbi decoding."""
        words = strip_boundaries(sentence)
        if not words:
            return []

        if not self.state_tags:
            raise ValueError("The tagger must be fitted before decoding.")

        viterbi = []
        backpointers = []

        first_column = {}
        first_backpointer = {}
        first_word = words[0]
        for tag in self.state_tags:
            transition_prob = self.transition_probability(BOS_TAG, tag)
            emission_prob = self.emission_probability(tag, first_word)
            first_column[tag] = safe_log(transition_prob) + safe_log(emission_prob)
            first_backpointer[tag] = BOS_TAG
        viterbi.append(first_column)
        backpointers.append(first_backpointer)

        for position in range(1, len(words)):
            current_word = words[position]
            current_column = {}
            current_backpointer = {}

            for current_tag in self.state_tags:
                emission_prob = self.emission_probability(current_tag, current_word)
                best_score = NEG_INF
                best_previous_tag = self.state_tags[0]

                for previous_tag in self.state_tags:
                    previous_score = viterbi[-1][previous_tag]
                    transition_prob = self.transition_probability(previous_tag, current_tag)
                    score = previous_score + safe_log(transition_prob) + safe_log(emission_prob)
                    if score > best_score:
                        best_score = score
                        best_previous_tag = previous_tag

                current_column[current_tag] = best_score
                current_backpointer[current_tag] = best_previous_tag

            viterbi.append(current_column)
            backpointers.append(current_backpointer)

        best_final_score = NEG_INF
        best_final_tag = self.state_tags[0]
        for tag in self.state_tags:
            final_score = viterbi[-1][tag] + safe_log(self.transition_probability(tag, EOS_TAG))
            if final_score > best_final_score:
                best_final_score = final_score
                best_final_tag = tag

        predicted_tags = [best_final_tag]
        for position in range(len(words) - 1, 0, -1):
            best_final_tag = backpointers[position][best_final_tag]
            predicted_tags.append(best_final_tag)

        predicted_tags.reverse()
        return predicted_tags

    def predict(self, sentences):
        """Predict tags for a collection of sentences."""
        return [self.decode(sentence) for sentence in sentences]