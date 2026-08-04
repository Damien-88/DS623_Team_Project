"""
Corpus parsing, vocabulary extraction, boundary padding, and OOV dataset 
splitting for the Morphological HMM Tagger.
"""

import random
from collections import Counter
from typing import List, Tuple, Dict, Set, Optional
import nltk


# Ensure boundary tokens are standardized
BOS_TOKEN = "<BOS>"  # Beginning of sentence
EOS_TOKEN = "<EOS>"  # End of sentence
UNK_TOKEN = "<UNK>"  # Out-of-vocabulary fallback symbol

BOS_TAG = "<BOS_TAG>" # Beginning of sentence tag
EOS_TAG = "<EOS_TAG>" # End of sentence tag


class CorpusDataLoader:
    """
    Handles downloading, loading, filtering, and splitting POS-tagged corpora 
    (NLTK Brown Corpus & Universal Dependencies) with custom OOV partitioning.
    """

    def __init__(self, dataset_name = "brown", universal_tagset = True):
        """
        Initializes the CorpusDataLoader with the specified dataset and tagset 
        preference.
        """
        # Normalize dataset name
        self.dataset_name = dataset_name.lower()
        # Use universal tagset if True, else use original tags
        self.universal_tagset = universal_tagset
        # List of sentences, each sentence is a list of (word, tag) tuples
        self.sentences: List[List[Tuple[str, str]]] = []


    def load_corpus(self, filepath = None):
        """
        Loads and standardizes sentences with (word, tag) pairs.
        Adds explicit <BOS> and <EOS> sentence boundary tags.
        """
        # Check for supported datasets and load accordingly
        if self.dataset_name == "brown":
            # Ensure the Brown corpus is available, downloading if necessary
            try:
                nltk.data.find("corpora/brown")
            except LookupError:
                nltk.download("brown")

            # Ensure universal tagset mappings are present when requested
            if self.universal_tagset:
                try:
                    nltk.data.find("taggers/universal_tagset")
                except LookupError:
                    nltk.download("universal_tagset")

            # Use universal tagset if specified, else use original tags
            tagset = "universal" if self.universal_tagset else None
            # Load tagged sentences from the Brown corpus
            raw_sents = nltk.corpus.brown.tagged_sents(tagset=tagset)

            # Standardize and pad sentence boundaries
            self.sentences = [
                [(BOS_TOKEN, BOS_TAG)] + [(word, tag) for word, tag in sent] + 
                [(EOS_TOKEN, EOS_TAG)] for sent in raw_sents if len(sent) > 0
            ]

        # Handle Universal Dependencies dataset loading
        elif self.dataset_name == "ud":
            # Validate that a file path is provided for the UD dataset
            if not filepath:
                raise ValueError("A valid file path to a .conllu file must be provided for 'ud'.")
            # Attempt to import the 'conllu' package for parsing .conllu files
            try:
                import conllu
            except ImportError:
                raise ImportError("Please ensure 'conllu' package is installed: pip install conllu")

            # Read and parse the .conllu file, standardizing sentences
            with open(filepath, "r", encoding="utf-8") as f:
                parsed_sents = conllu.parse_incr(f)
                self.sentences = []

                # Iterate through parsed sentences and extract (word, tag) pairs
                for token_list in parsed_sents:
                    sent = [(BOS_TOKEN, BOS_TAG)] # Start with beginning of sentence token

                    # Iterate through tokens in the sentence
                    for token in token_list:
                        word = token["form"] # Get word form
                        tag = token["upos"] if self.universal_tagset else token["xpos"] # Get POS tag

                        # Only append valid (word, tag) pairs
                        if word and tag:
                            sent.append((word, tag))

                    # Append end of sentence token and add to sentences list if valid        
                    sent.append((EOS_TOKEN, EOS_TAG))
                    if len(sent) > 2:  # Avoid empty sentences
                        self.sentences.append(sent)

        # Handle unsupported dataset names
        else:
            raise ValueError(f"Unsupported dataset: '{self.dataset_name}'. Choose 'brown' or 'ud'.")

        return self.sentences # Return standardized sentences for potential further processing


    def create_oov_split(self, test_ratio = 0.2, oov_target_rate = 0.1, seed = 42, filepath = None):
        """
        Splits data into train and test sets, artificially ensuring a target 
        Out-of-Vocabulary (OOV) rate in test set by removing low-frequency words 
        from training.

        For dataset_name='ud', provide filepath to a .conllu file when corpus
        has not been loaded yet.
        """
        # Load corpus if not already loaded
        if not self.sentences:
            self.load_corpus(filepath=filepath)

        # Shuffle sentences to ensure randomness in train/test split
        random.seed(seed)
        shuffled_sents = list(self.sentences)
        random.shuffle(shuffled_sents)

        # Split into train and test sets based on specified ratio
        split_idx = int(len(shuffled_sents) * (1 - test_ratio))
        train_sents = shuffled_sents[:split_idx]
        test_sents = shuffled_sents[split_idx:]

        # Extract initial vocabulary from train set
        word_counts = Counter(
            word for sent in train_sents for word, _ in sent 
            if word not in (BOS_TOKEN, EOS_TOKEN)
        )

        # Pre-calculate test word frequencies ONCE (O(N_test) instead of O(V * N_test))
        test_word_counts = Counter(
            word for sent in test_sents for word, _ in sent 
            if word not in (BOS_TOKEN, EOS_TOKEN)
        )   

        # Calculate total number of tokens in test set (excluding boundary tokens)
        test_total_tokens = sum(test_word_counts.values())
        # Determine how many tokens we want to be OOV in test set
        target_oov_count = int(test_total_tokens * oov_target_rate)

        # Count tokens in test set are already OOV based on current training vocabulary
        current_oov_count = sum(
            count for word, count in test_word_counts.items()
            if word not in word_counts
        )

        # Filter out rare words to hit desired target OOV induction
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1])  # Ascending frequency
        words_to_remove = set()

        # Remove low-frequency words from training until we reach target OOV count
        for word, count in sorted_words:
            # Check if we've already reached or exceeded target OOV count
            if current_oov_count >= target_oov_count:
                break

            # Add word to the removal set and update current OOV count
            words_to_remove.add(word)
            # Use precomputed test counts to avoid rescanning test set per word
            current_oov_count += test_word_counts.get(word, 0)

        # Final active training vocabulary
        train_vocab = set(word_counts.keys()) - words_to_remove
        train_vocab.add(UNK_TOKEN)
        train_vocab.add(BOS_TOKEN)
        train_vocab.add(EOS_TOKEN)

        # Align train sentences with induced OOV setup by mapping removed words to UNK
        if words_to_remove:
            mapped_train_sents = []
            for sent in train_sents:
                mapped_sent = []
                for word, tag in sent:
                    if word in (BOS_TOKEN, EOS_TOKEN):
                        mapped_sent.append((word, tag))
                    elif word in words_to_remove:
                        mapped_sent.append((UNK_TOKEN, tag))
                    else:
                        mapped_sent.append((word, tag))
                mapped_train_sents.append(mapped_sent)
            train_sents = mapped_train_sents

        # Return the split datasets and the final training vocabulary
        return train_sents, test_sents, train_vocab


def get_vocabulary_stats(train_sents, test_sents, train_vocab):
    """
    Utility function calculating vocabulary coverage and actual OOV statistics.
    """
    # Calculate total tokens in test set (excluding boundary tokens)
    test_tokens = [
        word for sent in test_sents for word, _ in sent 
        if word not in (BOS_TOKEN, EOS_TOKEN)
    ]
    # Calculate total number of tokens in test set
    total_test_tokens = len(test_tokens)
    # Identify OOV tokens in test set based on training vocabulary
    oov_tokens = [w for w in test_tokens if w not in train_vocab]

    # Compile statistics into a dictionary
    stats = {
        "total_train_sentences": len(train_sents),
        "total_test_sentences": len(test_sents),
        "vocabulary_size": len(train_vocab),
        "total_test_tokens": total_test_tokens,
        "oov_token_count": len(oov_tokens),
        "oov_type_count": len(set(oov_tokens)),
        "actual_oov_rate": len(oov_tokens) / total_test_tokens if total_test_tokens > 0 else 0.0
    }

    return stats # Return statistics dictionary for further analysis or reporting


# Test the CorpusDataLoader functionality with NLTK Brown Corpus
if __name__ == "__main__":
    print("Testing data_loader.py on NLTK Brown Corpus...")
    loader = CorpusDataLoader(dataset_name="brown", universal_tagset=True)
    train, test, vocab = loader.create_oov_split(test_ratio=0.2, oov_target_rate=0.1)

    stats = get_vocabulary_stats(train, test, vocab)
    print("\n--- Corpus Dataset Statistics ---")
    for k, v in stats.items():
        print(f"{k}: {v if not isinstance(v, float) else f'{v:.2%}'}")