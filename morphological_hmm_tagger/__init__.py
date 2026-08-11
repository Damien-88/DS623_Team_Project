"""Morphological HMM tagger package."""

from .data_loader import CorpusDataLoader, get_vocabulary_stats
from .hmm_core import FirstOrderHMMTagger
from .morphological import SuffixMorphologyModel

EXPORTED_SYMBOLS = [
    "CorpusDataLoader",
    "FirstOrderHMMTagger",
    "SuffixMorphologyModel",
    "get_vocabulary_stats",
]