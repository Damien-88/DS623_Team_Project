"""Model evaluation utilities and a small command-line benchmark runner."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

try:
    from .data_loader import BOS_TOKEN, EOS_TOKEN, CorpusDataLoader, get_vocabulary_stats
    from .hmm_core import FirstOrderHMMTagger
except ImportError:
    from data_loader import BOS_TOKEN, EOS_TOKEN, CorpusDataLoader, get_vocabulary_stats
    from hmm_core import FirstOrderHMMTagger


@dataclass
class BenchmarkResult:
    model_name: str
    overall_accuracy: float
    known_token_accuracy: float
    oov_token_accuracy: float
    total_tokens: int
    known_tokens: int
    oov_tokens: int
    decode_seconds: float
    milliseconds_per_sentence: float
    milliseconds_per_token: float


def sentence_words(sentence):
    return [word for word, _ in sentence if word not in (BOS_TOKEN, EOS_TOKEN)]


def sentence_tags(sentence):
    return [tag for word, tag in sentence if word not in (BOS_TOKEN, EOS_TOKEN)]


def evaluate_predictions(gold_sentences, predicted_tags, train_vocab):
    train_vocab = set(train_vocab)
    total_tokens = known_tokens = oov_tokens = 0
    correct_tokens = known_correct = oov_correct = 0

    for gold_sentence, predicted_sentence in zip(gold_sentences, predicted_tags):
        gold_tags = sentence_tags(gold_sentence)
        gold_words = sentence_words(gold_sentence)
        prediction_length = min(len(gold_tags), len(predicted_sentence))

        for index in range(prediction_length):
            word = gold_words[index]
            gold_tag = gold_tags[index]
            predicted_tag = predicted_sentence[index]

            total_tokens += 1
            is_oov = word not in train_vocab
            if is_oov:
                oov_tokens += 1
            else:
                known_tokens += 1

            if predicted_tag == gold_tag:
                correct_tokens += 1
                if is_oov:
                    oov_correct += 1
                else:
                    known_correct += 1

    overall_accuracy = correct_tokens / total_tokens if total_tokens else 0.0
    known_accuracy = known_correct / known_tokens if known_tokens else 0.0
    oov_accuracy = oov_correct / oov_tokens if oov_tokens else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "known_token_accuracy": known_accuracy,
        "oov_token_accuracy": oov_accuracy,
        "total_tokens": total_tokens,
        "known_tokens": known_tokens,
        "oov_tokens": oov_tokens,
    }


def benchmark_configuration(model_name, train_sentences, test_sentences, train_vocab, **tagger_kwargs):
    print(f"Running benchmark: {model_name}", flush=True)
    tagger = FirstOrderHMMTagger(**tagger_kwargs)
    tagger.fit(train_sentences, train_vocab=train_vocab)

    decode_start = perf_counter()
    predicted_tags = tagger.predict([sentence_words(sentence) for sentence in test_sentences])
    decode_seconds = perf_counter() - decode_start

    metrics = evaluate_predictions(test_sentences, predicted_tags, train_vocab)
    sentence_count = len(test_sentences)
    token_count = metrics["total_tokens"]

    return BenchmarkResult(
        model_name = model_name,
        overall_accuracy = metrics["overall_accuracy"],
        known_token_accuracy = metrics["known_token_accuracy"],
        oov_token_accuracy = metrics["oov_token_accuracy"],
        total_tokens = token_count,
        known_tokens = metrics["known_tokens"],
        oov_tokens = metrics["oov_tokens"],
        decode_seconds = decode_seconds,
        milliseconds_per_sentence = (decode_seconds / sentence_count * 1000.0) if sentence_count else 0.0,
        milliseconds_per_token = (decode_seconds / token_count * 1000.0) if token_count else 0.0,
    )


def benchmark_all(train_sentences, test_sentences, train_vocab, selected_models=None):
    configurations = [
        ("baseline_uniform", {"smoothing": "mle", "oov_strategy": "uniform"}),
        ("add_alpha", {"smoothing": "add_alpha", "alpha": 0.1, "oov_strategy": "smoothed"}),
        ("good_turing", {"smoothing": "good_turing", "oov_strategy": "smoothed"}),
        ("morphological", {"smoothing": "add_alpha", "alpha": 0.1, "oov_strategy": "morphological"}),
    ]

    if selected_models is not None:
        selected_models = set(selected_models)
        configurations = [item for item in configurations if item[0] in selected_models]

    results = []
    for model_name, kwargs in configurations:
        results.append(benchmark_configuration(model_name, train_sentences, test_sentences, train_vocab, **kwargs))
    return results


def results_to_frame(results):
    return pd.DataFrame([asdict(result) for result in results])


def run_from_command_line():
    parser = argparse.ArgumentParser(description="Benchmark the morphological HMM tagger.")
    parser.add_argument("--dataset", choices=["brown", "ud"], default="brown")
    parser.add_argument("--filepath", default=None, help="Path to a UD .conllu file when dataset='ud'.")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--oov-target-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-sentences", type=int, default=None, help="Optional cap on train sentences for a faster run.")
    parser.add_argument("--max-test-sentences", type=int, default=None, help="Optional cap on test sentences for a faster run.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional list of models to run: baseline_uniform, add_alpha, good_turing, morphological.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    loader = CorpusDataLoader(dataset_name=args.dataset, universal_tagset=True)
    train_sentences, test_sentences, train_vocab = loader.create_oov_split(
        test_ratio = args.test_ratio,
        oov_target_rate = args.oov_target_rate,
        seed = args.seed,
        filepath = args.filepath,
    )

    if args.max_train_sentences is not None:
        train_sentences = train_sentences[: args.max_train_sentences]
    if args.max_test_sentences is not None:
        test_sentences = test_sentences[: args.max_test_sentences]

    stats = get_vocabulary_stats(train_sentences, test_sentences, train_vocab)

    selected_models = set(args.models) if args.models else None
    results = benchmark_all(train_sentences, test_sentences, train_vocab, selected_models=selected_models)
    results_frame = results_to_frame(results)

    print("Corpus statistics:")
    print(json.dumps(stats, indent=2))
    print("\nBenchmark results:")
    print(results_frame.to_string(index=False))

    if args.output_json:
        Path(args.output_json).write_text(results_frame.to_json(orient="records", indent=2), encoding="utf-8")
    if args.output_csv:
        results_frame.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    run_from_command_line()