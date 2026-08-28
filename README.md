# Morphological HMM Tagger

> **Master's Research Project**  
> **Department:** School of Technology and Computing 
> **Institution:** City University of Seattle  
> **Project Title:** *Suffix-Based Morphological Priors for OOV Handling in First-Order HMM Taggers*

**Nikolaj Wochnik; Master of Science in Computer Science**
**Kimberly Thomas; Master of Science in Data Science**

### Overview
Out-of-Vocabulary (OOV) tokens present a critical challenge for First-Order Hidden Markov Models (HMMs) in \
Part-of-Speech (POS) tagging, as standard Maximum Likelihood Estimation assigns zero emission probability to unseen \
words. Traditional remedies rely on naive uniform probability smoothing, which fails to leverage lexical structure. This \
paper evaluates the efficacy of combining suffix-based morphological prior estimation with probability smoothing to \
improve OOV tagging accuracy. We implement a bigram HMM from scratch in Python, comparing baseline models against \
Add-$\alpha$ smoothing, Good-Turing estimation, and a morphological suffix analyzer evaluated on the Brown Corpus and \
Universal Dependencies (English) datasets. We hypothesize that incorporating character-level suffix priors will \
significantly mitigate emission sparsity, yielding higher accuracy on unknown tokens compared to standard uniform \
smoothing without increasing inference latency during Viterbi decoding.

### File Structure:
```
morphological_hmm_tagger/
│
├── data_loader.py        # Corpus parsing (UD & Brown), vocabulary extraction, & OOV splitting
├── hmm_core.py           # Log-space transition/emission matrices & Viterbi decoder
├── smoothing.py          # Add-α and Good-Turing smoothing helpers
├── morphological.py      # Suffix extraction & morphological prior calculation
├── evaluator.py          # Benchmark runner, accuracy metrics, & latency measurement
├── visualizer.py         # Seaborn/Matplotlib plot generation & LaTeX table formatting
├── __init__.py           # Package exports
└── demo.ipynb            # Interactive walkthrough, live testing & figure generator
```

### Implementation Notes
- The HMM core is implemented from scratch with explicit count estimation and log-space Viterbi decoding.
- OOV handling supports uniform fallback, smoothed fallback, and suffix-based morphological priors.
- Helper libraries are used only for corpus access, plotting, and notebook workflow; the model logic itself stays self-contained.
- The corpus loader can save and reload precomputed splits as JSON when you want to reuse the same experiment setup.
- Punctuation and symbol tokens are intentionally kept in the corpus rather than stripped by a preprocessing step; see [Implementation Updates](#implementation-updates-2026-08-28) for the rationale.

### Implementation Updates (2026-08-28)
A sanity check of the code against the paper surfaced two gaps between the described experimental design and the original implementation, both now fixed in `hmm_core.py` and `evaluator.py`:

- **Transition model was not held constant across conditions.** `transition_probability()` previously branched on each condition's `smoothing` setting (add-α vs. Good-Turing) and used different α values per benchmark config, so the transition matrix differed between OOV strategies even though the paper's Experimental Controls state it should not. Transitions now always use a fixed module-level constant (`TRANSITION_ALPHA = 0.1`), independent of the OOV strategy under test, isolating the OOV-handling method as the only variable that changes between conditions.
- **"Baseline HMM" was not literally MLE with a uniform fallback.** The old `oov_strategy="uniform"` path returned a smoothed, per-tag probability derived from `<UNK>`-token counts rather than a flat distribution, and known-word emissions were add-α smoothed (α=1.0) rather than raw MLE counts. `smoothing="mle"` now gives pure maximum-likelihood known-word emissions, and `oov_strategy="uniform"` now returns a literal $1/|\text{tags}|$ constant, matching this README's original formula table. The previous smoothed-fallback behavior (used by Add-α and Good-Turing) is preserved under the explicit `oov_strategy="smoothed"` value.

`demo.ipynb` was completed with a full interactive walkthrough: corpus loading, exploratory data checks (`head`, `tail`, `info`, `describe`, NaN/empty-token checks, tag distribution, sentence-length stats), the induced OOV split with vocabulary/OOV-rate inspection, a standalone suffix-prior demo, a single-sentence decode example, the four-strategy benchmark, result visualization, LaTeX table export, and a save/reload round-trip for the split artifact. `artifacts/` was added to `.gitignore` since the split-export cell writes a multi-megabyte JSON file.

Punctuation and symbol tokens (14 distinct marks tagged `.` under the Brown corpus's universal tagset: `! ' '' ( ) , -- . : ; ? [ ] ​\`\``) were reviewed and intentionally left in the pipeline rather than routed through a new `preprocessing.py`. Punctuation is a legitimate, closed-set POS category that the transition model and standard tagging-accuracy metrics depend on; stripping it would distort sentence-boundary transitions and make results incomparable to standard Brown/UD benchmarks.

### Mathematical Overview
1. First-Order HMM Parameter Estimation
- Transition Matrix ($A$): Bigram tag transitions $P(t_j \mid t_i) = \frac{C(t_i, t_j)}{C(t_i)}$
- Emission Matrix ($B$): In-vocabulary word emissions $P(w_k \mid t_i) = \frac{C(t_i, w_k)}{C(t_i)}$

2. Suffix-Based Morphological Priors
- For an unseen token $w_{\text{OOV}} \notin V$, we extract trailing character $L$-grams ($L \in \{1, 2, 3, 4\}$) \
and calculate suffix conditional distributions:

$$P(t_i \mid \text{suffix}_L) = \frac{C(t_i, \text{suffix}_L)}{C(\text{suffix}_L)}$$

The implementation then interpolates suffix priors with the smoothed unknown-token emission:

$$P(w_{\text{OOV}} \mid t_i) = \lambda \, P(t_i \mid \text{suffix}(w)) + (1-\lambda) \, P(\text{<UNK>} \mid t_i),\quad \lambda=0.75$$

3. Log-Space Viterbi Lattice Decoding
- To prevent numerical underflow over long sequences, exact decoding operates in log-probability space: \

$$\ln v_t(j) = \max_{i} \left[ \ln v_{t-1}(i) + \ln A_{i,j} \right] + \ln B_j(w_t)$$

### Quick Start Guide
1. Prerequisites and Installation
Ensure you have Python 3.9 or higher installed. Clone the repository and install dependencies: \

```
git clone [https://github.com/your-username/morphological_hmm_tagger.git](https://github.com/your-username/morphological_hmm_tagger.git)
cd morphological_hmm_tagger
pip install -r requirements.txt
```

2. Download NLTK Datasets
Initialize the NLTK Brown Corpus inside a Python shell:

```
import nltk
nltk.download('brown')
nltk.download('universal_tagset')
```

3. Running the Benchmark Suite
Run the full comparative evaluation across models directly from the command line:

```
python -m morphological_hmm_tagger.evaluator --dataset brown --test-ratio 0.2 --oov-target-rate 0.1
```

4. Save a Dataset Split
If you want to reuse the exact same split later, save the loader output once:

```python
from morphological_hmm_tagger.data_loader import CorpusDataLoader

loader = CorpusDataLoader(dataset_name="brown", universal_tagset=True)
train_sents, test_sents, train_vocab = loader.create_oov_split()
loader.save_oov_split("artifacts/brown_oov_split.json", train_sents, test_sents, train_vocab)
```

5. Load a Saved Split

```python
from morphological_hmm_tagger.data_loader import CorpusDataLoader

train_sents, test_sents, train_vocab = CorpusDataLoader.load_oov_split("artifacts/brown_oov_split.json")
```

6. Demo Notebook Walkthrough
Launch the interactive demo notebook to execute code step-by- step and inline-render paper artifacts:

```
jupyter notebook demo.ipynb
```

### Experimental Baselines and Comparisons
The benchmark suite evaluates four distinct OOV handling strategies under an identical transition model, tag set, training/testing data, and Viterbi decoder — only the emission-handling method for unseen words differs between conditions.

| Strategy | `smoothing` | `oov_strategy` | Emission for unseen word $w$ |
| --- | --- | --- | --- |
| Baseline HMM | `mle` | `uniform` | $P(w \mid t_i) = \dfrac{1}{\|\text{tags}\|}$ (literal, tag-independent) |
| Add-α (Laplace) | `add_alpha` | `smoothed` | $P(w \mid t_i) = \dfrac{\alpha}{C(t_i) + \alpha(\|V\|+1)}$ |
| Good-Turing | `good_turing` | `smoothed` | Re-estimated zero-count mass ($N_1/N$) |
| Morphological Prior (Proposed) | `add_alpha` | `morphological` | $0.75\,P(t_i \mid \text{suffix}(w)) + 0.25\,P(\text{<UNK>} \mid t_i)$ |

All four conditions estimate transition probabilities with the same fixed add-α smoothing ($\alpha=0.1$), matching the paper's Experimental Controls.

### Analysis and Interpretation
Running the four-strategy benchmark from `demo.ipynb` (2,000 training / 500 test sentences, capped for interactive speed — see `evaluator.py` for a full-corpus run) produced:

| model | overall_accuracy | known_token_accuracy | oov_token_accuracy | ms/sentence | ms/token |
| --- | --- | --- | --- | --- | --- |
| baseline_uniform | 0.8936 | 0.9352 | 0.5235 | 1.49 | 0.069 |
| add_alpha | 0.8970 | 0.9229 | 0.6664 | 1.59 | 0.074 |
| good_turing | 0.9158 | 0.9448 | 0.6571 | 28.78 | 1.338 |
| morphological | 0.9066 | 0.9237 | 0.7539 | 1.71 | 0.079 |

**OOV accuracy (primary metric).** The morphological prior gives the largest OOV-accuracy gain among the low-latency strategies: +23.0 points over the uniform baseline (0.524 → 0.754) and +8.8 points over Add-α (0.666 → 0.754). This supports the hypothesis that suffix priors carry OOV-relevant signal beyond what simple probability smoothing provides.

**Overall/known accuracy (secondary metric).** Good-Turing edges out the morphological model on overall and known-token accuracy (0.916 vs. 0.907), but at roughly 17–19x the decoding latency (28.8 ms/sentence vs. 1.7 ms/sentence) because its zero-count mass is recomputed on every emission lookup rather than cached. This trades away the "without increasing inference latency" half of the hypothesis, whereas the morphological model's latency stays within about 15% of the fastest baseline.

**Interpretation.** These results are consistent with the paper's central claim: suffix-based morphological priors improve OOV tagging accuracy more than uniform or Add-α smoothing, at a latency cost close to the cheapest baselines. Good-Turing is the only strategy that trades meaningfully higher latency for accuracy, and it does not surpass the morphological model on the primary OOV metric. Because these figures come from a subset of the Brown corpus used for interactive demo speed, the paper's reported figures should be reproduced from a full-corpus run via `python -m morphological_hmm_tagger.evaluator`.