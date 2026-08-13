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
- OOV handling supports uniform fallback and suffix-based morphological priors.
- Helper libraries are used only for corpus access, plotting, and notebook workflow; the model logic itself stays self-contained.
- The corpus loader can save and reload precomputed splits as JSON when you want to reuse the same experiment setup.

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
The benchmark suite evaluates four distinct OOV handling strategies:

Strategy,Description,Emission for Unseen w
Baseline HMM,Uniform Fallback,P(w \mid t_i) = \frac{1}{\|V\
Add-α (Laplace),Fixed Pseudo-Count Smoothing,P(w \mid t_i) = \frac{\alpha}{C(t_i) + \alpha(\|V\|+1
Good-Turing,Discounting based on singletons (N1​/N),Re-adjusted zero-count frequency
Morphological Prior (Proposed),Suffix-to-Tag Character Distribution,P(suffixL​∣ti​) via Bayes Inversion