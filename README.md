# Sustainable Time Series Classification (anonymized for reviewers)

This repository accompanies our research paper on **Pruning Extensions and Efficiency Trade-Offs for Sustainable Time Series Classification**, which is currently under review.
All code and results were anonymized for reviewing.

In short, our work systematically explores predictive performance, runtime, and energy consumption in TSC across models, datasets, and hardware setups. As contributions, we

- Propose a **holistic evaluation framework** for comparing quality vs. resources in TSC
- Apply a **theoretically bounded pruning strategy** to state-of-the-art hybrid classifiers `Hydra` and `Quant`
- Introduce the novel `Hydrant` model variant, combining both feature transformations
- Conduct **4000+ experimental runs** across:
  - 20 MONSTER datasets  
  - 13 TSC methods  
  - 3 compute environments

## 📂 Repository Structure

```
.
├── results/                   # Experimental logs and figures
├── tsc/                       # Code for running experiments
├── run_analysis.py            # Analysis & exploration (based on STREP)
├── run_deep_train.sh          # Train deep learning models
├── run_deep_eval.sh           # Evaluate deep learning models
├── run_hybrid_train.sh        # Train hybrid and pruned models
├── run_hybrid_eval.sh         # Evaluate hybrid and pruned models
├── requirements.txt           # Python dependencies
└── README.md
```

## 👨‍💻 Usage

Install dependencies (tested with Python 3.12):

```bash
pip install -r requirements.txt
```

### 📊 Analysis & Interactive Exploration

You can re-run the analysis based on our experiment logs. For reproducing the paper figures and exploring TSC trade-offs via the [STREP exploration tool](https://github.com/raphischer/strep), simply run:

```bash
python run_analysis.py
```

### 🚀 Run TSC Experiments

Our experiments are streamlined with [MLflow](https://mlflow.org/), so running them requires an [Anaconda](https://www.anaconda.com/) environment with the dependencies from our `requirements.txt`. 
You can then use MLflow to run individual experiments, which will automatically create another Conda environment with the necessary [experiment libraries](tsc/conda.yaml).
As such, to train and evaluate a TSC model for a specific configuration, simply call

```bash
mlflow run -e main.py ./tsc
```

In this run command, you can also use the `-P` syntax to adjust the configuration parameters of our [main.py script](tsc/main.py), for example to evaluate the `P80Hydra` model on the main GPU and `Pedestrian` data:

```bash
mlflow run -e main.py -P gpu=0 -P dataset=Pedestrian -P model=Hydra -P prune_rate=0.8 ./tsc
```

#### Additional Setup
1. For accessing the [MONSTER](https://huggingface.co/monster-monash) datasets, you might need to log in to Hugging Face on your machine. In the newly created experiment environment, run `huggingface-cli login`.
2. Energy tracking via [CodeCarbon and ground-truth Smart Sockets](https://github.com/lamarr-institute/lamarr-energy-tracker) is included in our code, but remember that [CodeCarbon](https://github.com/mlco2/codecarbon) might require [special permissions for measuring CPU energy via RAPL](https://docs.codecarbon.io/latest/introduction/rapl/).
3. Some models like MLP and MCDCNN were observed to occasionally crash for certain configurations. One central issue is the handling of `nan` probabilities, which can be manually fixed by changing lines 78-82 in `[conda_env_dir]/lib/python3.12/site-packages/sktime/classification/deep_learning/base.py` to

```python
return np.array([
        self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))] if np.all(~np.isnan(prob)) else 0 # always predict first class for errors
        for prob in probs
    ]
)
```

### Run All Experiments

You can use our bash scripts to run all experimental configurations. They also iteratively create a experiment summary, similar to the csv files in our [results](results/) directory. Make sure to provide the correct command-line parameters for your setup, as they are passed to our [main.py script](tsc/main).

```bash
bash run_deep_train.sh [gpu_id] [data_dir] [results_dir] # trains standard and special DL classifiers
bash run_hybrid_train.sh [gpu_id] [data_dir] [results_dir] # trains Quant, Hydra, Hydrant and pruned variants
```

After training, you can evaluate the trained classifiers with different batch sizes, pointing to the `csv_summary` files created by the train scripts:

```bash
bash run_deep_eval.sh [gpu_id] [data_dir] [results_dir] [deep_csv_summary] # evaluates standard and special DL classifiers
bash run_hybrid_eval.sh [gpu_id] [data_dir] [results_dir] [pruned_csv_summary] # evaluates Quant, Hydra, Hydrant and pruned variants
```

Code and results © by authors of the paper under review