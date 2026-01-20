# On Temperature-Constrained Non-Deterministic Machine Translation: Potential and Evaluation

This repository contains the official implementation and evaluation data for the paper **"On Temperature-Constrained Non-Deterministic Machine Translation: Potential and Evaluation"**.

## 📄 Description

This study investigates Temperature-Constrained Non-Deterministic Machine Translation (ND-MT), a phenomenon emerging from modern Large Language Models (LLMs). While ND-MT offers higher potential quality and better handling of multi-modality compared to traditional Deterministic MT (D-MT), the authors find that current evaluation frameworks are unreliable for these stochastic systems. Through a systematic evaluation of five state-of-the-art systems, the paper identifies a "Buckets Effect": a system's overall ranking is consistently determined by its worst-quality candidate rather than its best or average output. To address evaluation inconsistencies, the authors propose a new strategy called ExpectoSample.
## 🌟 Key Features

* **ND-MT Generation Framework**: A unified pipeline to generate temperature-constrained translations using SOTA LLMs (Llama3, Qwen2.5, Qwen3, DeepSeek) and specialized models (NLLB, mBART).
* **Comprehensive Evaluation**: Support for both lexical-based (BLEU, METEOR, ROUGE) and semantic-based (COMET, BLEURT, BERTScore) metrics.
* **ExpectoSample Strategy**: You can utilize the evaluation results to implement our proposed strategy to assess metric reliability in non-deterministic settings.
* **Multi-Direction Support**: Native support for 6 translation directions, including ZH↔EN, EN↔DE, and EN↔RU.


## 🛠️ Installation

```bash
git clone https://github.com/yourusername/TC-DN-MT.git
cd TC-DN-MT
pip install -r requirements.txt

```

*Note: Ensure you have `torch` installed with CUDA support for GPU acceleration.*

## 🚀 Usage

### 1. Translation Generation

Use `sampling_runs.py` to generate candidates under different temperature constraints. The script supports both **Instruct Mode** (for chat models) and **Pretrain Mode** (few-shot).

```bash
python sampling_runs.py \
    --input_name <dataset_name>_<model_name> \
    --sampling_size <num_samples> \
    --temp <temperature> \
    --model_path <path_to_model> \
    --dataset_path ./data \
    --storage_path <output_path> \
    [--greedy] \
    [--cache_dir <cache_directory>]
```

**Example**:
```bash
# Generate 10 candidate translations with temperature sampling
python sampling_runs.py \
    --input_name 23en-zh_qwen3_chat \
    --sampling_size 10 \
    --temp 0.5 \
    --model_path /path/to/qwen3 \
    --dataset_path ./data \
    --storage_path ./translation_results

# Use greedy decoding
python sampling_runs.py \
    --input_name 23en-zh_qwen3_chat \
    --greedy \
    --model_path /path/to/qwen3 \
    --dataset_path ./data \
    --storage_path ./translation_results
```

### 2. Metric Evaluation

Calculate lexical and semantic scores for the generated hypotheses.

```bash
# Lexical Metrics (BLEU, METEOR, ROUGE, TER, chrF++, GLVS, etc.)
python before_lexical.py \
    --input_folder <input_folder> \
    --output_folder <output_folder> \
    --device cuda:0 \
    --skip_existing True

# Semantic Metrics (COMET, COMETKIWI, LASER, LaBSE, XNLI, SONAR, BLASER, etc.)
python before_semantic.py \
    --input <input_file_or_folder> \
    --output <output_folder> \
    --step <num_candidates_per_sample>
```

### 3. Batch Evaluation

Use the provided bash scripts for batch evaluation:

```bash
# Batch lexical evaluation
bash before_lexical.bash

# Batch semantic evaluation
bash semantic.bash
```

**Note**: Modify the paths and GPU device configurations in the bash scripts according to your setup.

## 📂 Repository Structure

```text
.
├── data/                   # Dataset files (WMT 2023/2024 data)
│   ├── 23en-zh.en/.zh     # English-Chinese datasets
│   ├── 23zh-en.zh/.en     # Chinese-English datasets
│   ├── 23en-de.en/.de     # English-German datasets
│   ├── 23en-ru.en/.ru     # English-Russian datasets
│   └── ...                # Other language pairs
├── sampling_runs.py        # Main translation generation script
├── before_lexical.py       # Lexical evaluation script (BLEU, METEOR, ROUGE, etc.)
├── before_semantic.py      # Semantic evaluation script (COMET, LASER, etc.)
├── semantic_tools.py       # Semantic evaluation utilities and metrics
├── before_lexical.bash     # Batch lexical evaluation script
├── semantic.bash           # Batch semantic evaluation script
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```


## 🧠 Supported Models

* **LLMs:** Llama-3, Qwen-2.5, Qwen-3 (Chain-of-Thought support), DeepSeek-Llama/qwen, etc.
* **Traditional MT:** NLLB (No Language Left Behind), mBART.
* **Modes:** Greedy Decoding, Temperature Sampling, 4-bit Quantization.

## 📉 Evaluation Data & Collaboration

We are releasing our full evaluation results to facilitate further research into the non-deterministic nature of modern machine translation. We believe that community collaboration is essential to addressing the challenges of ND-MT evaluation.

**🔗 [Join our Research Network / Access Data](https://forms.gle/vNyvrvwUgU9NkUp59)**

If you are interested in:
* Accessing the raw evaluation logs and generated candidates.
* Contributing new models or metrics to the benchmark.
* Discussing potential collaborations on ND-MT research.

Please fill out the form above. We are actively looking for partners to expand the **ND-MT** research to more languages and domains.

## 📜 Citation

If you use this code or findings in your research, please cite our paper:

```bibtex
@article{YourName2024NDMT,
  title={On Temperature-Constrained Non-Deterministic Machine Translation: Potential and Evaluation},
  author={Author One and Author Two and Author Three},
  journal={arXiv preprint arXiv:24XX.XXXXX},
  year={2024}
}

```

## 📄 License

This project is licensed under the MIT License.