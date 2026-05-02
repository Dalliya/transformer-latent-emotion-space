# 🌌 [S-E-M-A-N-T-I-C] Latent Space Topology Engine

<p align="center">
  <img src="https://img.shields.io/badge/Version-2.1.0-00FF41?style=for-the-badge&logo=matrix&logoColor=00FF41" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-08F7FE?style=for-the-badge&logo=python&logoColor=08F7FE" />
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Research--Active-FF003C?style=for-the-badge&logo=statuspage&logoColor=white" />
</p>

<p align="center">
  <b>Advanced Manifold Projection and Zero-Shot Domain Transfer for NLP Models.</b><br>
  <i>"Visualizing the semantic chasm between machine emotion and human contextual logic."</i>
</p>

---

## ⚡️ Mission Statement
**[S-E-M-A-N-T-I-C]** is an analytical framework designed to audit and visualize the behavior of Large Language Models (LLMs) when transitioning between complex NLP tasks. 

While pre-trained models can identify raw emotions, applying them directly to business logic (like product sentiment) often leads to critical misclassifications. This engine projects high-dimensional neural representations into a 2D topological space, allowing researchers to visually pinpoint where algorithmic heuristics fail and where **Human-in-the-Loop** annotation is absolutely required.

---

## 🧠 Core Methodology: The Latent Space Projection
The backbone of this engine relies on bridging **Transformer Architectures** with advanced **Manifold Learning**.

### 1. High-Dimensional Extraction (BERT)
Unlike traditional text analysis, this pipeline does not rely on keywords. It processes text through a pre-trained `BertForSequenceClassification` model (trained on the 28-class *GoEmotions* taxonomy). We extract the `[CLS]` token from the final hidden state—a **768-dimensional dense vector** representing the pure semantic essence of the sentence.

### 2. Zero-Shot Domain Transfer (28 ➔ 2)
The engine executes a heuristic mapping algorithm to collapse the 28 fine-grained emotions into a binary business requirement (Positive / Negative IMDB Reviews). 
* *Positive mapped:* Joy, Admiration, Excitement, etc.
* *Negative mapped:* Anger, Disgust, Fear, Neutral, etc.

### 3. Topological Manifold Learning (UMAP)
To make the 768-D space comprehensible to the human eye, the engine utilizes **UMAP** (Uniform Manifold Approximation and Projection). UMAP acts as the mathematical bridge, preserving both the local clustering of emotions and the global semantic distances, effectively creating a "map" of human language.

---

## 👁️ The Matrix UI: Interactive Visual Audit
The true power of **[S-E-M-A-N-T-I-C]** lies in its custom rendering engine. The data is visualized through a bespoke, dual-panel Matrix-styled dashboard.

> 🌐 **[CLICK HERE TO LAUNCH THE INTERACTIVE UMAP DASHBOARD](INSERT_YOUR_GITHUB_PAGES_OR_NETLIFY_LINK_HERE)**

*(If viewing offline, download and open `data/processed/umap_matrix_comparative.html` in any modern web browser).*

### Decoding the Visualization:
* **The Coordinates (Z1 / Z2):** Proximity dictates contextual similarity. Points grouped together share deep semantic meaning, regardless of the specific words used.
* **Left Panel (Ground Truth):** How human annotators judged the text based on IMDB rules.
* **Right Panel (Machine Logic):** How the Zero-Shot heuristic mapped the emotion.
* **The Discovery:** By hovering over the data points, researchers can instantly spot contextual failures. For example, the machine detects **"Fear"** and flags it as *Negative*. However, the human flags it as *Positive* because the text is a glowing review of a horror movie. **Context is King.**

---

## 🏗 System Architecture
Built on a modular pipeline designed for scalability and rapid inference across Apple Silicon (MPS), CUDA, or CPU environments.

### Core Modules:
1.  **`DataLoader` (Telemetry & Balancing):** * Implements robust memory-management by utilizing **Stratified Sampling**. This ensures that even when datasets are reduced by 90% (to accommodate local RAM limits), the mathematical balance of all 28 emotion classes remains identical to the original population.
2.  **`InferenceEngine` (Tensor Operations):** * Handles tokenization, batching, and `[CLS]` embedding extraction using PyTorch `no_grad()` contexts for optimized memory footprint.
3.  **`MatrixVisualizer` (The Observer):** * A high-performance `Plotly` suite generating the signature Cyber-Grid aesthetic. It features dynamic Neon-HSV color mapping and unescaped HTML hover tooltips for zero-latency data auditing.
4.  **`Profiler` (Optimization Audit):** * Includes custom `@timeit` decorators proving that C-level vectorized `pandas.apply()` operations outperform standard Python `for` loops by >200% during the mapping phase.

---

## 🚀 Execution Guide

1. **Initialize the Environment:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
   cd YOUR_REPOSITORY
   pip install -r requirements.txt

   2. **Engage the Pipeline:**
   Execute the main orchestrator script to initiate the analysis:
   ```bash
   python src/main.py
The engine will automatically detect available hardware accelerators (MPS/CUDA/CPU), perform the high-dimensional semantic extraction, and compile the interactive Matrix dashboard.

<div align="center">

### 👩‍💻 Dariia Zhdanova
**ML Developer | Architect of Semantic Intelligence**

*I specialize in decoding the hidden topological structures within human language. True AI engineering isn't just about training models to recognize patterns, but about understanding the precise semantic boundaries where rigid machine logic collides with nuanced human context.*

</div>

> **Project Thesis:** "In this deployment, I audited the gap between zero-shot heuristic mapping and contextual ground truth. The conclusion is absolute: blind algorithmic domain transfer is a dangerous illusion in complex linguistic environments. The true power of an NLP engine lies in its transparent topology—revealing the exact threshold where raw machine-detected emotion meets the reality of human evaluation and business logic."

<div align="center">
<br>

📫 **Connect:** &nbsp; [LinkedIn](https://www.linkedin.com/in/dariia-z-b7146223a) &nbsp;|&nbsp; [GitHub (@Dalliya)](https://github.com/Dalliya)

</div>