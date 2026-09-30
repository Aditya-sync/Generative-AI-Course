# AI Engineer Roadmap (Machine Learning to Production)

A hands-on implementation repository covering the complete AI engineering lifecycle: mathematical foundations, classical ML, deep learning architectures, modern NLP & Transformers, and production infrastructure.

## Roadmap Structure

- **01_data_science_foundations/**: Vector operations (NumPy), wrangling (Pandas), and visualization (Matplotlib & Seaborn).
- **02_classical_machine_learning/**: Supervised & unsupervised algorithms built from scratch and benchmarked with Scikit-Learn.
- **03_deep_learning_pytorch/**: Neural network mechanics, backpropagation, CNNs, and custom PyTorch training loops.
- **04_nlp_and_sequence_models/**: Tokenization, embeddings (Word2Vec), RNNs, LSTMs, and Seq2Seq with attention.
- **05_transformers_and_llms/**: Multi-head attention from scratch, Hugging Face pipelines, LoRA/QLoRA fine-tuning, and RAG systems.
- **06_ai_infrastructure_mlops/**: Production serving with FastAPI, ONNX optimization, experiment tracking with MLflow, and Docker packaging.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt