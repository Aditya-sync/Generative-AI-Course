🚀 AI Engineer Mastery: From First Principles to Production

An implementation-first learning repository covering the complete AI engineering journey — from scientific computing and classical machine learning to deep learning, NLP, transformers, LLM systems, and production-grade AI infrastructure.

The goal is simple: understand the fundamentals, implement the important concepts from scratch, and build systems that resemble real-world AI engineering workflows.

🗺️ Master Curriculum Architecture

flowchart TD
    classDef foundation fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef classical fill:#1e293b,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef deep fill:#1e293b,stroke:#f59e0b,stroke-width:2px,color:#fff;
    classDef nlp fill:#1e293b,stroke:#a855f7,stroke-width:2px,color:#fff;
    classDef infra fill:#1e293b,stroke:#ef4444,stroke-width:2px,color:#fff;

    subgraph P1["Phase 01: Foundations & Scientific Computing"]
        A1["NumPy: Tensor Operations & Broadcasting"]
        A2["Pandas: Data Wrangling, Merging & Aggregation"]
        A3["Matplotlib & Seaborn: EDA & Data Visualization"]
    end
    class P1,A1,A2,A3 foundation;

    subgraph P2["Phase 02: Classical Machine Learning"]
        B1["Regression: OLS, Polynomial Regression & Regularization"]
        B2["Classification: Logistic Regression, SVM & Decision Trees"]
        B3["Ensembles: Bagging, AdaBoost, XGBoost & LightGBM"]
        B4["Unsupervised Learning: K-Means, DBSCAN, PCA & t-SNE"]
        B5["Evaluation & Scikit-Learn Pipelines"]
    end
    class P2,B1,B2,B3,B4,B5 classical;

    subgraph P3["Phase 03: Deep Learning & Neural Computation"]
        C1["Matrix Calculus, Backpropagation & PyTorch Autograd"]
        C2["Optimization: Momentum, RMSprop & AdamW"]
        C3["Computer Vision: CNNs, Pooling & ResNet"]
        C4["Modular Training & Validation Pipelines"]
    end
    class P3,C1,C2,C3,C4 deep;

    subgraph P4["Phase 04: NLP, Sequence Models & Transformers"]
        D1["Embeddings: Word2Vec & Subword Tokenization"]
        D2["Sequence Models: RNNs, LSTMs & GRUs"]
        D3["Attention: Additive & Scaled Dot-Product"]
        D4["Transformers from Scratch: MHA & Residual Blocks"]
        D5["LLM Systems: PEFT, LoRA, QLoRA & RAG"]
    end
    class P4,D1,D2,D3,D4,D5 nlp;

    subgraph P5["Phase 05: AI Infrastructure & MLOps"]
        E1["Model Serving: FastAPI & Async Inference"]
        E2["Inference Optimization: ONNX & Quantization"]
        E3["Containerization & Experiment Tracking"]
    end
    class P5,E1,E2,E3 infra;

    P1 --> P2 --> P3 --> P4 --> P5

📁 Repository Anatomy

Machine Learning-II/
│
├── 01_data_science_foundations/
│   ├── numpy/
│   │   └── 01_array_math_and_broadcasting.ipynb
│   │       # Vectorized math, axis manipulation, broadcasting rules
│   │
│   ├── pandas/
│   │   └── 02_data_wrangling_and_aggregation.ipynb
│   │       # Indexing, grouping, reshaping, and memory optimization
│   │
│   └── data_visualization/
│       └── 03_eda_matplotlib_seaborn.ipynb
│           # Univariate/bivariate analysis and statistical distributions
│
├── 02_classical_machine_learning/
│   ├── regression/
│   │   ├── 01_linear_and_polynomial_regression.ipynb
│   │   │   # OLS normal equation, gradient descent, and polynomial features
│   │   │
│   │   └── 02_ridge_lasso_regularization.ipynb
│   │       # L1 sparsity vs. L2 regularization
│   │
│   ├── classification/
│   │   ├── 03_logistic_regression_from_scratch.ipynb
│   │   │   # Sigmoid, binary cross-entropy, and decision boundaries
│   │   │
│   │   ├── 04_knn_and_svm.ipynb
│   │   │   # Distance metrics, maximum-margin hyperplanes, and kernel methods
│   │   │
│   │   └── 05_decision_trees_and_naive_bayes.ipynb
│   │       # Information gain, Gini impurity, and Bayes' theorem
│   │
│   ├── ensemble_methods/
│   │   ├── 06_random_forest_and_adaboost.ipynb
│   │   │   # Bootstrap aggregation, feature subsampling, and adaptive weighting
│   │   │
│   │   └── 07_xgboost_and_lightgbm.ipynb
│   │       # Gradient boosting, second-order optimization, and histogram binning
│   │
│   ├── unsupervised_learning/
│   │   ├── 08_kmeans_and_dbscan.ipynb
│   │   │   # Centroid optimization and density-based clustering
│   │   │
│   │   └── 09_pca_dimensionality_reduction.ipynb
│   │       # Covariance matrices, eigenvalue decomposition, and explained variance
│   │
│   └── evaluation_and_pipelines/
│       └── 10_cross_validation_and_pipelines.ipynb
│           # Leak-free preprocessing, cross-validation, ROC-AUC, and PR curves
│
├── 03_deep_learning_pytorch/
│   ├── foundations/
│   │   ├── 01_tensors_autograd_and_mlp.ipynb
│   │   │   # Computational graphs, backward pass, and manual MLP implementation
│   │   │
│   │   └── 02_custom_loss_and_optimizers.ipynb
│   │       # Custom losses and AdamW implementation from equations
│   │
│   ├── convolutional_networks/
│   │   └── 03_cnn_and_resnet_transfer_learning.ipynb
│   │       # Convolutions, residual connections, and pretrained vision backbones
│   │
│   └── custom_training_loops/
│       └── 04_modular_pytorch_pipeline.py
│           # Dataset, DataLoader, Trainer, and Evaluator abstractions
│
├── 04_nlp_and_sequence_models/
│   ├── text_preprocessing_embeddings/
│   │   └── 01_tokenization_and_word2vec.ipynb
│   │       # BPE tokenization and Skip-Gram with negative sampling
│   │
│   ├── recurrent_networks/
│   │   ├── 02_rnn_and_lstm_from_scratch.ipynb
│   │   │   # Vanishing gradients and LSTM input/forget/output gates
│   │   │
│   │   └── 03_gru_text_classification.ipynb
│   │       # Reset/update gates and sequence-to-label modeling
│   │
│   └── attention_mechanisms/
│       └── 04_seq2seq_with_attention.ipynb
│           # Bahdanau additive and Luong multiplicative attention
│
├── 05_transformers_and_llms/
│   ├── transformer_from_scratch/
│   │   ├── 01_scaled_dot_product_attention.ipynb
│   │   │   # Q, K, V projections and attention maps
│   │   │
│   │   └── 02_multihead_attention_and_transformer_blocks.ipynb
│   │       # Pre-LN, positional encodings, RoPE, and FFN blocks
│   │
│   ├── huggingface_implementations/
│   │   └── 03_bert_fine_tuning.ipynb
│   │       # Hugging Face Trainer, tokenizers, and sequence classification
│   │
│   ├── peft_and_finetuning/
│   │   └── 04_qlora_instruction_tuning.ipynb
│   │       # PEFT, LoRA, and 4-bit QLoRA fine-tuning
│   │
│   └── rag_systems/
│       └── 05_vector_search_rag_pipeline.py
│           # Hybrid retrieval, vector indexing, and retrieval pipelines
│
├── 06_ai_infrastructure_mlops/
│   ├── serving_fastapi/
│   │   └── app.py
│   │       # Async model inference API
│   │
│   ├── model_quantization_onnx/
│   │   └── export_to_onnx.py
│   │       # PyTorch-to-ONNX export and INT8 quantization
│   │
│   ├── docker_packaging/
│   │   └── Dockerfile
│   │       # Multi-stage production container build
│   │
│   └── mlflow_tracking/
│       └── track_experiments.py
│           # Experiment logging, artifact storage, and metric tracking
│
├── data/
│   ├── raw/
│   │   # Unmodified raw datasets
│   │
│   └── processed/
│       # Cleaned and transformed datasets
│
├── .gitignore
│   # Ignore large model weights, virtual environments, caches, and artifacts
│
├── requirements.txt
│   # Reproducible Python dependencies
│
└── README.md

📚 Section Details

01. Data Science Foundations

NumPy

Vectorized calculations, matrix multiplication, array broadcasting, and axis-based operations without relying on slow Python loops.

Pandas

Efficient data wrangling, missing-data handling, merging and joining, groupby aggregations, reshaping, and memory optimization.

Data Visualization

Exploratory data analysis using histograms, KDE plots, correlation heatmaps, and multivariate feature distributions with Matplotlib and Seaborn.

02. Classical Machine Learning

Regression

Ordinary Least Squares (OLS), closed-form matrix solutions, Gradient Descent, polynomial features, Ridge (L2) regularization, and Lasso (L1) regularization.

Classification

Binary cross-entropy, Logistic Regression from scratch, maximum-margin classification with SVMs, and Decision Tree splitting using Entropy and Gini impurity.

Ensemble Methods

Bagging with Random Forests, out-of-bag evaluation, AdaBoost, Gradient Boosting, XGBoost, and LightGBM.

Unsupervised Learning

Centroid optimization with K-Means, density-based clustering with DBSCAN, and dimensionality reduction with PCA and t-SNE.

Pipelines & Evaluation

Cross-validation, leakage-free preprocessing, ColumnTransformer pipelines, Precision-Recall curves, and ROC-AUC evaluation.

03. Deep Learning with PyTorch

Foundations

PyTorch tensors, the autograd engine, computational graphs, manual backpropagation, and fully connected Multi-Layer Perceptrons (MLPs).

Optimization & Losses

Custom loss functions and optimization algorithms, including implementations of AdamW and RMSprop from their underlying equations.

Computer Vision

2D convolutions, pooling, stride calculations, CNN architectures, ResNet skip connections, and transfer learning with pretrained vision models.

Modular Training Pipelines

Clean and reusable training workflows built around Dataset, DataLoader, Trainer, and Evaluator components.

04. NLP & Sequence Models

Embeddings

Subword tokenization with BPE, Continuous Bag of Words (CBOW), and Skip-Gram Word2Vec with negative sampling.

Recurrent Architectures

RNNs, vanishing and exploding gradients, LSTM gating mechanisms, and GRUs for sequence modeling.

Attention Mechanisms

Additive attention (Bahdanau) and multiplicative attention (Luong) for sequence-to-sequence architectures.

05. Transformers & Modern LLMs

Transformers from Scratch

Scaled dot-product attention, Multi-Head Attention, positional encodings, Rotary Positional Embeddings (RoPE), Pre-LN architecture, residual connections, and feed-forward blocks.

Hugging Face Ecosystem

Fine-tuning transformer models using the Hugging Face ecosystem, including Transformers, Datasets, tokenizers, and Accelerate.

PEFT & Quantization

Parameter-Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LoRA), and 4-bit Quantized LoRA (QLoRA) for adapting open-source language models.

RAG Systems

Semantic chunking, dense embeddings, vector retrieval, FAISS/ChromaDB indexing, hybrid retrieval, and retrieval-augmented generation pipelines.

06. AI Infrastructure & MLOps

Model Serving

Production-oriented asynchronous REST APIs using FastAPI and Pydantic schemas.

Inference Optimization

Exporting PyTorch models to ONNX, graph optimization, and INT8 quantization for more efficient inference.

Containerization

Multi-stage Docker builds for reproducible and deployment-ready AI services.

Experiment Tracking

Tracking hyperparameters, training metrics, loss curves, model artifacts, and experiments using MLflow.

🎯 Learning Philosophy

This repository follows three principles:

Understand the mathematics — learn what is happening under the hood.

Implement from scratch — build core algorithms before relying entirely on abstractions.

Build production systems — turn individual concepts into deployable AI applications.

Learn → Implement → Experiment → Build → Deploy

🧭 Roadmap

Scientific Computing
        ↓
Classical Machine Learning
        ↓
Deep Learning
        ↓
NLP & Sequence Models
        ↓
Transformers
        ↓
LLMs & RAG
        ↓
AI Infrastructure & MLOps
        ↓
Production AI Systems

🚧 This repository is continuously evolving.

New implementations, experiments, projects, and advanced AI engineering topics will be added as the roadmap progresses.