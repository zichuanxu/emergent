# Emergent Communication in Multi-Agent Systems

A research project investigating emergent symbolic communication in a cooperative "Architect-Builder" multi-agent environment using deep reinforcement learning.

## Overview

This project implements a two-agent cooperative game where:

- **Architect (Sender)**: Sees the target blueprint but cannot act. Generates symbolic messages to guide the Builder.
- **Builder (Receiver)**: Can manipulate the environment but cannot see the blueprint. Must interpret symbolic messages to construct the target.

The goal is to study how meaningful, interpretable communication emerges from the necessity of cooperation.

## Quick Start

### Installation

```bash
# Create conda environment
conda create -n emergent-comm python=3.9
conda activate emergent-comm

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pygame seaborn scikit-learn tqdm tensorboard
```

### Running the Research Suite

```bash
# Quick examples
python examples.py

# Train with enhanced dashboard
python research_suite.py train --dashboard --episodes 1000

# Run ablation studies
python research_suite.py ablation --category message_length --episodes 500

# Test generalization
python research_suite.py generalization --episodes 50

# Start research dashboard
python research_suite.py dashboard --mode training

# Run complete pipeline
python research_suite.py pipeline --episodes 500

# Check project status
python research_suite.py status
```

## Project Structure

```
emergent-communication/
├── config/                 # Configuration files
├── src/                    # Source code
│   ├── agents/            # Neural network architectures
│   ├── environments/      # Grid world environment
│   ├── training/          # MAPPO training logic
│   ├── evaluation/        # Evaluation frameworks
│   ├── research/          # Ablation & generalization tools
│   ├── visualization/     # Dashboard and plotting tools
│   ├── utils/             # Utilities and logging
│   └── cli/               # Command line interface
├── scripts/               # Legacy training scripts
├── results/               # Experiment outputs
├── research_suite.py      # Main entry point
├── examples.py            # Quick usage examples
└── README.md
```

## Key Features

- **2D Grid Environment**: Configurable NxN grid with colored blocks
- **Neural Architectures**: CNN vision + Transformer communication for Architect, CNN + RNN for Builder
- **MAPPO Training**: Multi-Agent Proximal Policy Optimization
- **Advanced Interpretability Tools**:
  - Normalized Mutual Information (NMI) analysis
  - Comprehensive consistency tests
  - Compositional structure analysis
  - Communication efficiency metrics
  - Emergence indicators (systematicity, productivity, stability)
- **Multi-Level Evaluation Framework**: Basic, comprehensive, and integrated evaluation approaches
- **Real-time Visualization**: Live environment rendering with communication logs

## Configuration

Edit `config/config.py` to modify:

- Environment parameters (grid size, number of blocks, colors)
- Network architectures (hidden dimensions, layers)
- Training hyperparameters (learning rates, batch sizes)
- Evaluation settings (test blueprints, metrics)

## Experiment Tracking

Results are automatically saved to `results/[experiment_name]/`:

- `models/`: Trained model checkpoints
- `logs/`: Training logs and TensorBoard files
- `plots/`: Generated visualizations and analysis plots
- `report.md`: Automated experiment summary

## Hardware Requirements

- GPU recommended (CUDA support included)
- 8GB+ RAM for default configurations
- Tested on RTX 4060 with CUDA 11.8

## Evaluation Framework

The project includes a comprehensive multi-level evaluation framework for analyzing emergent communication:

### 1. Basic Evaluation Framework (`evaluation_framework.py`)

- **NMI Computation**: Measures information shared between messages and various targets (actions, states, rewards)
- **Consistency Tests**: Analyzes message stability and action consistency
- **Interpretability Metrics**: Communication efficiency and message diversity analysis

### 2. Comprehensive Evaluation (`scripts/evaluate_model.py`)

- **Episode Data Collection**: Gathers communication data from multiple episodes
- **Behavioral Analysis**: Success rates, reward progression, learning indicators
- **Advanced Visualizations**: t-SNE message space, correlation heatmaps, performance over time
- **Detailed Reporting**: JSON reports with human-readable summaries

### 3. Advanced Interpretability Analysis (`interpretability_analysis.py`)

- **Compositionality Analysis**: Tests if message dimensions encode different information types
- **Message Structure Analysis**: Entropy, clustering, and diversity metrics
- **Communication Efficiency**: Information-theoretic measures of communication quality
- **Emergence Indicators**: Systematicity, productivity, and temporal stability

### 4. Integrated Evaluation (`integrated_evaluation.py`)

- **Unified Framework**: Combines all evaluation approaches
- **Comprehensive Reporting**: Integrated analysis with cross-framework comparisons
- **Quality Scoring**: Overall assessment with specific recommendations

### Usage Examples

```python
# Basic framework usage
from src.evaluation.framework import EvaluationFramework

evaluator = EvaluationFramework(model, data_loader)
evaluator.collect_interaction_data(num_episodes=100)
results = evaluator.evaluate()

# Advanced interpretability analysis
from src.evaluation.interpretability import InterpretabilityAnalyzer

analyzer = InterpretabilityAnalyzer(messages, actions, states, rewards)
report = analyzer.generate_interpretability_report()
analyzer.visualize_interpretability(save_path='analysis.png')

# Integrated evaluation
from src.evaluation.integrated import IntegratedEvaluationFramework

evaluator = IntegratedEvaluationFramework(model_path, num_episodes=100)
report = evaluator.generate_integrated_report()
```

### Evaluation Metrics

**Communication Effectiveness:**

- Message-Action NMI: How well messages correlate with actions
- Message-State NMI: Relationship between messages and environmental states
- Message-Reward NMI: Connection between communication and task success

**Consistency Measures:**

- Message Stability: Same contexts produce similar messages
- Action Consistency: Similar messages lead to similar actions
- Compositional Consistency: Message components are used systematically

**Interpretability Indicators:**

- Compositionality Score: Different message dimensions encode different information
- Message Diversity: Variety in communication patterns
- Communication Efficiency: Information content relative to capacity
- Emergence Strength: Systematicity, productivity, and stability measures

### Output Files

Evaluation generates several output files in the `evaluation_results/` directory:

- `evaluation_report.json`: Comprehensive metrics in JSON format
- `evaluation_summary.txt`: Human-readable summary with recommendations
- `integrated_evaluation_report.json`: Combined analysis from all frameworks
- `message_space_tsne.png`: t-SNE visualization of message space
- `message_action_heatmap.png`: Correlation between messages and actions
- `performance_over_time.png`: Learning progression visualization
- `interpretability_analysis.png`: Advanced interpretability visualizations

## Enhanced Research Suite

The project has been reorganized into a clean, modular structure with a comprehensive research suite that includes three major enhancements:

### 1. Enhanced Real-time Visualization Dashboard

**Multi-mode Dashboard:**

- **Training Mode**: Real-time training monitoring with advanced metrics
- **Ablation Mode**: Live comparison of ablation study results
- **Generalization Mode**: Zero-shot performance visualization
- **Comparison Mode**: Multi-experiment comparison interface

**Advanced Features:**

- Emergence indicators and systematicity analysis
- Communication stability tracking
- Generalization proxy metrics
- Research insights and recommendations
- Performance zone analysis with quality thresholds

### 2. Comprehensive Ablation Studies Framework

**Systematic Testing of:**

- **Message Length**: 1, 3, 5 dimensions
- **Vocabulary Size**: 4, 8, 16 symbols
- **Architecture**: 32, 64, 128 hidden dimensions
- **Communication Mechanisms**: Discrete vs continuous, no communication baseline
- **Training Hyperparameters**: Learning rates, batch sizes
- **Environment Complexity**: Grid sizes, color combinations

**Features:**

- Automated experiment execution
- Real-time monitoring dashboard
- Statistical analysis and visualization
- Performance comparison across categories
- Best configuration identification

### 3. Zero-shot Generalization Tests

**Comprehensive Generalization Testing:**

- **Grid Size Generalization**: 3x3, 6x6, 8x8 environments
- **Color Generalization**: Novel colors and combinations
- **Complexity Generalization**: More blocks, denser environments
- **Pattern Generalization**: Lines, corners, symmetric patterns
- **Episode Length Generalization**: Shorter/longer time horizons

**Analysis Features:**

- Communication pattern analysis across scenarios
- Performance degradation measurement
- Generalization capability assessment
- Failure mode identification

## Quick Start with Research Suite

### Unified Research Interface

```bash
# Run complete research pipeline
python research_suite.py pipeline --episodes 500

# Run training with enhanced dashboard
python research_suite.py train --dashboard --episodes 1000

# Run specific ablation category
python research_suite.py ablation --category message_length --episodes 500

# Run generalization tests
python research_suite.py generalization --model_path results/model.pt

# Start research dashboard
python research_suite.py dashboard --mode comparison

# Check project status
python research_suite.py status
```

### Individual Components

#### Enhanced Training Dashboard

```bash
# Training mode with advanced metrics
python -m src.visualization.dashboard --mode training

# Ablation monitoring
python -m src.visualization.dashboard --mode ablation --ablation_results ablation_results/

# Generalization visualization
python -m src.visualization.dashboard --mode generalization --generalization_results generalization_results/
```

#### Ablation Studies

```bash
# Run all ablation categories
python -m src.research.ablation --category all --episodes 500

# Run specific category
python -m src.research.ablation --category communication --episodes 300

# Custom output directory
python -m src.research.ablation --output_dir my_ablation_results/
```

#### Generalization Tests

```bash
# Full generalization test suite
python -m src.research.generalization --model_path results/model.pt --episodes 50

# Custom scenarios
python -m src.research.generalization --episodes 100 --output_dir custom_gen_results/
```

## Research Dashboard Features

### Training Mode Dashboard

**4x4 Grid Layout with 16 Plots:**

**Row 1 - Training Progress:**

- Episode Rewards with trend analysis
- Success Rate with performance zones
- Episode Length with efficiency metrics
- Training Loss with convergence analysis

**Row 2 - Communication Analysis:**

- Message Evolution & Space Coverage
- Communication Quality (NMI) with zones
- Message Diversity with optimal ranges
- Communication Consistency with stability

**Row 3 - Advanced Analysis:**

- Message-Action Correlation with significance
- Action Distribution with entropy analysis
- Communication Efficiency tracking

**Row 4 - Research Insights:**

- Emergence Indicators
- Generalization Proxy
- Communication Stability
- Research Insights & Recommendations

### Ablation Mode Dashboard

- Real-time comparison across experiments
- Category-specific performance analysis
- Learning curve comparisons
- Best configuration identification

### Generalization Mode Dashboard

- Zero-shot performance overview
- Category-specific generalization analysis
- Communication pattern changes
- Failure mode visualization

## Interpreting Research Metrics

### Emergence Indicators

- **Systematicity**: Similar contexts → similar messages
- **Compositionality**: Message dimensions encode different information
- **Productivity**: Novel message combinations
- **Stability**: Temporal consistency of communication

### Communication Quality Zones

- **Excellent (NMI > 0.5)**: Strong message-action correlation
- **Good (NMI > 0.3)**: Moderate correlation
- **Fair (NMI > 0.1)**: Weak correlation
- **Poor (NMI < 0.1)**: No meaningful correlation

### Generalization Assessment

- **Strong (>70%)**: Good zero-shot transfer
- **Moderate (40-70%)**: Limited transfer
- **Weak (<40%)**: Poor generalization

## Research Pipeline Workflow

1. **Training Phase**: Train agents with enhanced monitoring
2. **Ablation Phase**: Test different architectures and configurations
3. **Generalization Phase**: Evaluate zero-shot performance
4. **Analysis Phase**: Comprehensive evaluation and reporting
5. **Comparison Phase**: Multi-experiment dashboard analysis

## Output Structure

```
project/
├── results/                    # Training results
│   └── architect_builder_v1/
├── ablation_results/          # Ablation study results
│   ├── analysis/             # Visualizations and reports
│   └── *_results.json        # Category results
├── generalization_results/    # Generalization test results
│   ├── analysis/             # Analysis and visualizations
│   └── generalization_results.json
└── evaluation_results/        # Integrated evaluation results
```
