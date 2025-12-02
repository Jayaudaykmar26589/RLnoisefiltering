import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from collections import deque, defaultdict
import random
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import time
from datetime import datetime
import warnings
from contextlib import contextmanager

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('noise_filtering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Centralized configuration management."""
    # Model Architecture
    embedding_dim: int = 768
    hidden_dim: int = 512
    num_quantiles: int = 51
    dropout_rate: float = 0.2
    
    # Training
    batch_size: int = 32
    num_episodes: int = 100
    learning_rate: float = 1e-4
    buffer_size: int = 100000
    alpha: float = 0.6  # Priority exponent
    beta: float = 0.4   # Importance sampling exponent
    
    # RL Hyperparameters
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    
    # Bayesian Neural Network
    prior_sigma: float = 0.1
    kl_weight: float = 0.01
    num_samples_uncertainty: int = 10
    
    # Noise Generation
    noise_levels: Dict[str, float] = None
    noise_probabilities: Dict[str, float] = None
    
    # Evaluation
    num_test_samples: int = 5
    plot_dpi: int = 150
    
    # Output
    output_dir: str = "outputs"
    save_plots: bool = True
    interactive_plots: bool = False
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = {
                'OCR': 0.1, 'ASR': 0.15, 'Typo': 0.1, 
                'Adversarial': 0.05, 'Semantic': 0.1, 'Mixed': 0.15
            }
        if self.noise_probabilities is None:
            self.noise_probabilities = {
                'OCR': 0.2, 'ASR': 0.2, 'Typo': 0.2, 
                'Adversarial': 0.15, 'Semantic': 0.15, 'Mixed': 0.1
            }
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

# Utility functions
def setup_matplotlib_for_plotting():
    """Setup matplotlib and seaborn for plotting with proper configuration."""
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

@contextmanager
def timer(name="Operation"):
    """Context manager for timing operations."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} completed in {elapsed:.2f} seconds")

class MetricsLogger:
    """Enhanced metrics logging with multiple backends."""
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = defaultdict(list)
        self.timestamps = []
        self.start_time = time.time()
        
    def log_metric(self, name: str, value: float, step: int = None):
        """Log a single metric."""
        self.metrics[name].append(value)
        if step is not None:
            self.timestamps.append(time.time() - self.start_time)
        
    def log_dict(self, metrics_dict: Dict[str, float], step: int = None):
        """Log multiple metrics at once."""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)
            
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1] if values else 0
                }
        return summary
        
    def save_to_json(self, filepath: str = None):
        """Save metrics to JSON file."""
        if filepath is None:
            filepath = Path(self.config.output_dir) / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'metrics': dict(self.metrics),
            'timestamps': self.timestamps,
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Metrics saved to {filepath}")
        return filepath

# ==================== Enhanced Visualization Module ====================
class TrainingVisualizer:
    """Enhanced visualization module with interactive features."""
    
    def __init__(self, config: Config):
        self.config = config
        setup_matplotlib_for_plotting()
        
    def plot_training_metrics(self, history: Dict, filename: str = None) -> str:
        """Enhanced training metrics visualization with trend analysis."""
        if filename is None:
            filename = Path(self.config.output_dir) / 'training_results.png'
            
        epochs = range(1, len(history['rewards']) + 1)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Training Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Enhanced Average Reward with confidence intervals
        self._plot_metric_with_trend(ax1, epochs, history['rewards'], 
                                   'Average Reward per Episode', 'blue', 'Reward')
        
        # Plot 2: Detection Accuracy with moving average
        self._plot_metric_with_trend(ax2, epochs, history['detection_acc'], 
                                   'Noise Detection Accuracy', 'green', 'Accuracy')
        
        # Plot 3: Classification Accuracy
        self._plot_metric_with_trend(ax3, epochs, history['classification_acc'], 
                                   'Noise Classification Accuracy', 'red', 'Accuracy')
        
        # Plot 4: Epsilon Decay with annotations
        self._plot_epsilon_decay(ax4, epochs, history['epsilon'])
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"Training plots saved to {filename}")
        plt.close()
        
        if self.config.interactive_plots:
            return self._create_interactive_training_plot(history)
        
        return str(filename)
    
    def _plot_metric_with_trend(self, ax, epochs, values, title: str, color: str, ylabel: str):
        """Plot metric with trend analysis and confidence intervals."""
        # Main line plot
        ax.plot(epochs, values, color=color, linewidth=2, alpha=0.8, 
               marker='o', markersize=2, markevery=max(1, len(epochs)//20))
        
        # Moving average
        if len(values) > 10:
            window = min(10, len(values)//4)
            moving_avg = pd.Series(values).rolling(window=window, center=True).mean()
            ax.plot(epochs, moving_avg, color='orange', linewidth=2, 
                   linestyle='--', alpha=0.7, label=f'Moving Avg ({window})')
        
        # Trend line
        if len(epochs) > 10:
            z = np.polyfit(list(epochs), values, min(3, len(epochs)//3))
            p = np.poly1d(z)
            trend_values = p(list(epochs))
            ax.plot(epochs, trend_values, "purple", linestyle=':', 
                   linewidth=2, alpha=0.6, label='Trend')
            
            # Calculate R-squared
            ss_res = np.sum((values - trend_values) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Add annotation
            ax.annotate(f'R² = {r_squared:.3f}', 
                       xy=(0.02, 0.98), xycoords='axes fraction',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9)
        
        # Set y-axis limits for accuracy plots
        if ylabel == 'Accuracy':
            ax.set_ylim(0, 1.05)
    
    def _plot_epsilon_decay(self, ax, epochs, epsilon_values):
        """Enhanced epsilon decay plot with annotations."""
        ax.plot(epochs, epsilon_values, 'm-', linewidth=2, marker='D', 
               markersize=3, markevery=max(1, len(epochs)//20))
        
        # Add horizontal lines for key values
        final_epsilon = epsilon_values[-1]
        ax.axhline(y=final_epsilon, color='purple', linestyle=':', 
                  linewidth=1.5, alpha=0.7, label=f'Final: {final_epsilon:.3f}')
        
        # Calculate decay rate
        if len(epsilon_values) > 1:
            decay_rate = (epsilon_values[0] - epsilon_values[-1]) / (len(epsilon_values) - 1)
            ax.text(0.02, 0.98, f'Decay Rate: {decay_rate:.4f}/episode', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
                   verticalalignment='top')
        
        ax.set_title('Epsilon Decay (Exploration Rate)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Epsilon', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
    
    def plot_evaluation_results(self, results: Dict, filename: str = None) -> str:
        """Enhanced evaluation visualization with interactive features."""
        if filename is None:
            filename = Path(self.config.output_dir) / 'evaluation_results.png'
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance by Noise Type with error bars
        self._plot_noise_type_performance(ax1, results)
        
        # Plot 2: Domain-specific performance
        self._plot_domain_performance(ax2, results)
        
        # Plot 3: Enhanced confusion matrix
        self._plot_confusion_matrix(ax3, results)
        
        # Plot 4: Comprehensive summary
        self._plot_summary_statistics(ax4, results)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {filename}")
        plt.close()
        
        if self.config.interactive_plots:
            return self._create_interactive_evaluation_plot(results)
        
        return str(filename)
    
    def _plot_noise_type_performance(self, ax, results):
        """Plot performance metrics by noise type."""
        noise_types = list(results['by_noise_type'].keys())
        
        detection_rates = []
        classification_accs = []
        sample_counts = []
        
        for nt in noise_types:
            stats = results['by_noise_type'][nt]
            detection_rate = stats['detected'] / max(1, stats['total'])
            classification_acc = stats['classified_correct'] / max(1, stats['detected'])
            
            detection_rates.append(detection_rate)
            classification_accs.append(classification_acc)
            sample_counts.append(stats['total'])
        
        x = np.arange(len(noise_types))
        width = 0.35
        
        # Add error bars based on sample size
        detection_errors = [np.sqrt(p * (1-p) / max(1, count)) for p, count in zip(detection_rates, sample_counts)]
        classification_errors = [np.sqrt(p * (1-p) / max(1, count)) for p, count in zip(classification_accs, sample_counts)]
        
        bars1 = ax.bar(x - width/2, detection_rates, width, 
                      yerr=detection_errors, capsize=5, label='Detection Rate', 
                      color='skyblue', edgecolor='navy', linewidth=1.5)
        bars2 = ax.bar(x + width/2, classification_accs, width, 
                      yerr=classification_errors, capsize=5, label='Classification Acc', 
                      color='lightcoral', edgecolor='darkred', linewidth=1.5)
        
        # Add value labels
        for bars, values in [(bars1, detection_rates), (bars2, classification_accs)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{value:.1%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('Noise Detection & Classification by Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Noise Type', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(noise_types, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_domain_performance(self, ax, results):
        """Plot performance by domain with confidence intervals."""
        domains = list(results['by_domain'].keys())
        domain_accs = []
        domain_counts = []
        
        for d in domains:
            stats = results['by_domain'][d]
            acc = stats['correct_detection'] / max(1, stats['total'])
            domain_accs.append(acc)
            domain_counts.append(stats['total'])
        
        # Create color map based on performance
        colors = plt.cm.RdYlGn([acc for acc in domain_accs])
        
        bars = ax.bar(domains, domain_accs, color=colors, 
                     edgecolor='black', linewidth=1.5)
        
        # Add confidence intervals
        errors = [np.sqrt(acc * (1-acc) / max(1, count)) 
                 for acc, count in zip(domain_accs, domain_counts)]
        ax.errorbar(domains, domain_accs, yerr=errors, fmt='none', 
                   ecolor='black', capsize=5, capthick=2)
        
        # Add value labels and sample counts
        for bar, acc, count in zip(bars, domain_accs, domain_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, height/2,
                   f'n={count}', ha='center', va='center', fontsize=9, 
                   color='white', fontweight='bold')
        
        ax.set_title('Detection Accuracy by Domain', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Domain', fontsize=12)
        ax.set_xticklabels(domains, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_confusion_matrix(self, ax, results):
        """Plot enhanced confusion matrix with annotations."""
        cm = results['confusion_matrix']
        noise_types = list(results['by_noise_type'].keys())
        
        # Normalize to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        im = ax.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Percentage (%)', rotation=270, labelpad=15)
        
        # Set ticks and labels
        tick_marks = np.arange(len(noise_types))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(noise_types, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(noise_types, fontsize=9)
        ax.set_ylabel('True Noise Type', fontsize=12)
        ax.set_xlabel('Predicted Noise Type', fontsize=12)
        ax.set_title('Confusion Matrix\\n(Detected Samples Only)', fontsize=14, fontweight='bold')
        
        # Add text annotations with counts and percentages
        thresh = cm_percent.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = int(cm[i, j])
                percent = cm_percent[i, j]
                if count > 0:
                    text = f'{count}\\n({percent:.1f}%)'
                    color = "white" if percent > thresh else "black"
                    ax.text(j, i, text, ha="center", va="center",
                           color=color, fontsize=9, fontweight='bold')
    
    def _plot_summary_statistics(self, ax, results):
        """Plot comprehensive summary statistics."""
        ax.axis('off')
        
        # Calculate key metrics
        total_samples = results['total_samples']
        detection_acc = results['correct_detection'] / max(1, total_samples)
        total_detected_noisy = results['detected_as_noisy']
        class_acc = results['correct_classification'] / max(1, total_detected_noisy) if total_detected_noisy > 0 else 0
        
        # Create summary text
        summary_lines = [
            "Model Performance Summary",
            "=" * 50,
            "",
            f"Dataset Overview:",
            f"  Total Samples: {total_samples:,}",
            f"  Actually Noisy: {results['actually_noisy']:,}",
            f"  Detected as Noisy: {total_detected_noisy:,}",
            f"  Detection Rate: {total_detected_noisy/total_samples:.1%}",
            "",
            "Detection Performance:",
            f"  Overall Accuracy: {detection_acc:.2%}",
            f"  Correct Detections: {results['correct_detection']:,}/{total_samples:,}",
            "",
            "Classification Performance:",
            f"  Among Detected Noisy: {class_acc:.2%}",
            f"  Correct Classifications: {results['correct_classification']:,}/{total_detected_noisy:,}",
            "",
            "Performance by Noise Type:",
        ]
        
        # Add performance by noise type
        for nt in results['by_noise_type']:
            stats = results['by_noise_type'][nt]
            if stats['total'] > 0:
                det_acc = stats['detected'] / stats['total']
                class_acc_type = stats['classified_correct'] / max(1, stats['detected'])
                summary_lines.append(f"  {nt:12}: {det_acc:.1%} det, {class_acc_type:.1%} cls")
        
        # Add domain performance
        summary_lines.extend([
            "",
            "Domain Performance:",
        ])
        
        for domain in results['by_domain']:
            stats = results['by_domain'][domain]
            if stats['total'] > 0:
                acc = stats['correct_detection'] / stats['total']
                summary_lines.append(f"  {domain:15}: {acc:.1%}")
        
        summary_text = "\\n".join(summary_lines)
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.3))
        
        ax.set_title('Performance Overview', fontsize=14, fontweight='bold', pad=20)
    
    def plot_advanced_metrics(self, data: Dict, filename: str = None) -> str:
        """Advanced analysis with multiple dimensionality reduction techniques."""
        if filename is None:
            filename = Path(self.config.output_dir) / 'advanced_analysis.png'
            
        fig = plt.figure(figsize=(20, 8))
        fig.suptitle('Advanced Model Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Uncertainty Distribution with statistics
        ax1 = plt.subplot(1, 3, 1)
        self._plot_uncertainty_distribution(ax1, data)
        
        # Plot 2: Quantile analysis
        ax2 = plt.subplot(1, 3, 2)
        self._plot_quantile_analysis(ax2, data)
        
        # Plot 3: Embedding visualization with multiple methods
        ax3 = plt.subplot(1, 3, 3)
        self._plot_embedding_analysis(ax3, data)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"Advanced analysis plots saved to {filename}")
        plt.close()
        
        return str(filename)
    
    def _plot_uncertainty_distribution(self, ax, data):
        """Plot uncertainty distributions with statistical tests."""
        clean_unc = data.get('uncertainty_clean', [])
        noisy_unc = data.get('uncertainty_noisy', [])
        
        if len(clean_unc) > 0 and len(noisy_unc) > 0:
            # Plot histograms
            ax.hist(clean_unc, bins=25, alpha=0.6, label='Clean Text', 
                   color='green', density=True, edgecolor='darkgreen', linewidth=1.2)
            ax.hist(noisy_unc, bins=25, alpha=0.6, label='Noisy Text', 
                   color='red', density=True, edgecolor='darkred', linewidth=1.2)
            
            # Add statistical annotations
            mean_clean = np.mean(clean_unc)
            mean_noisy = np.mean(noisy_unc)
            std_clean = np.std(clean_unc)
            std_noisy = np.std(noisy_unc)
            
            ax.axvline(mean_clean, color='darkgreen', linestyle='--', linewidth=2, 
                      label=f'Clean Mean: {mean_clean:.3f}±{std_clean:.3f}')
            ax.axvline(mean_noisy, color='darkred', linestyle='--', linewidth=2, 
                      label=f'Noisy Mean: {mean_noisy:.3f}±{std_noisy:.3f}')
            
            # Add separation analysis
            if len(clean_unc) > 1 and len(noisy_unc) > 1:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(clean_unc, noisy_unc)
                effect_size = (mean_noisy - mean_clean) / np.sqrt((std_clean**2 + std_noisy**2) / 2)
                
                ax.text(0.02, 0.98, f'Separation Analysis:\\nt-stat: {t_stat:.3f}\\np-value: {p_value:.3e}\\nEffect Size: {effect_size:.3f}', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax.set_title('Bayesian Uncertainty Distribution\\n(Predictive Entropy)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predictive Entropy', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    def _plot_quantile_analysis(self, ax, data):
        """Analyze quantile distributions for distributional RL insights."""
        quantiles = data.get('quantiles', [])
        
        if len(quantiles) > 0:
            quantiles = np.array(quantiles)
            
            # Plot quantile distributions
            for i in range(min(10, len(quantiles))):
                ax.plot(quantiles[i], linewidth=2, alpha=0.7, 
                       label=f'Sample {i+1}' if i < 5 else '')
            
            # Add statistics
            mean_quantiles = np.mean(quantiles, axis=0)
            std_quantiles = np.std(quantiles, axis=0)
            
            ax.plot(mean_quantiles, 'black', linewidth=3, label='Mean')
            ax.fill_between(range(len(mean_quantiles)), 
                           mean_quantiles - std_quantiles,
                           mean_quantiles + std_quantiles,
                           alpha=0.2, color='gray', label='±1 STD')
            
            ax.set_title('Return Distribution Quantiles\\n(QR-DQN Analysis)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Quantile Index (τ)', fontsize=12)
            ax.set_ylabel('Return Value', fontsize=12)
            if len(quantiles) < 6:
                ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No quantile data available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Return Distribution Quantiles', fontsize=14, fontweight='bold')
    
    def _plot_embedding_analysis(self, ax, data):
        """Multi-method embedding analysis."""
        embeddings = np.array(data.get('embeddings', []))
        labels = np.array(data.get('noise_labels', []))
        
        if len(embeddings) > 10 and embeddings.shape[0] > 2:
            try:
                # PCA
                pca = PCA(n_components=2)
                pca_reduced = pca.fit_transform(embeddings)
                
                # t-SNE (if enough samples)
                if embeddings.shape[0] > 30:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]-1))
                    tsne_reduced = tsne.fit_transform(embeddings)
                
                # Determine which method to show based on sample size
                if embeddings.shape[0] > 30:
                    reduced_data = tsne_reduced
                    method = 't-SNE'
                    explained_var = 'N/A (non-linear)'
                else:
                    reduced_data = pca_reduced
                    method = 'PCA'
                    explained_var = f'{sum(pca.explained_variance_ratio_):.1%}'
                
                # Separate clean and noisy samples
                clean_mask = labels == -1
                noisy_mask = labels != -1
                
                # Plot clean samples
                if np.any(clean_mask):
                    ax.scatter(reduced_data[clean_mask, 0], reduced_data[clean_mask, 1], 
                              c='lightgreen', s=100, alpha=0.7, edgecolors='green',
                              linewidth=2, label='Clean', marker='o')
                
                # Plot noisy samples colored by noise type
                if np.any(noisy_mask):
                    unique_labels = np.unique(labels[noisy_mask])
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                    
                    for i, label in enumerate(unique_labels):
                        mask = labels == label
                        ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                                  c=[colors[i]], s=100, alpha=0.7, 
                                  edgecolors='black', linewidth=1,
                                  label=f'Type {label}', marker='s')
                
                ax.set_title(f'Embedding Space ({method})\\n{explained_var} Variance Explained', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel(f'Dimension 1', fontsize=12)
                ax.set_ylabel(f'Dimension 2', fontsize=12)
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.3, linestyle='--')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Reduction failed: {str(e)}', 
                       ha='center', va='center', fontsize=10)
                ax.set_title('Embedding Space Analysis', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient embedding data', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Embedding Space Analysis', fontsize=14, fontweight='bold')
    
    def _create_interactive_training_plot(self, history: Dict) -> str:
        """Create interactive Plotly training visualization."""
        epochs = list(range(1, len(history['rewards']) + 1))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Reward', 'Detection Accuracy', 'Classification Accuracy', 'Epsilon Decay'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces for each metric
        fig.add_trace(go.Scatter(x=epochs, y=history['rewards'], 
                               name='Avg Reward', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history['detection_acc'], 
                               name='Detection Acc', line=dict(color='green')), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=history['classification_acc'], 
                               name='Classification Acc', line=dict(color='red')), row=2, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history['epsilon'], 
                               name='Epsilon', line=dict(color='purple')), row=2, col=2)
        
        fig.update_layout(
            title_text="Interactive Training Dashboard",
            height=800,
            showlegend=False
        )
        
        filename = Path(self.config.output_dir) / 'interactive_training.html'
        pyo.plot(fig, filename=str(filename), auto_open=False)
        
        return str(filename)
    
    def _create_interactive_evaluation_plot(self, results: Dict) -> str:
        """Create interactive Plotly evaluation visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance by Noise Type', 'Domain Performance', 'Confusion Matrix', 'Summary')
        )
        
        # Performance by noise type
        noise_types = list(results['by_noise_type'].keys())
        detection_rates = [results['by_noise_type'][nt]['detected'] / max(1, results['by_noise_type'][nt]['total']) 
                          for nt in noise_types]
        
        fig.add_trace(go.Bar(x=noise_types, y=detection_rates, name='Detection Rate'), row=1, col=1)
        
        # Domain performance
        domains = list(results['by_domain'].keys())
        domain_accs = [results['by_domain'][d]['correct_detection'] / max(1, results['by_domain'][d]['total']) 
                      for d in domains]
        
        fig.add_trace(go.Bar(x=domains, y=domain_accs, name='Domain Acc'), row=1, col=2)
        
        # Confusion matrix
        cm = results['confusion_matrix']
        fig.add_trace(go.Heatmap(z=cm, colorscale='Blues'), row=2, col=1)
        
        fig.update_layout(
            title_text="Interactive Evaluation Dashboard",
            height=800
        )
        
        filename = Path(self.config.output_dir) / 'interactive_evaluation.html'
        pyo.plot(fig, filename=str(filename), auto_open=False)
        
        return str(filename)


# ==================== Enhanced Model Components ====================
class QuantileDuelingDQN(nn.Module):
    """Enhanced QR-DQN with regularization and better initialization."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, num_quantiles: int = 51):
        super().__init__()
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        
        # Weight initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        # Shared feature extractor with batch normalization
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ).apply(init_weights)
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_quantiles)
        ).apply(init_weights)
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * num_quantiles)
        ).apply(init_weights)
    
    def forward(self, state):
        batch_size = state.size(0)
        
        # Handle single samples to avoid BatchNorm issues during evaluation
        if batch_size == 1:
            # Temporarily set BatchNorm layers to eval mode for single samples
            with torch.no_grad():
                # Store current training states
                batch_norm_states = []
                for module in self.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        batch_norm_states.append((module, module.training))
                        module.eval()
                
                features = self.feature(state)
                
                # Restore BatchNorm training states
                for module, was_training in batch_norm_states:
                    if was_training:
                        module.train()
        else:
            features = self.feature(state)
        
        # Value: (batch, 1, num_quantiles)
        value = self.value(features).view(batch_size, 1, self.num_quantiles)
        
        # Advantage: (batch, action_dim, num_quantiles)
        advantage = self.advantage(features).view(batch_size, self.action_dim, self.num_quantiles)
        
        # Q(s,a) = V(s) + [A(s,a) - mean(A(s,a'))]
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_dist = value + (advantage - advantage_mean)
        
        return q_dist
    
    def get_q_values(self, state):
        """Get mean Q-values for action selection."""
        q_dist = self.forward(state)
        return q_dist.mean(dim=2)


class EnhancedNoiseDetector(nn.Module):
    """Enhanced noise detection with attention and better feature processing."""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Multi-scale feature extraction
        self.feature_layers = nn.ModuleList([
            nn.Linear(embedding_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        ])
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Detection head with residual connections
        self.detection_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # RL policy for threshold selection
        self.threshold_policy = QuantileDuelingDQN(
            state_dim=embedding_dim + 1,
            action_dim=10,
            hidden_dim=128
        )
    
    def forward(self, embeddings):
        """Forward pass with attention mechanism."""
        # Add attention if batch size > 1
        if embeddings.size(0) > 1:
            # Reshape for attention: (batch, 1, embed_dim)
            embeddings_attn = embeddings.unsqueeze(1)
            attended, _ = self.attention(embeddings_attn, embeddings_attn, embeddings_attn)
            embeddings = attended.squeeze(1)
        
        return self.detection_head(embeddings).squeeze(-1)
    
    def select_threshold(self, embeddings, detection_scores, epsilon=0.1):
        """Enhanced threshold selection with confidence estimation."""
        batch_size = embeddings.size(0)
        state = torch.cat([embeddings, detection_scores.unsqueeze(-1)], dim=-1)
        
        # Epsilon-greedy with confidence-based exploration
        if random.random() < epsilon:
            actions = torch.randint(0, 10, (batch_size,))
        else:
            with torch.no_grad():
                q_values = self.threshold_policy.get_q_values(state)
                actions = q_values.argmax(dim=-1)
        
        # Convert actions to thresholds with noise for exploration
        thresholds = (actions + 1) * 0.1
        if self.training:
            # Add small noise during training
            thresholds += torch.randn_like(thresholds) * 0.01
        
        return thresholds, actions


class BayesianLinear(nn.Module):
    """Enhanced Bayesian Linear Layer with better initialization and regularization."""
    
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 0.1, activation: str = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.activation = activation
        
        # Improved weight initialization
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
        self.kl_div = 0
        
    def reset_parameters(self):
        """Initialize parameters with Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.bias_mu, 0)
        
        # Initialize rho for softplus to give reasonable sigma
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_rho, -3.0)
    
    def forward(self, x):
        # Calculate sigma from rho
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        if self.training:
            # Reparameterization trick
            weight_epsilon = torch.randn_like(weight_sigma)
            bias_epsilon = torch.randn_like(bias_sigma)
            
            weight = self.weight_mu + weight_sigma * weight_epsilon
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            # Use mean weights for inference
            weight = self.weight_mu
            bias = self.bias_mu
            
        # Compute KL divergence
        self.kl_div = (
            self._kl_term(self.weight_mu, weight_sigma) +
            self._kl_term(self.bias_mu, bias_sigma)
        )
        
        output = F.linear(x, weight, bias)
        
        if self.activation:
            if self.activation == 'relu':
                output = F.relu(output)
            elif self.activation == 'sigmoid':
                output = torch.sigmoid(output)
            elif self.activation == 'tanh':
                output = torch.tanh(output)
        
        return output
    
    def _kl_term(self, mu, sigma):
        """Compute KL divergence between posterior and prior."""
        return (
            torch.log(self.prior_sigma / sigma) +
            (sigma**2 + mu**2) / (2 * self.prior_sigma**2) - 0.5
        ).sum()


class EnhancedBayesianNoiseClassifier(nn.Module):
    """Enhanced Bayesian classifier with dropout and better architecture."""
    
    def __init__(self, embedding_dim: int = 768, num_classes: int = 6, hidden_dim: int = 512):
        super().__init__()
        
        self.noise_types = ['OCR', 'ASR', 'Typo', 'Adversarial', 'Semantic', 'Mixed']
        
        # Layer dimensions with gradual reduction
        dims = [embedding_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4, num_classes]
        
        # Bayesian layers with activations
        self.layers = nn.ModuleList([
            BayesianLinear(dims[i], dims[i+1], activation='relu' if i < len(dims)-2 else None)
            for i in range(len(dims)-1)
        ])
        
        self.dropout = nn.Dropout(0.2)
        
        # RL policy for classification decision
        self.classification_policy = QuantileDuelingDQN(
            state_dim=embedding_dim + num_classes,
            action_dim=num_classes,
            hidden_dim=256
        )
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 2:  # Don't dropout before final layer
                x = self.dropout(x)
        return x
    
    def kl_divergence(self):
        """Total KL divergence of the network."""
        return sum(layer.kl_div for layer in self.layers)
    
    def predict_with_uncertainty(self, embeddings, num_samples: int = 10):
        """Enhanced uncertainty estimation with multiple methods."""
        # Monte Carlo Dropout for additional uncertainty
        original_mode = self.training
        
        probs_list = []
        for _ in range(num_samples):
            # Sample with dropout
            self.train()
            logits = self.forward(embeddings)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)
        
        self.train(original_mode)
        
        # Stack predictions
        probs_stack = torch.stack(probs_list)
        
        # Mean prediction
        mean_probs = probs_stack.mean(dim=0)
        
        # Epistemic uncertainty (uncertainty about model parameters)
        epistemic = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        
        # Aleatoric uncertainty (inherent data uncertainty)
        aleatoric = torch.mean(
            -(probs_stack * torch.log(probs_stack + 1e-10)).sum(dim=-1), dim=0
        )
        
        # Total uncertainty
        total_uncertainty = epistemic + aleatoric
        
        return mean_probs, total_uncertainty, epistemic, aleatoric


# ==================== Enhanced Data Generation ====================
class EnhancedDataGenerator:
    """Enhanced synthetic data generator with better noise models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.noise_types = list(config.noise_levels.keys())
        
        # Enhanced clean corpus with more diverse examples
        self.clean_corpus = [
            # Technical texts
            "The quantum algorithm efficiently solves the optimization problem.",
            "Machine learning models require extensive hyperparameter tuning.",
            "The neural network architecture achieved state-of-the-art results.",
            
            # Business texts
            "The quarterly earnings exceeded analyst expectations significantly.",
            "Digital transformation initiatives drive competitive advantage.",
            "Customer satisfaction metrics show consistent improvement.",
            
            # Medical texts
            "The clinical trial demonstrated significant therapeutic efficacy.",
            "Genomic analysis reveals novel biomarkers for disease progression.",
            "Surgical intervention provided optimal patient outcomes.",
            
            # Scientific texts
            "Climate models predict accelerated environmental changes.",
            "The research methodology ensures reproducible experimental results.",
            "Statistical analysis confirms the primary endpoint hypothesis.",
            
            # Legal texts
            "The contract amendment requires board approval and ratification.",
            "Intellectual property rights protect innovative technological developments.",
            "Regulatory compliance necessitates comprehensive documentation protocols.",
            
            # General texts
            "The committee reviewed the proposal and recommended approval.",
            "Educational institutions adapt curricula to meet industry demands.",
            "Renewable energy sources reduce carbon footprint significantly.",
            
            # Common phrases for robustness
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!",
            "The five boxing wizards jump quickly.",
            "Sphinx of black quartz, judge my vow.",
        ]
        
        # Enhanced noise transformations with better linguistic rules
        self._setup_noise_transforms()
    
    def _setup_noise_transforms(self):
        """Setup comprehensive noise transformation rules."""
        
        # Enhanced OCR substitutions
        self.ocr_substitutions = {
            # Numbers and letters
            'O': '0', '0': 'O', 'I': 'l', 'l': 'I', 'S': '5', '5': 'S',
            'Z': '2', '2': 'Z', 'B': '8', '8': 'B', 'G': '6', 'q': 'g',
            # Common confusions
            'rn': 'm', 'cl': 'd', 'vv': 'w', 'vv': 'w', 'ffi': 'fi',
            'ffi': 'fi', 'fl': 'f1', 'fi': 'fl', 'ca': 'ca', 'ch': 'rn',
            # Case sensitive confusions
            'D': '0', 'Q': '0', 'C': 'G', 'P': 'R', 'T': 'F'
        }
        
        # Enhanced ASR homophones with context
        self.asr_homophones = {
            'their': 'there', 'there': 'their', 'your': "you're", "you're": 'your',
            'to': 'too', 'too': 'to', 'its': "it's", "it's": 'its',
            'hear': 'here', 'here': 'hear', 'write': 'right', 'right': 'write',
            'know': 'no', 'no': 'know', 'new': 'knew', 'knew': 'new',
            'weather': 'whether', 'whether': 'weather', 'accept': 'except',
            'affect': 'effect', 'effect': 'affect', 'advice': 'advise',
            'advise': 'advice', 'brake': 'break', 'break': 'brake',
            'buy': 'by', 'by': 'buy', 'cite': 'site', 'site': 'cite',
            'die': 'dye', 'dye': 'die', 'fair': 'fare', 'fare': 'fair',
            'fourth': 'forth', 'forth': 'fourth', 'here': 'hear',
            'hear': 'here', 'hole': 'whole', 'whole': 'hole',
            'meat': 'meet', 'meet': 'meat', 'one': 'won', 'won': 'one',
            'pair': 'pear', 'pear': 'pair', 'peace': 'piece', 'piece': 'peace',
            'principal': 'principle', 'principle': 'principal',
            'stationary': 'stationery', 'stationery': 'stationary',
            'steal': 'steel', 'steel': 'steal', 'tale': 'tail', 'tail': 'tale',
            'two': 'too', 'too': 'two', 'way': 'weigh', 'weigh': 'way',
            'weak': 'week', 'week': 'weak', 'weather': 'whether',
            'whether': 'weather', 'whose': "who's", "who's": 'whose',
            'your': "you're", "you're": 'your'
        }
        
        # Enhanced keyboard proximity matrix
        self.keyboard_neighbors = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcxs', 'e': 'wrdsf',
            'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'uojkl', 'j': 'uikmnh',
            'k': 'iolmj', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'iplk',
            'p': 'ol', 'q': 'wa', 'r': 'etfdg', 's': 'awedxza', 't': 'ryfgh',
            'u': 'yihj', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tughj',
            'z': 'asx', '1': '2q', '2': '13qw', '3': '24ew', '4': '35re',
            '5': '46tr', '6': '57yt', '7': '68yu', '8': '79iu', '9': '80io',
            '0': '9op', '-': '=p', '=': '-p', '[': ']', ']': '[', '\\': ']',
            ';': 'l', '\'': ';', ',': '.', '.': ',', '/': '.', ' ': 'asdf'
        }
        
        # Semantic replacements with domain awareness
        self.semantic_replacements = {
            'increase': 'decrease', 'decrease': 'increase',
            'positive': 'negative', 'negative': 'positive',
            'success': 'failure', 'failure': 'success',
            'advance': 'retreat', 'retreat': 'advance',
            'improve': 'degrade', 'degrade': 'improve',
            'complex': 'simple', 'simple': 'complex',
            'large': 'small', 'small': 'large',
            'significant': 'insignificant', 'insignificant': 'significant',
            'important': 'trivial', 'trivial': 'important',
            'major': 'minor', 'minor': 'major',
            'enhanced': 'degraded', 'degraded': 'enhanced',
            'optimize': 'suboptimize', 'suboptimize': 'optimize',
            'efficient': 'inefficient', 'inefficient': 'efficient',
            'accurate': 'inaccurate', 'inaccurate': 'accurate',
            'reliable': 'unreliable', 'unreliable': 'reliable'
        }
    
    def add_ocr_noise(self, text: str, noise_level: float = None) -> str:
        """Enhanced OCR noise simulation."""
        if noise_level is None:
            noise_level = self.config.noise_levels['OCR']
            
        chars = list(text)
        num_errors = max(1, int(len(chars) * noise_level))
        
        for _ in range(num_errors):
            if random.random() < 0.6 and len(chars) > 0:
                # Character substitution with probability
                idx = random.randint(0, len(chars) - 1)
                char = chars[idx]
                
                # Case-insensitive matching for substitutions
                if char in self.ocr_substitutions:
                    replacement = self.ocr_substitutions[char]
                elif char.lower() in self.ocr_substitutions:
                    # Preserve case
                    replacement = self.ocr_substitutions[char.lower()]
                    if char.isupper():
                        replacement = replacement.upper()
                    else:
                        replacement = replacement.lower()
                else:
                    continue
                    
                chars[idx] = replacement
            elif len(chars) > 1:
                # Random operations
                operation = random.choice(['swap', 'duplicate', 'delete'])
                if operation == 'swap' and len(chars) > 1:
                    idx = random.randint(0, len(chars) - 2)
                    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                elif operation == 'duplicate' and len(chars) > 0:
                    idx = random.randint(0, len(chars) - 1)
                    chars.insert(idx, chars[idx])
                elif operation == 'delete':
                    idx = random.randint(0, len(chars) - 1)
                    chars.pop(idx)
        
        return ''.join(chars)
    
    def add_asr_noise(self, text: str, noise_level: float = None) -> str:
        """Enhanced ASR noise with context awareness."""
        if noise_level is None:
            noise_level = self.config.noise_levels['ASR']
            
        words = text.split()
        num_errors = max(1, int(len(words) * noise_level))
        
        for _ in range(num_errors):
            if len(words) == 0:
                continue
                
            idx = random.randint(0, len(words) - 1)
            word = words[idx].lower().strip('.,!?;:"')
            
            # Check for homophones with case preservation
            if word in self.asr_homophones:
                original_word = words[idx]
                replacement = self.asr_homophones[word]
                
                # Preserve capitalization
                if original_word[0].isupper():
                    replacement = replacement.capitalize()
                
                words[idx] = original_word.replace(word, replacement)
        
        return ' '.join(words)
    
    def add_typo_noise(self, text: str, noise_level: float = None) -> str:
        """Enhanced keyboard typo simulation."""
        if noise_level is None:
            noise_level = self.config.noise_levels['Typo']
            
        chars = list(text)
        num_errors = max(1, int(len(chars) * noise_level))
        
        for _ in range(num_errors):
            if len(chars) == 0:
                continue
                
            idx = random.randint(0, len(chars) - 1)
            char = chars[idx].lower()
            
            # Keyboard substitution
            if char in self.keyboard_neighbors:
                replacement = random.choice(self.keyboard_neighbors[char])
                # Preserve original case
                if chars[idx].isupper():
                    replacement = replacement.upper()
                chars[idx] = replacement
            else:
                # Other typo operations
                operation = random.choice(['duplicate', 'delete', 'insert'])
                
                if operation == 'duplicate' and idx < len(chars) - 1:
                    chars.insert(idx + 1, chars[idx])
                elif operation == 'delete' and len(chars) > 1:
                    chars.pop(idx)
                elif operation == 'insert':
                    chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz'))
        
        return ''.join(chars)
    
    def add_adversarial_noise(self, text: str, noise_level: float = None) -> str:
        """Enhanced adversarial noise with Unicode tricks."""
        if noise_level is None:
            noise_level = self.config.noise_levels['Adversarial']
            
        chars = list(text)
        num_errors = max(1, int(len(chars) * noise_level))
        
        for _ in range(num_errors):
            if len(chars) == 0:
                continue
                
            idx = random.randint(0, len(chars) - 1)
            
            if chars[idx].isalpha() and random.random() < 0.7:
                # Subtle character substitution
                operation = random.choice(['shift', 'unicode', 'zero_width'])
                
                if operation == 'shift':
                    if chars[idx].islower():
                        chars[idx] = chr(ord(chars[idx]) + random.choice([-1, 1]))
                    elif chars[idx].isupper():
                        chars[idx] = chr(ord(chars[idx]) + random.choice([-1, 1]))
                
                elif operation == 'unicode':
                    # Use Unicode look-alikes
                    unicode_map = {'a': 'а', 'e': 'е', 'i': 'і', 'o': 'о', 'u': 'υ'}
                    if chars[idx].lower() in unicode_map:
                        replacement = unicode_map[chars[idx].lower()]
                        if chars[idx].isupper():
                            replacement = replacement.upper()
                        chars[idx] = replacement
                
                elif operation == 'zero_width':
                    # Insert zero-width space (simulated)
                    if random.random() < 0.5:
                        chars.insert(idx, '')
                    else:
                        chars[idx] = '' + chars[idx]
        
        return ''.join(chars)
    
    def add_semantic_noise(self, text: str, noise_level: float = None) -> str:
        """Enhanced semantic noise with context preservation."""
        if noise_level is None:
            noise_level = self.config.noise_levels['Semantic']
            
        words = text.split()
        num_errors = max(1, int(len(words) * noise_level))
        
        for _ in range(num_errors):
            if len(words) == 0:
                continue
                
            idx = random.randint(0, len(words) - 1)
            word = words[idx].lower().strip('.,!?;:"')
            
            # Context-aware semantic replacement
            if word in self.semantic_replacements:
                original_word = words[idx]
                replacement = self.semantic_replacements[word]
                
                # Preserve original formatting and punctuation
                # Find the word in original form with punctuation
                word_start = idx
                word_end = idx + 1
                
                # Preserve leading/trailing punctuation
                prefix = ''
                suffix = ''
                
                if word_start > 0 and not words[word_start - 1][-1].isalnum():
                    prefix = words[word_start - 1][-1]
                
                if word_end < len(words) and not words[word_end][0].isalnum():
                    suffix = words[word_end][0]
                
                # Capitalize if original was capitalized
                if original_word[0].isupper():
                    replacement = replacement.capitalize()
                
                words[idx] = prefix + replacement + suffix
        
        return ' '.join(words)
    
    def add_mixed_noise(self, text: str, noise_level: float = None) -> str:
        """Enhanced mixed noise with intelligent combination."""
        if noise_level is None:
            noise_level = self.config.noise_levels['Mixed']
            
        # Select 2-3 random noise types with probability weighting
        noise_funcs = [
            (self.add_ocr_noise, self.config.noise_probabilities['OCR']),
            (self.add_asr_noise, self.config.noise_probabilities['ASR']),
            (self.add_typo_noise, self.config.noise_probabilities['Typo']),
            (self.add_adversarial_noise, self.config.noise_probabilities['Adversarial']),
            (self.add_semantic_noise, self.config.noise_probabilities['Semantic'])
        ]
        
        # Weighted selection
        selected_funcs = []
        available_funcs = [func for func, prob in noise_funcs]
        
        num_types = random.choices([2, 3], weights=[0.6, 0.4])[0]
        
        # Select without replacement
        selected_funcs = random.sample(available_funcs, num_types)
        
        # Apply with reduced noise levels
        result = text
        for func in selected_funcs:
            result = func(result, noise_level * 0.4)
        
        return result
    
    def generate_sample(self, noise_type: str = None) -> Dict[str, Any]:
        """Generate a single training sample with metadata."""
        # Select clean text
        clean_text = random.choice(self.clean_corpus)
        
        # Decide whether to add noise
        if noise_type is None:
            is_noisy = random.random() < 0.7
            if is_noisy:
                # Weighted noise type selection
                noise_type = random.choices(
                    list(self.config.noise_probabilities.keys()),
                    weights=list(self.config.noise_probabilities.values())
                )[0]
        else:
            is_noisy = True
        
        if is_noisy:
            # Generate noisy version
            noise_func = getattr(self, f'add_{noise_type.lower()}_noise')
            noisy_text = noise_func(clean_text)
            label = list(self.config.noise_probabilities.keys()).index(noise_type)
        else:
            noisy_text = clean_text
            label = -1
        
        return {
            'text': noisy_text,
            'clean_text': clean_text,
            'is_noisy': int(is_noisy),
            'noise_type': label,
            'noise_type_name': noise_type if is_noisy else 'Clean',
            'complexity_score': self._calculate_complexity(noisy_text),
            'word_count': len(noisy_text.split()),
            'char_count': len(noisy_text),
            'noise_level_applied': self.config.noise_levels.get(noise_type, 0) if is_noisy else 0
        }
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score for better training."""
        words = text.split()
        
        # Factors affecting complexity
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        unique_words = len(set(words))
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        
        # Normalize and combine
        complexity = (
            min(word_count / 20, 1.0) * 0.3 +  # Word count factor
            min(avg_word_length / 10, 1.0) * 0.3 +  # Average word length
            min(unique_words / max(word_count, 1), 1.0) * 0.2 +  # Vocabulary diversity
            min(punctuation_count / 10, 1.0) * 0.2  # Punctuation factor
        )
        
        return complexity
    
    def generate_batch(self, batch_size: int = 32, noise_type: str = None) -> List[Dict[str, Any]]:
        """Generate a batch with balanced noise types."""
        return [self.generate_sample(noise_type) for _ in range(batch_size)]
    
    def generate_balanced_batch(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Generate a batch with balanced representation of noise types."""
        if batch_size <= len(self.noise_types):
            # For small batches, ensure at least one sample per noise type
            samples = []
            noise_types_list = list(self.noise_types) + ['Clean']
            
            for i in range(batch_size):
                noise_type = noise_types_list[i % len(noise_types_list)]
                if noise_type == 'Clean':
                    sample = self.generate_sample(None)
                    sample['noise_type'] = -1
                    sample['noise_type_name'] = 'Clean'
                else:
                    sample = self.generate_sample(noise_type)
                samples.append(sample)
            
            return samples
        else:
            # For larger batches, ensure proportional representation
            samples_per_type = batch_size // (len(self.noise_types) + 1)
            remaining = batch_size % (len(self.noise_types) + 1)
            
            samples = []
            
            # Add samples for each noise type
            for noise_type in self.noise_types:
                for _ in range(samples_per_type):
                    sample = self.generate_sample(noise_type)
                    samples.append(sample)
            
            # Add clean samples
            for _ in range(samples_per_type):
                sample = self.generate_sample(None)
                sample['noise_type'] = -1
                sample['noise_type_name'] = 'Clean'
                samples.append(sample)
            
            # Add remaining samples
            for _ in range(remaining):
                sample = self.generate_sample()
                samples.append(sample)
            
            return samples[:batch_size]


# ==================== Enhanced Evaluation System ====================
class EnhancedRealWorldDataset:
    """Enhanced real-world dataset with better domain coverage."""
    
    def __init__(self):
        self.domains = {
            'Customer Support': [
                "My order #12345 hasn't arrived yet, please help.",
                "I want to return the item I purchased yesterday.",
                "The application keeps crashing when I try to login.",
                "How do I reset my password? I didn't receive the email.",
                "Is there a discount available for students?",
                "The refund process is taking longer than expected.",
                "Customer service was very helpful and responsive.",
                "I need to modify my subscription plan urgently.",
                "The delivery tracking shows it's stuck at the warehouse.",
                "Technical support couldn't resolve my connectivity issue."
            ],
            'Social Media': [
                "Just saw the new movie, it was amazing! #recommend",
                "Can't believe this is happening right now. SMH.",
                "Check out my new photo from the vacation!",
                "Does anyone know a good restaurant in downtown?",
                "The game last night was absolutely insane!",
                "Feeling blessed and grateful for everything in life.",
                "That concert was absolutely electrifying tonight!",
                "New workout routine is killing me but results show.",
                "Coffee shop recommendations in the city center please.",
                "Just finished reading the most thought-provoking book."
            ],
            'Medical': [
                "Patient reports severe migraine and nausea since yesterday.",
                "Prescribed 500mg of Amoxicillin twice daily.",
                "Blood pressure is slightly elevated at 140/90.",
                "Symptoms include fever, cough, and fatigue.",
                "Follow-up appointment scheduled for next Tuesday.",
                "Surgical procedure went according to plan successfully.",
                "Laboratory results show significant improvement markers.",
                "Patient exhibits signs of allergic reaction to medication.",
                "Diagnostic imaging reveals no abnormalities detected.",
                "Treatment protocol modified based on recent test results."
            ],
            'Technical': [
                "The server process terminated with exit code 1 due to OOM.",
                "Please update the database schema to include the new column.",
                "Python 3.11 introduces significant performance improvements.",
                "The API endpoint returns a 404 error for valid requests.",
                "Memory leak detected in the background worker service.",
                "Container orchestration simplifies deployment workflows.",
                "Microservices architecture improves system scalability.",
                "Real-time data streaming enables instant analytics.",
                "Machine learning pipeline automates feature engineering.",
                "Cloud infrastructure provides elastic compute resources."
            ],
            'Academic': [
                "Research methodology section requires additional citations.",
                "Literature review demonstrates comprehensive understanding.",
                "Statistical analysis confirms hypothesis validation criteria.",
                "Peer review process ensures publication quality standards.",
                "Data visualization effectively communicates key findings.",
                "Graduate program applications due next month deadline.",
                "Seminar presentation covers advanced theoretical concepts.",
                "Laboratory experiments yield reproducible experimental data.",
                "Conference presentation abstracts accepted for publication.",
                "Thesis defense scheduled for end of semester evaluation."
            ],
            'Legal': [
                "Contract amendment requires board ratification approval.",
                "Intellectual property rights protect innovative technology.",
                "Regulatory compliance necessitates comprehensive documentation.",
                "Litigation strategy focuses on precedent case analysis.",
                "Due diligence process identifies potential risk factors.",
                "Employment agreement includes non-disclosure clauses.",
                "Securities regulations govern public company disclosures.",
                "Corporate governance standards ensure fiduciary responsibilities.",
                "Mergers and acquisitions require antitrust clearance review.",
                "Litigation settlement terms remain confidential pending approval."
            ]
        }
    
    def get_all_samples(self) -> Dict[str, List[str]]:
        """Return all samples organized by domain."""
        return self.domains
    
    def get_domain_samples(self, domain: str, count: int = None) -> List[str]:
        """Get samples from a specific domain."""
        if domain not in self.domains:
            raise ValueError(f"Domain '{domain}' not found. Available: {list(self.domains.keys())}")
        
        samples = self.domains[domain]
        if count is None:
            return samples
        return samples[:count]
    
    def get_balanced_samples(self, samples_per_domain: int = 3) -> Dict[str, List[str]]:
        """Get balanced samples across all domains."""
        balanced_data = {}
        for domain, samples in self.domains.items():
            balanced_data[domain] = samples[:samples_per_domain]
        return balanced_data
    
    def add_custom_domain(self, domain_name: str, samples: List[str]):
        """Add a custom domain with samples."""
        self.domains[domain_name] = samples
        logger.info(f"Added custom domain '{domain_name}' with {len(samples)} samples")


# ==================== Enhanced Trainer ====================
class EnhancedHierarchicalRLTrainer:
    """Enhanced trainer with better architecture and monitoring."""
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics_logger = MetricsLogger(config)
        
        # Initialize components
        with timer("Initializing models"):
            self.detector = EnhancedNoiseDetector(config.embedding_dim, config.hidden_dim // 2)
            self.classifier = EnhancedBayesianNoiseClassifier(
                config.embedding_dim, 
                len(config.noise_levels), 
                config.hidden_dim
            )
            
            # Calculate state dimensions for router and filter
            state_dim = config.embedding_dim + 1 + len(config.noise_levels) + 2  # emb + det + probs + unc
            
            self.filter_selector = QuantileDuelingDQN(
                state_dim=state_dim,
                action_dim=32,  # Filter combinations
                hidden_dim=config.hidden_dim // 2
            )
            
            self.router = QuantileDuelingDQN(
                state_dim=state_dim,
                action_dim=15,  # Number of models
                hidden_dim=config.hidden_dim
            )
        
        # Enhanced data generator
        self.data_generator = EnhancedDataGenerator(config)
        self.real_world_data = EnhancedRealWorldDataset()
        
        # Enhanced replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.buffer_size,
            alpha=config.alpha,
            beta=config.beta
        )
        
        # Optimizers with learning rate scheduling
        self.detector_optimizer = torch.optim.AdamW(
            self.detector.parameters(), lr=config.learning_rate, weight_decay=1e-4
        )
        self.classifier_optimizer = torch.optim.AdamW(
            self.classifier.parameters(), lr=config.learning_rate, weight_decay=1e-4
        )
        self.router_optimizer = torch.optim.AdamW(
            self.router.parameters(), lr=config.learning_rate, weight_decay=1e-4
        )
        
        # Learning rate schedulers
        self.detector_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.detector_optimizer, mode='max', factor=0.5, patience=10
        )
        self.classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.classifier_optimizer, mode='max', factor=0.5, patience=10
        )
        
        # Training state
        self.current_episode = 0
        self.best_accuracy = 0.0
        self.patience_counter = 0
        
        # Visualization
        self.visualizer = TrainingVisualizer(config)
        
        logger.info(f"Enhanced trainer initialized successfully")
        self._log_model_summary()
    
    def _log_model_summary(self):
        """Log detailed model summary."""
        logger.info("="*60)
        logger.info("MODEL ARCHITECTURE SUMMARY")
        logger.info("="*60)
        
        total_params = 0
        
        # Detector
        detector_params = sum(p.numel() for p in self.detector.parameters())
        total_params += detector_params
        logger.info(f"Noise Detector: {detector_params:,} parameters")
        
        # Classifier
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total_params += classifier_params
        logger.info(f"Bayesian Classifier: {classifier_params:,} parameters")
        
        # Router
        router_params = sum(p.numel() for p in self.router.parameters())
        total_params += router_params
        logger.info(f"Model Router: {router_params:,} parameters")
        
        # Filter Selector
        filter_params = sum(p.numel() for p in self.filter_selector.parameters())
        total_params += filter_params
        logger.info(f"Filter Selector: {filter_params:,} parameters")
        
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info("="*60)
    
    def quantile_huber_loss(self, current_quantiles, target_quantiles, actions, rewards, dones, gamma=0.99):
        """Enhanced quantile huber loss with gradient clipping."""
        batch_size, action_dim, num_quantiles = current_quantiles.shape
        
        # Select quantiles for taken actions
        actions = actions.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, num_quantiles)
        current_quantiles = current_quantiles.gather(1, actions).squeeze(1)
        
        # Compute target distribution
        rewards = rewards.unsqueeze(1).expand(batch_size, num_quantiles)
        dones = dones.unsqueeze(1).expand(batch_size, num_quantiles)
        target_quantiles = rewards + gamma * (1 - dones) * target_quantiles
        
        # Huber loss computation
        diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        
        k = 1.0
        abs_diff = diff.abs()
        huber_loss = torch.where(
            abs_diff < k,
            0.5 * diff.pow(2),
            k * (abs_diff - 0.5 * k)
        )
        
        # Quantile regression loss
        tau = (torch.arange(num_quantiles).float() + 0.5) / num_quantiles
        tau = tau.view(1, num_quantiles, 1)
        
        loss = (torch.abs(tau - (diff.detach() < 0).float()) * huber_loss).mean()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.router.parameters(), max_norm=1.0)
        
        return loss
    
    def train_episode(self, batch_size: int = None, balanced_batch: bool = True):
        """Enhanced training episode with better metrics and validation."""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        with timer(f"Training episode {self.current_episode + 1}"):
            total_reward = 0
            
            # Generate batch (balanced for better learning)
            if balanced_batch:
                batch = self.data_generator.generate_balanced_batch(batch_size)
            else:
                batch = self.data_generator.generate_batch(batch_size)
            
            # Process batch
            texts = [s['text'] for s in batch]
            embeddings = self.get_embeddings(texts)
            
            # 1. Enhanced Noise Detection
            detection_scores = self.detector(embeddings)
            thresholds, threshold_actions = self.detector.select_threshold(
                embeddings, detection_scores, self.config.epsilon
            )
            predicted_noisy = (detection_scores > thresholds).float()
            
            # 2. Enhanced Noise Classification (Bayesian)
            noise_probs, total_uncertainty, epistemic, aleatoric = \
                self.classifier.predict_with_uncertainty(embeddings, self.config.num_samples_uncertainty)
            predicted_types = noise_probs.argmax(dim=-1)
            
            # 3. Filter Selection State
            uncertainty_expanded = torch.cat([epistemic.unsqueeze(1), aleatoric.unsqueeze(1)], dim=1)
            filter_state = torch.cat([
                embeddings, 
                detection_scores.unsqueeze(1), 
                noise_probs, 
                uncertainty_expanded
            ], dim=-1)
            
            # 4. Model Routing State
            route_state = filter_state
            
            # Batch operations for efficiency
            with torch.no_grad():
                # Filter selection
                filter_q_values = self.filter_selector.get_q_values(filter_state)
                filter_actions = filter_q_values.argmax(dim=-1)
                
                # Model routing
                route_q_values = self.router.get_q_values(route_state)
                route_actions = route_q_values.argmax(dim=-1)
            
            # Calculate rewards and metrics
            episode_metrics = self._calculate_episode_metrics(
                batch, predicted_noisy, predicted_types, 
                detection_scores, route_actions
            )
            
            total_reward = episode_metrics['total_reward']
            
            # 5. Optimization Steps
            
            # Bayesian Classifier Optimization (ELBO)
            target_types = torch.tensor([
                s['noise_type'] if s['is_noisy'] == 1 else 0 
                for s in batch
            ])
            
            noise_logits = self.classifier(embeddings)
            ce_loss = F.cross_entropy(noise_logits, target_types, reduction='mean')
            kl_loss = self.classifier.kl_divergence() / batch_size
            
            classifier_loss = ce_loss + self.config.kl_weight * kl_loss
            
            self.classifier_optimizer.zero_grad()
            classifier_loss.backward(retain_graph=True)
            
            # Gradient clipping for Bayesian layers
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            self.classifier_optimizer.step()
            
            # 6. RL Optimization
            current_quantiles = self.router(route_state.detach())
            mean_q = current_quantiles.mean(dim=2)
            actions = mean_q.argmax(dim=1)
            
            # Target quantiles (simplified bandit setting)
            episode_reward_mean = torch.tensor([total_reward / batch_size] * batch_size)
            dones_tensor = torch.ones(batch_size)
            
            # Create target distribution
            target_quantiles = episode_reward_mean.unsqueeze(1).expand(batch_size, self.config.num_quantiles)
            
            router_loss = self.quantile_huber_loss(
                current_quantiles, target_quantiles, actions, 
                episode_reward_mean, dones_tensor
            )
            
            self.router_optimizer.zero_grad()
            router_loss.backward()
            self.router_optimizer.step()
            
            # 7. Epsilon decay
            self.config.epsilon = max(
                self.config.epsilon_min, 
                self.config.epsilon * self.config.epsilon_decay
            )
            
            # 8. Learning rate scheduling
            if episode_metrics['detection_accuracy'] > 0.6:
                self.detector_scheduler.step(episode_metrics['detection_accuracy'])
                self.classifier_scheduler.step(episode_metrics['classification_accuracy'])
            
            # 9. Update metrics
            self.current_episode += 1
            
            # Store episode metrics
            self.metrics_logger.log_dict({
                'episode_reward': episode_metrics['avg_reward'],
                'detection_accuracy': episode_metrics['detection_accuracy'],
                'classification_accuracy': episode_metrics['classification_accuracy'],
                'total_uncertainty_mean': episode_metrics['avg_uncertainty'],
                'epsilon': self.config.epsilon,
                'learning_rate_detector': self.detector_optimizer.param_groups[0]['lr'],
                'learning_rate_classifier': self.classifier_optimizer.param_groups[0]['lr']
            })
            
            # Check for improvement
            if episode_metrics['detection_accuracy'] > self.best_accuracy:
                self.best_accuracy = episode_metrics['detection_accuracy']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            return episode_metrics
    
    def _calculate_episode_metrics(self, batch, predicted_noisy, predicted_types, 
                                 detection_scores, route_actions):
        """Calculate comprehensive episode metrics."""
        episode_correct_detection = 0
        episode_correct_classification = 0
        total_reward = 0
        uncertainty_sum = 0
        
        model_names = [
            'GPT-4', 'GPT-3.5', 'Claude-3-Sonnet', 'Claude-3-Haiku', 
            'LLaMA4-70B', 'Mixtral-8x7B', 'LLaMA-13B', 'Qwen-7B',
            'BERT', 'RoBERTa', 'DeBERTa', 'T5', 'BART', 'Pegasus', 'GPT-2'
        ]
        
        for i, sample in enumerate(batch):
            # Reward components
            detection_reward = 0.0
            classification_reward = 0.0
            efficiency_reward = 0.0
            
            is_noisy = sample['is_noisy']
            pred_noisy = predicted_noisy[i].item()
            
            # Detection Reward
            if pred_noisy == is_noisy:
                detection_reward = 0.5
                episode_correct_detection += 1
            else:
                detection_reward = -0.2
            
            # Classification Reward (only if noisy and detected as noisy)
            if is_noisy and pred_noisy:
                pred_type = predicted_types[i].item()
                if pred_type == sample['noise_type']:
                    classification_reward = 0.5
                    episode_correct_classification += 1
            
            # Efficiency Reward
            model_name = model_names[route_actions[i]]
            efficiency_reward = self.compute_efficiency_reward(model_name, sample)
            
            # Total Reward
            reward = detection_reward + classification_reward + efficiency_reward * 0.3
            total_reward += reward
        
        # Calculate averages
        avg_reward = total_reward / len(batch)
        detection_acc = episode_correct_detection / len(batch)
        
        noisy_samples = sum(s['is_noisy'] for s in batch)
        classification_acc = episode_correct_classification / max(1, noisy_samples)
        
        # Calculate uncertainty statistics
        uncertainty_sum = torch.sum(self.get_uncertainty_statistics(batch, predicted_noisy, predicted_types))
        
        return {
            'avg_reward': avg_reward,
            'detection_accuracy': detection_acc,
            'classification_accuracy': classification_acc,
            'total_reward': total_reward,
            'avg_uncertainty': uncertainty_sum / len(batch)
        }
    
    def get_uncertainty_statistics(self, batch, predicted_noisy, predicted_types):
        """Calculate uncertainty statistics for the batch."""
        uncertainties = []
        
        for i, sample in enumerate(batch):
            # Simple uncertainty proxy based on prediction confidence
            is_noisy = sample['is_noisy']
            pred_noisy = predicted_noisy[i].item()
            
            if is_noisy and pred_noisy:
                # High uncertainty for misclassified noisy samples
                pred_type = predicted_types[i].item()
                if pred_type != sample['noise_type']:
                    uncertainties.append(0.8)  # High uncertainty
                else:
                    uncertainties.append(0.2)  # Low uncertainty
            elif not is_noisy and not pred_noisy:
                uncertainties.append(0.1)  # Very low uncertainty for correct clean detection
            else:
                uncertainties.append(0.6)  # Medium uncertainty for wrong predictions
        
        return torch.tensor(uncertainties)
    
    def compute_efficiency_reward(self, model_name: str, sample: Dict) -> float:
        """Enhanced efficiency reward with domain-specific considerations."""
        # Enhanced model performance characteristics
        model_specs = {
            'GPT-4': {'latency': 2000, 'cost': 0.03, 'accuracy': 0.95, 'complexity': 'high'},
            'GPT-3.5': {'latency': 500, 'cost': 0.002, 'accuracy': 0.88, 'complexity': 'medium'},
            'Claude-3-Sonnet': {'latency': 1500, 'cost': 0.015, 'accuracy': 0.92, 'complexity': 'high'},
            'Claude-3-Haiku': {'latency': 400, 'cost': 0.001, 'accuracy': 0.85, 'complexity': 'medium'},
            'LLaMA4-70B': {'latency': 1200, 'cost': 0.008, 'accuracy': 0.90, 'complexity': 'high'},
            'Mixtral-8x7B': {'latency': 800, 'cost': 0.004, 'accuracy': 0.87, 'complexity': 'high'},
            'LLaMA-13B': {'latency': 300, 'cost': 0.001, 'accuracy': 0.82, 'complexity': 'medium'},
            'Qwen-7B': {'latency': 200, 'cost': 0.0005, 'accuracy': 0.80, 'complexity': 'medium'},
            'BERT': {'latency': 50, 'cost': 0.0001, 'accuracy': 0.75, 'complexity': 'low'},
            'RoBERTa': {'latency': 60, 'cost': 0.0001, 'accuracy': 0.76, 'complexity': 'low'},
            'DeBERTa': {'latency': 80, 'cost': 0.0001, 'accuracy': 0.78, 'complexity': 'low'},
            'T5': {'latency': 150, 'cost': 0.0003, 'accuracy': 0.81, 'complexity': 'medium'},
            'BART': {'latency': 120, 'cost': 0.0002, 'accuracy': 0.79, 'complexity': 'medium'},
            'Pegasus': {'latency': 140, 'cost': 0.0003, 'accuracy': 0.80, 'complexity': 'medium'},
            'GPT-2': {'latency': 100, 'cost': 0.0001, 'accuracy': 0.73, 'complexity': 'low'},
        }
        
        specs = model_specs.get(model_name, {'latency': 500, 'cost': 0.002, 'accuracy': 0.8, 'complexity': 'medium'})
        
        # Normalize metrics
        latency_penalty = specs['latency'] / 2000.0
        cost_penalty = specs['cost'] / 0.03
        
        # Complexity-aware base accuracy
        complexity_bonus = {'low': 0.05, 'medium': 0.02, 'high': 0.0}[specs['complexity']]
        base_accuracy = specs['accuracy'] + complexity_bonus
        
        # Sample complexity adjustment
        sample_complexity = sample.get('complexity_score', 0.5)
        complexity_factor = 1.0 - (sample_complexity * 0.2)  # Lower accuracy for complex samples
        
        # Total reward calculation
        total_reward = base_accuracy * complexity_factor \
                      - (0.2 * latency_penalty) \
                      - (0.15 * cost_penalty)
        
        return max(0.0, total_reward)  # Ensure non-negative reward
    
    def get_embeddings(self, inputs: List[str]) -> torch.Tensor:
        """Enhanced embedding function with better feature engineering."""
        embeddings = []
        
        for text in inputs:
            # Initialize feature vector
            features = torch.zeros(self.config.embedding_dim)
            
            # Basic text statistics
            text_len = len(text)
            word_count = len(text.split())
            
            features[0] = min(text_len / 200.0, 1.0)  # Normalized length
            features[1] = min(word_count / 50.0, 1.0)  # Normalized word count
            features[2] = sum(c.isupper() for c in text) / max(text_len, 1)  # Uppercase ratio
            features[3] = sum(c.isdigit() for c in text) / max(text_len, 1)  # Digit ratio
            features[4] = sum(c in '.,!?;:' for c in text) / max(text_len, 1)  # Punctuation ratio
            features[5] = text.count(' ') / max(text_len, 1)  # Space ratio
            
            # Character n-gram features with better distribution
            char_features_start = 50
            char_features_count = 300
            
            for i, char in enumerate(text[:150]):  # Increased character coverage
                if i < char_features_count:
                    idx = char_features_start + (hash(char.lower()) % char_features_count)
                    features[idx] += 0.01
            
            # Word-level features with TF-IDF-like weighting
            word_features_start = 350
            word_features_count = 200
            
            words = text.lower().split()
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            for word, freq in word_freq.items():
                if len(word) > 2:  # Filter short words
                    tf_score = freq / len(words)  # Term frequency
                    idx = word_features_start + (hash(word) % word_features_count)
                    features[idx] += tf_score * 0.05
            
            # Bigram features
            bigram_features_start = 550
            bigram_features_count = 150
            
            for i in range(len(words) - 1):
                bigram = f"{words[i]}_{words[i+1]}"
                idx = bigram_features_start + (hash(bigram) % bigram_features_count)
                features[idx] += 0.02
            
            # Part-of-speech like features (simplified)
            pos_features_start = 700
            
            # Count different types of words
            proper_nouns = sum(1 for word in words if word[0].isupper() and len(word) > 1)
            long_words = sum(1 for word in words if len(word) > 6)
            short_words = sum(1 for word in words if len(word) <= 3)
            
            features[pos_features_start] = proper_nouns / max(len(words), 1)
            features[pos_features_start + 1] = long_words / max(len(words), 1)
            features[pos_features_start + 2] = short_words / max(len(words), 1)
            
            # Sentence structure features
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            features[pos_features_start + 3] = sentence_count / max(len(text.split('.')) + len(text.split('!')) + len(text.split('?')), 1)
            
            # Normalize features
            norm = features.norm()
            if norm > 0:
                features = features / norm
            
            embeddings.append(features)
        
        return torch.stack(embeddings)
    
    def evaluate_model(self, num_samples_per_domain: int = None) -> Dict:
        """Enhanced model evaluation with comprehensive metrics."""
        if num_samples_per_domain is None:
            num_samples_per_domain = self.config.num_test_samples
        
        logger.info(f"Starting comprehensive evaluation with {num_samples_per_domain} samples per domain")
        
        all_samples = self.real_world_data.get_balanced_samples(num_samples_per_domain)
        
        results = {
            'total_samples': 0,
            'correct_detection': 0,
            'correct_classification': 0,
            'detected_as_noisy': 0,
            'actually_noisy': 0,
            'by_noise_type': {nt: {'total': 0, 'detected': 0, 'classified_correct': 0} 
                            for nt in self.data_generator.noise_types},
            'by_domain': {d: {'total': 0, 'correct_detection': 0, 'detection_scores': []} 
                         for d in self.real_world_data.domains.keys()},
            'confusion_matrix': np.zeros((len(self.data_generator.noise_types), 
                                       len(self.data_generator.noise_types)), dtype=int),
            'uncertainty_clean': [],
            'uncertainty_noisy': [],
            'quantiles': [],
            'embeddings': [],
            'noise_labels': [],
            'detailed_predictions': []
        }
        
        # Map noise types to indices
        noise_map = {nt: i for i, nt in enumerate(self.data_generator.noise_types)}
        
        for domain, samples in all_samples.items():
            logger.info(f"Evaluating domain: {domain}")
            
            for text in samples:
                # Test with clean + all noise types
                test_cases = [('Clean', text, False)]
                for nt in self.data_generator.noise_types:
                    noisy_text = getattr(self.data_generator, f'add_{nt.lower()}_noise')(text)
                    test_cases.append((nt, noisy_text, True))
                
                for noise_type, test_text, is_noisy in test_cases:
                    with timer(f"Evaluating {noise_type} sample"):
                        metrics = self._evaluate_single_sample(test_text, is_noisy, noise_type, domain)
                        
                        # Update comprehensive results
                        self._update_evaluation_results(results, metrics, noise_type, noise_map, domain)
        
        # Calculate additional metrics
        self._calculate_additional_metrics(results)
        
        logger.info(f"Evaluation completed: {results['total_samples']} samples processed")
        return results
    
    def _evaluate_single_sample(self, text: str, is_noisy: bool, noise_type: str, domain: str) -> Dict:
        """Enhanced single sample evaluation with detailed metrics."""
        embeddings = self.get_embeddings([text])
        
        # Detection
        detection_score = self.detector(embeddings).item()
        threshold, _ = self.detector.select_threshold(embeddings, torch.tensor([detection_score]), epsilon=0.0)
        predicted_noisy = detection_score > threshold
        
        # Classification with uncertainty
        noise_probs, total_uncertainty, epistemic, aleatoric = \
            self.classifier.predict_with_uncertainty(embeddings)
        predicted_class_idx = noise_probs.argmax(dim=-1).item()
        confidence = noise_probs.max().item()
        
        # Routing analysis
        uncertainty_expanded = torch.cat([epistemic.unsqueeze(1), aleatoric.unsqueeze(1)], dim=1)
        filter_state = torch.cat([embeddings, torch.tensor([detection_score]).unsqueeze(1), 
                                noise_probs, uncertainty_expanded], dim=-1)
        
        with torch.no_grad():
            quantiles = self.router(filter_state)
            mean_q = quantiles.mean(dim=2)
            best_action = mean_q.argmax(dim=1)
            best_quantiles = quantiles[0, best_action, :].squeeze()
        
        # Map noise type to index
        noise_map = {nt: i for i, nt in enumerate(self.data_generator.noise_types)}
        true_label_idx = noise_map.get(noise_type, -1)
        
        return {
            'detection_correct': predicted_noisy == is_noisy,
            'predicted_noisy': predicted_noisy,
            'detection_score': detection_score,
            'threshold': threshold,
            'classification_correct': predicted_class_idx == true_label_idx if is_noisy else False,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'total_uncertainty': total_uncertainty.item(),
            'epistemic_uncertainty': epistemic.item(),
            'aleatoric_uncertainty': aleatoric.item(),
            'embedding': embeddings[0],
            'quantiles': best_quantiles,
            'true_label_idx': true_label_idx,
            'noise_type': noise_type,
            'domain': domain,
            'text': text[:100] + "..." if len(text) > 100 else text
        }
    
    def _update_evaluation_results(self, results, metrics, noise_type, noise_map, domain):
        """Update results dictionary with evaluation metrics."""
        results['total_samples'] += 1
        
        # Detection tracking
        if metrics['detection_correct']:
            results['correct_detection'] += 1
        
        if metrics['predicted_noisy']:
            results['detected_as_noisy'] += 1
        
        if metrics.get('is_noisy', False):  # Assuming is_noisy was added to metrics
            results['actually_noisy'] += 1
        
        # Domain tracking
        results['by_domain'][domain]['total'] += 1
        results['by_domain'][domain]['detection_scores'].append(metrics['detection_score'])
        
        if metrics['detection_correct']:
            results['by_domain'][domain]['correct_detection'] += 1
        
        # Noise type tracking for noisy samples
        if noise_type != 'Clean':
            true_idx = noise_map[noise_type]
            results['by_noise_type'][noise_type]['total'] += 1
            
            if metrics['predicted_noisy']:
                results['by_noise_type'][noise_type]['detected'] += 1
                
                # Confusion matrix (only for detected samples)
                pred_idx = metrics['predicted_class_idx']
                results['confusion_matrix'][true_idx, pred_idx] += 1
                
                if metrics['classification_correct']:
                    results['correct_classification'] += 1
                    results['by_noise_type'][noise_type]['classified_correct'] += 1
            
            # Advanced metrics
            results['uncertainty_noisy'].append(metrics['total_uncertainty'])
            results['embeddings'].append(metrics['embedding'].numpy())
            results['noise_labels'].append(true_idx)
        else:
            results['uncertainty_clean'].append(metrics['total_uncertainty'])
            results['embeddings'].append(metrics['embedding'].numpy())
            results['noise_labels'].append(-1)
        
        # Quantiles and detailed predictions
        if len(results['quantiles']) < 20:  # Limit to prevent memory issues
            results['quantiles'].append(metrics['quantiles'].numpy())
        
        results['detailed_predictions'].append({
            'text': metrics['text'],
            'domain': domain,
            'noise_type': noise_type,
            'is_noisy': noise_type != 'Clean',
            'detection_score': metrics['detection_score'],
            'threshold': metrics['threshold'],
            'predicted_noisy': metrics['predicted_noisy'],
            'detection_correct': metrics['detection_correct'],
            'classification_correct': metrics['classification_correct'],
            'confidence': metrics['confidence'],
            'uncertainty': metrics['total_uncertainty']
        })
    
    def _calculate_additional_metrics(self, results):
        """Calculate additional evaluation metrics."""
        # Domain-specific detection statistics
        for domain in results['by_domain']:
            scores = results['by_domain'][domain]['detection_scores']
            if scores:
                results['by_domain'][domain]['mean_score'] = np.mean(scores)
                results['by_domain'][domain]['std_score'] = np.std(scores)
                results['by_domain'][domain]['min_score'] = np.min(scores)
                results['by_domain'][domain]['max_score'] = np.max(scores)
        
        # Overall statistics
        results['detection_rate'] = results['detected_as_noisy'] / results['total_samples']
        results['actual_noise_rate'] = results['actually_noisy'] / results['total_samples']
        
        if results['detected_as_noisy'] > 0:
            results['precision'] = results['correct_classification'] / results['detected_as_noisy']
        else:
            results['precision'] = 0.0
        
        if results['actually_noisy'] > 0:
            results['recall'] = results['correct_classification'] / results['actually_noisy']
        else:
            results['recall'] = 0.0
        
        if results['precision'] + results['recall'] > 0:
            results['f1_score'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])
        else:
            results['f1_score'] = 0.0


# ==================== Enhanced Experience Replay ====================
class PrioritizedReplayBuffer:
    """Enhanced prioritized experience replay with better prioritization."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 1e-6
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Add small epsilon to prevent zero priorities
        self.eps = 1e-6
        
    def push(self, state, action, reward, next_state, done, td_error: float = None):
        """Store transition with priority based on TD error."""
        # Calculate priority if not provided
        if td_error is None:
            max_priority = self.priorities.max() if self.size > 0 else 1.0
        else:
            max_priority = abs(td_error) + self.eps
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities[self.position] = max_priority
            self.size += 1
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch with importance sampling weights."""
        if self.size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]
        
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Compute importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        
        # Get batch
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            indices,
            torch.FloatTensor(weights)
        )
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on new TD errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = abs(priority) + self.eps
    
    def __len__(self):
        return self.size


# ==================== Main Execution ====================
def main():
    """Enhanced main execution with comprehensive configuration."""
    # Setup
    setup_matplotlib_for_plotting()
    
    # Configuration
    config = Config(
        num_episodes=150,  # Increased for better training
        batch_size=48,     # Larger batch size
        learning_rate=1e-4,
        output_dir="enhanced_outputs",
        save_plots=True,
        interactive_plots=True,  # Enable interactive plots
        plot_dpi=200  # Higher resolution
    )
    
    logger.info("="*80)
    logger.info("ENHANCED HIERARCHICAL RL NOISE FILTERING SYSTEM")
    logger.info("="*80)
    
    # Initialize trainer
    trainer = EnhancedHierarchicalRLTrainer(config)
    
    # Test enhanced data generation
    logger.info("Testing Enhanced Data Generation")
    test_samples = trainer.data_generator.generate_balanced_batch(8)
    
    for i, sample in enumerate(test_samples, 1):
        logger.info(f"Sample {i} [{sample['noise_type_name']}]:")
        logger.info(f"  Original: {sample['clean_text'][:70]}...")
        logger.info(f"  Noisy:    {sample['text'][:70]}...")
        logger.info(f"  Complexity: {sample['complexity_score']:.2f}")
    
    # Training loop with enhanced monitoring
    logger.info("Starting Enhanced Training Loop")
    
    history = {
        'rewards': [],
        'detection_acc': [],
        'classification_acc': [],
        'epsilon': [],
        'learning_rates': []
    }
    
    # Early stopping
    patience = 15
    best_val_acc = 0
    
    for episode in range(config.num_episodes):
        # Train episode
        metrics = trainer.train_episode(balanced_batch=True)
        
        # Store history
        history['rewards'].append(metrics['avg_reward'])
        history['detection_acc'].append(metrics['detection_accuracy'])
        history['classification_acc'].append(metrics['classification_accuracy'])
        history['epsilon'].append(config.epsilon)
        history['learning_rates'].append(trainer.detector_optimizer.param_groups[0]['lr'])
        
        # Progress reporting
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}/{config.num_episodes}")
            logger.info(f"  Reward: {metrics['avg_reward']:.3f}")
            logger.info(f"  Detection: {metrics['detection_accuracy']:.2%}")
            logger.info(f"  Classification: {metrics['classification_accuracy']:.2%}")
            logger.info(f"  Uncertainty: {metrics['avg_uncertainty']:.3f}")
            logger.info(f"  Epsilon: {config.epsilon:.4f}")
        
        # Early stopping check
        if metrics['detection_accuracy'] > best_val_acc:
            best_val_acc = metrics['detection_accuracy']
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1
        
        if trainer.patience_counter >= patience:
            logger.info(f"Early stopping at episode {episode + 1}")
            break
    
    # Generate enhanced visualizations
    logger.info("Generating Enhanced Visualizations")
    
    # Training plots
    training_plot_file = trainer.visualizer.plot_training_metrics(history)
    logger.info(f"Training plots saved: {training_plot_file}")
    
    # Example inference
    logger.info("Example Enhanced Inference")
    test_sample = trainer.data_generator.generate_sample('Mixed')
    embeddings = trainer.get_embeddings([test_sample['text']])
    
    # Detection
    detection_score = trainer.detector(embeddings).item()
    predicted_noisy = detection_score > 0.5
    
    # Classification with uncertainty
    if predicted_noisy:
        noise_probs, total_unc, epi_unc, ale_unc = trainer.classifier.predict_with_uncertainty(embeddings)
        predicted_type = noise_probs.argmax(dim=-1).item()
        confidence = noise_probs.max().item()
        
        logger.info(f"Input Text: {test_sample['text']}")
        logger.info(f"Detection: {detection_score:.3f} (Noisy: {predicted_noisy})")
        logger.info(f"Noise Type: {trainer.data_generator.noise_types[predicted_type]} ({confidence:.2%})")
        logger.info(f"Uncertainty: Total={total_unc:.3f}, Epistemic={epi_unc:.3f}, Aleatoric={ale_unc:.3f}")
    
    # Comprehensive evaluation
    logger.info("Running Comprehensive Evaluation")
    eval_results = trainer.evaluate_model()
    
    # Evaluation plots
    eval_plot_file = trainer.visualizer.plot_evaluation_results(eval_results)
    logger.info(f"Evaluation plots saved: {eval_plot_file}")
    
    # Advanced analysis
    advanced_plot_file = trainer.visualizer.plot_advanced_metrics(eval_results)
    logger.info(f"Advanced analysis saved: {advanced_plot_file}")
    
    # Final summary
    logger.info("="*80)
    logger.info("TRAINING COMPLETE - FINAL SUMMARY")
    logger.info("="*80)
    
    # Calculate final metrics
    final_episodes = min(20, len(history['rewards']))
    avg_reward = np.mean(history['rewards'][-final_episodes:])
    avg_detection = np.mean(history['detection_acc'][-final_episodes:])
    avg_classification = np.mean(history['classification_acc'][-final_episodes:])
    
    logger.info(f"Final Performance (Last {final_episodes} episodes):")
    logger.info(f"  Average Reward: {avg_reward:.3f}")
    logger.info(f"  Detection Accuracy: {avg_detection:.2%}")
    logger.info(f"  Classification Accuracy: {avg_classification:.2%}")
    
    logger.info(f"\\nEvaluation Results:")
    logger.info(f"  Total Samples: {eval_results['total_samples']}")
    logger.info(f"  Detection Accuracy: {eval_results['correct_detection']/eval_results['total_samples']:.2%}")
    logger.info(f"  Overall F1 Score: {eval_results['f1_score']:.3f}")
    
    # Save metrics
    metrics_file = trainer.metrics_logger.save_to_json()
    logger.info(f"Metrics saved: {metrics_file}")
    
    logger.info("="*80)
    logger.info("Enhanced system training completed successfully!")
    logger.info("="*80)

if __name__ == "__main__":
    main()
