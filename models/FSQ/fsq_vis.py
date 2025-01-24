"""
FSQ Attention and Token Dynamics Visualization Suite
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

class FSQAttentionVisualizer:
    """Advanced visualization tools for FSQ attention patterns and token dynamics."""
    
    def __init__(self, model_dim: int, num_heads: int = 8):
        self.model_dim = model_dim
        self.num_heads = num_heads
        
    def visualize_attention_flow(
        self, 
        tokens: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> go.Figure:
        """
        Create an interactive attention flow visualization.
        
        Args:
            tokens: Shape [B, L, D]
            attention_weights: Shape [B, H, L, L]
        """
        # Average attention across heads
        avg_attention = attention_weights.mean(dim=1)[0]  # Take first batch
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = [f"Token_{i}" for i in range(tokens.shape[1])],
                color = "blue"
            ),
            link = dict(
                source = [i for i in range(tokens.shape[1]) for _ in range(tokens.shape[1])],
                target = [j for _ in range(tokens.shape[1]) for j in range(tokens.shape[1])],
                value = avg_attention.flatten().tolist()
            )
        )])
        
        fig.update_layout(title_text="FSQ Token Attention Flow", font_size=10)
        return fig
        
    def plot_token_embeddings(
        self, 
        tokens: torch.Tensor,
        method: str = 'tsne'
    ) -> plt.Figure:
        """
        Visualize token embeddings in 2D space.
        
        Args:
            tokens: Shape [B, L, D]
            method: 'tsne' or 'umap'
        """
        from sklearn.manifold import TSNE
        import umap
        
        # Flatten batch and sequence dimensions
        token_flat = tokens.reshape(-1, tokens.shape[-1])
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = umap.UMAP(random_state=42)
            
        embedded = reducer.fit_transform(token_flat.detach().cpu().numpy())
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], 
                           c=np.arange(len(embedded)), cmap='viridis',
                           alpha=0.6)
        plt.colorbar(scatter)
        ax.set_title(f'Token Embeddings ({method.upper()})')
        return fig
        
    def visualize_token_transitions(
        self,
        token_sequences: torch.Tensor,
        window_size: int = 5
    ) -> plt.Figure:
        """
        Visualize token transition patterns.
        
        Args:
            token_sequences: Shape [B, L]
            window_size: Size of transition window
        """
        # Create transition matrix
        n_tokens = token_sequences.max().item() + 1
        transitions = torch.zeros((n_tokens, n_tokens))
        
        for seq in token_sequences:
            for i in range(len(seq) - window_size):
                for j in range(window_size):
                    transitions[seq[i], seq[i+j]] += 1
                    
        # Normalize
        transitions = transitions / transitions.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(transitions.cpu().numpy(), ax=ax, cmap='viridis')
        ax.set_title('Token Transition Patterns')
        return fig
        
    def plot_attention_head_analysis(
        self,
        attention_weights: torch.Tensor,
        tokens: torch.Tensor
    ) -> plt.Figure:
        """
        Analyze and visualize attention head patterns.
        
        Args:
            attention_weights: Shape [B, H, L, L]
            tokens: Shape [B, L, D]
        """
        # Compute head importance scores
        head_importance = torch.norm(attention_weights, dim=(-2, -1)).mean(0)
        
        # Compute head specialization (entropy)
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=(-2, -1)).mean(0)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Head importance
        ax1.bar(range(self.num_heads), head_importance.cpu())
        ax1.set_title('Attention Head Importance')
        ax1.set_xlabel('Head')
        ax1.set_ylabel('Importance Score')
        
        # Head entropy
        ax2.bar(range(self.num_heads), entropy.cpu())
        ax2.set_title('Attention Head Entropy')
        ax2.set_xlabel('Head')
        ax2.set_ylabel('Entropy')
        
        plt.tight_layout()
        return fig
        
    def create_token_usage_dashboard(
        self,
        token_sequences: torch.Tensor,
        num_steps: int = 100
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Create a dashboard showing token usage patterns over time.
        
        Args:
            token_sequences: Shape [B, L]
            num_steps: Number of time steps to analyze
        """
        n_tokens = token_sequences.max().item() + 1
        usage_over_time = torch.zeros(num_steps, n_tokens)
        
        # Compute usage statistics
        for i in range(num_steps):
            idx = int(i * len(token_sequences) / num_steps)
            curr_tokens = token_sequences[:idx+1].flatten()
            usage = torch.bincount(curr_tokens, minlength=n_tokens)
            usage_over_time[i] = usage
            
        # Create usage evolution plot
        fig1, ax1 = plt.subplots(figsize=(15, 5))
        im = ax1.imshow(usage_over_time.T.cpu(), aspect='auto', cmap='viridis')
        ax1.set_title('Token Usage Evolution')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Token ID')
        plt.colorbar(im)
        
        # Create final usage distribution
        fig2, ax2 = plt.subplots(figsize=(15, 5))
        final_usage = usage_over_time[-1]
        ax2.bar(range(n_tokens), final_usage.cpu())
        ax2.set_title('Final Token Usage Distribution')
        ax2.set_xlabel('Token ID')
        ax2.set_ylabel('Usage Count')
        
        return fig1, fig2
        
    def analyze_semantic_clusters(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> plt.Figure:
        """
        Analyze semantic clustering of FSQ tokens.
        
        Args:
            tokens: Shape [B, L, D]
            labels: Optional ground truth labels
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Flatten tokens
        flat_tokens = tokens.reshape(-1, tokens.shape[-1]).cpu().numpy()
        
        # Perform clustering
        n_clusters = min(8, len(flat_tokens) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(flat_tokens)
        
        # Create visualization
        fig = plt.figure(figsize=(20, 5))
        
        # 1. Cluster sizes
        ax1 = plt.subplot(131)
        cluster_sizes = np.bincount(clusters)
        ax1.bar(range(n_clusters), cluster_sizes)
        ax1.set_title('Cluster Sizes')
        
        # 2. Intra-cluster distances
        ax2 = plt.subplot(132)
        distances = kmeans.transform(flat_tokens)
        intra_distances = distances[np.arange(len(clusters)), clusters]
        ax2.hist(intra_distances, bins=50)
        ax2.set_title('Intra-cluster Distances')
        
        # 3. Cluster relationships
        ax3 = plt.subplot(133)
        cluster_centers = kmeans.cluster_centers_
        relationships = np.corrcoef(cluster_centers)
        sns.heatmap(relationships, ax=ax3)
        ax3.set_title('Cluster Relationships')
        
        plt.tight_layout()
        return fig
        
    def visualize_temporal_dynamics(
        self,
        token_sequences: torch.Tensor,
        time_window: int = 50
    ) -> plt.Figure:
        """
        Visualize temporal dynamics of FSQ tokens.
        
        Args:
            token_sequences: Shape [B, L]
            time_window: Window size for temporal analysis
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Token frequency over time
        ax1 = plt.subplot(211)
        times = np.arange(len(token_sequences))
        for token in range(token_sequences.max().item() + 1):
            mask = (token_sequences == token).float()
            smoothed = torch.conv1d(
                mask.unsqueeze(1),
                torch.ones(1, 1, time_window) / time_window,
                padding=time_window//2
            ).squeeze()
            ax1.plot(times, smoothed.cpu(), alpha=0.5, label=f'Token {token}')
        ax1.set_title('Token Frequency Evolution')
        
        # 2. Token correlation matrix
        ax2 = plt.subplot(212)
        token_matrix = torch.zeros(
            (token_sequences.max().item() + 1, token_sequences.shape[1])
        )
        for i, seq in enumerate(token_sequences):
            token_matrix[seq, i] = 1
        correlation = torch.corrcoef(token_matrix)
        sns.heatmap(correlation.cpu(), ax=ax2)
        ax2.set_title('Token Temporal Correlation')
        
        plt.tight_layout()
        return fig
        
    def create_interactive_attention_map(
        self,
        attention_weights: torch.Tensor,
        tokens: torch.Tensor
    ) -> go.Figure:
        """
        Create interactive attention visualization using plotly.
        
        Args:
            attention_weights: Shape [B, H, L, L]
            tokens: Shape [B, L, D]
        """
        # Average attention across heads and batches
        avg_attention = attention_weights.mean(dim=(0,1)).cpu().numpy()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=avg_attention,
            text=[[f'{v:.2f}' for v in row] for row in avg_attention],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
        ))
        
        # Update layout
        fig.update_layout(
            title='Interactive Attention Pattern Analysis',
            xaxis_title='Target Token Position',
            yaxis_title='Source Token Position',
            width=800,
            height=800
        )
        
        return fig