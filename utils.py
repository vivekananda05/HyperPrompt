# utils/metrics.py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
# import spectral
import numpy as np
from sklearn.metrics import confusion_matrix



import torch

def compute_imf_weights_torch(
    ground_truth: torch.Tensor,
    n_classes: int = None
):
    """
    Compute Inverse Median Frequency (IMF) weights using torch.

    Args:
        ground_truth: torch.Tensor of labels (any shape)
        n_classes: total number of classes

    Returns:
        weights: torch.Tensor [n_classes]
    """

    # Flatten labels
    labels = ground_truth.view(-1)

    if n_classes is None:
        n_classes = int(labels.max().item()) + 1

    device = labels.device

    # Count pixels per class
    class_counts = torch.bincount(
        labels,
        minlength=n_classes
    ).float()

    # Compute frequencies
    total_pixels = class_counts.sum()

    if total_pixels == 0:
        raise ValueError("No pixels found for IMF computation.")

    frequencies = class_counts / total_pixels

    # Get non-zero frequencies
    nonzero_mask = frequencies > 0

    if nonzero_mask.sum() == 0:
        raise ValueError("All class frequencies are zero.")

    median_freq = frequencies[nonzero_mask].median()

    # Compute IMF weights
    weights = torch.zeros(n_classes, device=device)
    weights[nonzero_mask] = median_freq / frequencies[nonzero_mask]

    return weights

def compute_metrics(prediction, target, ignored_labels=None, n_classes=None):
    """
    HSI paper-style metrics
    Background (label=0) completely excluded
    """

    if ignored_labels is None:
        ignored_labels = []

    # -------------------------------------------------
    # Remove background pixels completely
    # -------------------------------------------------
    mask = np.ones(target.shape, dtype=bool)

    for l in ignored_labels:
        mask[target == l] = False

    target = target[mask]
    prediction = prediction[mask]

    # -------------------------------------------------
    # Re-index labels (1..C → 0..C-1)
    # -------------------------------------------------
    unique_labels = np.unique(target)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    target = np.array([label_map[t] for t in target])
    prediction = np.array([label_map[p] for p in prediction])

    n_classes = len(unique_labels)

    # -------------------------------------------------
    # Confusion Matrix (without background)
    # -------------------------------------------------
    cm = confusion_matrix(target, prediction, labels=range(n_classes))

    total = np.sum(cm)

    # -------------------------------------------------
    # Overall Accuracy (OA)
    # -------------------------------------------------
    OA = np.trace(cm) / total

    # -------------------------------------------------
    # Per-class Accuracy
    # -------------------------------------------------
    class_acc = np.zeros(n_classes)
    for i in range(n_classes):
        if np.sum(cm[i]) == 0:
            class_acc[i] = 0
        else:
            class_acc[i] = cm[i][i] / np.sum(cm[i])

    # -------------------------------------------------
    # Average Accuracy (AA)
    # -------------------------------------------------
    AA = np.mean(class_acc)

    # -------------------------------------------------
    # F1 Scores
    # -------------------------------------------------
    F1_scores = np.zeros(n_classes)
    for i in range(n_classes):
        denom = np.sum(cm[i, :]) + np.sum(cm[:, i])
        if denom == 0:
            F1_scores[i] = 0.
        else:
            F1_scores[i] = 2. * cm[i, i] / denom

    # -------------------------------------------------
    # Kappa (without background)
    # -------------------------------------------------
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)

    return {
        "Confusion_matrix": cm,
        "OA": OA,
        "AA": AA,
        "Kappa": kappa,
        "class_acc": class_acc,
        "F1_scores": F1_scores
    }



def convert_to_color(arr_2d, palette):
    h, w = arr_2d.shape
    arr_3d = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in palette.items():
        arr_3d[arr_2d == label] = color

    return arr_3d


# def create_rgb_composite(img, bands):
#     rgb = spectral.get_rgb(img, bands)
#     rgb /= np.max(rgb)
#     rgb = np.asarray(255 * rgb, dtype='uint8')
#     return rgb

def create_rgb_composite(img, bands):
    """
    RGB composite without spectral dependency
    """
    r, g, b = bands

    rgb = np.stack([
        img[:, :, r],
        img[:, :, g],
        img[:, :, b]
    ], axis=-1)

    # Normalize per channel
    rgb = rgb.astype(np.float32)
    rgb -= rgb.min()
    rgb /= (rgb.max())
    rgb = (rgb * 255).astype(np.uint8)

    return rgb


def create_false_color_composite(img, bands):
    """
    False-color composite generated from selected bands.
    Reorders channels from (0, 1, 2) to (2, 0, 1) and keeps HWC layout.
    """
    rgb = create_rgb_composite(img, bands)
    false_color = rgb[:, :, [2, 0, 1]]
    return false_color

def visualize_dataset_and_results(
    img,
    gt,
    pred,
    bands,
    palette,
    class_names,
    save_path=None
):
    """
    Paper-style visualization with class legend as a subplot on the right side
    """

    if palette is None:
        raise ValueError("Palette is required for visualization. Provide RGB_BANDS and PALETTE in config.")

    if bands is None:
        raise ValueError("RGB bands are required for visualization. Provide RGB_BANDS in config.")

    false_color = create_false_color_composite(img, bands)
    gt_color = convert_to_color(gt, palette)
    pred_color = convert_to_color(pred, palette)

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.35], wspace=0.3)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_legend = fig.add_subplot(gs[0, 3])

    # False color
    axes[0].imshow(false_color)
    axes[0].set_title(f"False Color (bands {bands[0]}, {bands[1]}, {bands[2]})", fontsize=12)
    axes[0].axis("off")

    # GT
    axes[1].imshow(gt_color)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(pred_color)
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    # -------------------------------
    # Create Legend as Subplot (Exclude Background)
    # Display vertically on the right side
    # -------------------------------
    ax_legend.axis("off")
    ax_legend.set_title("Class Legend", fontsize=12, fontweight="bold")
    
    # Create legend elements
    legend_elements = []
    for i in range(1, len(class_names) + 1):
        color = np.array(palette[i]) / 255.0
        legend_elements.append(
            Patch(facecolor=color, edgecolor='black', label=class_names[i-1])
        )

    # Display legend as items in the legend subplot (top to bottom)
    ax_legend.legend(
        handles=legend_elements,
        loc="upper left",
        ncol=1,  # Display in single column (top to bottom)
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_tsne_features(
    features,
    labels,
    class_names,
    palette=None,
    save_path=None,
    title="t-SNE Feature Visualization",
    perplexity=30,
    n_iter=1000
):
    """
    Plot t-SNE visualization of feature embeddings.
    
    Args:
        features: Feature array (N, D) - N samples, D features
        labels: Label array (N,) - class labels (1-indexed)
        class_names: List of class names
        palette: Dict of color palette for classes
        save_path: Path to save the figure
        title: Title of the plot
        perplexity: t-SNE perplexity parameter
        n_iter: t-SNE iterations
    """
    from sklearn.manifold import TSNE
    
    # Ensure features are 2D
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    print("Computing t-SNE (this may take a while)...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42,
        n_jobs=-1
    )
    features_2d = tsne.fit_transform(features)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot points for each class
    for class_idx in range(1, len(class_names) + 1):
        mask = labels == class_idx
        if np.any(mask):
            # Get color from palette if available
            if palette is not None and class_idx in palette:
                color = np.array(palette[class_idx]) / 255.0
            else:
                color = None
            
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[color] if color is not None else None,
                label=class_names[class_idx - 1],
                alpha=0.6,
                s=30,
                edgecolors='k',
                linewidth=0.5
            )
    
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ t-SNE plot saved to: {save_path}")
    
    plt.show()
    
    return features_2d


def plot_branch_agreement_map(
    pred_clip,
    pred_sam,
    pred_fused,
    gt,
    palette,
    save_path=None,
    hsi_shape=None
):
    """
    Visualize branch agreement map.
    - Green: Both branches correct
    - Red: Both branches wrong
    - Orange: Branches disagree (different predictions)
    
    Args:
        pred_clip: CLIP branch predictions (1-indexed)
        pred_sam: SAM branch predictions (1-indexed)
        pred_fused: Fused predictions (1-indexed)
        gt: Ground truth labels (1-indexed)
        palette: Color palette dict
        save_path: Path to save figure
        hsi_shape: Original HSI shape (H, W, C) for reshaping if needed
    
    Returns:
        agreement_map: RGB agreement map
    """
    # Ensure flat arrays
    pred_clip = pred_clip.flatten()
    pred_sam = pred_sam.flatten()
    gt = gt.flatten()
    
    # Create agreement map
    agreement_map = np.zeros((len(gt), 3), dtype=np.uint8)
    
    for i in range(len(gt)):
        clip_correct = (pred_clip[i] == gt[i])
        sam_correct = (pred_sam[i] == gt[i])
        
        if clip_correct and sam_correct:
            # Both correct - Green
            agreement_map[i] = [0, 255, 0]
        elif not clip_correct and not sam_correct:
            # Both wrong - Red
            agreement_map[i] = [255, 0, 0]
        else:
            # Disagreement - Orange
            agreement_map[i] = [255, 165, 0]
    
    # Reshape to 2D if original shape is provided
    if hsi_shape is not None:
        H, W, _ = hsi_shape
        agreement_map = agreement_map.reshape(H, W, 3)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Agreement map
    if len(agreement_map.shape) == 3 and agreement_map.shape[2] == 3:
        axes[0].imshow(agreement_map)
    else:
        axes[0].imshow(agreement_map.reshape(-1))
    axes[0].set_title("Branch Agreement Map", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Both Correct'),
        Patch(facecolor='red', edgecolor='black', label='Both Wrong'),
        Patch(facecolor='orange', edgecolor='black', label='Disagreement')
    ]
    axes[1].axis("off")
    axes[1].legend(handles=legend_elements, loc="center", fontsize=12, frameon=True)
    axes[1].set_title("Legend", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Branch agreement map saved to: {save_path}")
    
    plt.show()
    
    return agreement_map


def plot_branch_confidence_maps(
    conf_clip,
    conf_sam,
    conf_fused,
    hsi_shape=None,
    save_path=None
):
    """
    Visualize confidence heatmaps from both branches and fusion.
    
    Args:
        conf_clip: CLIP branch confidence scores (N,) or (H, W)
        conf_sam: SAM branch confidence scores (N,) or (H, W)
        conf_fused: Fused confidence scores (N,) or (H, W)
        hsi_shape: Original HSI shape (H, W, C) for reshaping
        save_path: Path to save figure
    
    Returns:
        conf_maps: Dict with heatmaps
    """
    # Reshape if needed
    if hsi_shape is not None:
        H, W, _ = hsi_shape
        conf_clip = conf_clip.flatten().reshape(H, W) if len(conf_clip.shape) == 1 else conf_clip
        conf_sam = conf_sam.flatten().reshape(H, W) if len(conf_sam.shape) == 1 else conf_sam
        conf_fused = conf_fused.flatten().reshape(H, W) if len(conf_fused.shape) == 1 else conf_fused
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # CLIP confidence
    im1 = axes[0, 0].imshow(conf_clip, cmap='hot')
    axes[0, 0].set_title("CLIP Branch Confidence", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], label="Confidence")
    
    # SAM confidence
    im2 = axes[0, 1].imshow(conf_sam, cmap='hot')
    axes[0, 1].set_title("SAM Branch Confidence", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], label="Confidence")
    
    # Fused confidence
    im3 = axes[1, 0].imshow(conf_fused, cmap='hot')
    axes[1, 0].set_title("Fused Confidence", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], label="Confidence")
    
    # Confidence difference (CLIP - SAM)
    conf_diff = conf_clip - conf_sam
    im4 = axes[1, 1].imshow(conf_diff, cmap='RdBu_r')
    axes[1, 1].set_title("Confidence Difference (CLIP - SAM)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], label="Difference")
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Branch confidence maps saved to: {save_path}")
    
    plt.show()
    
    return {
        'clip': conf_clip,
        'sam': conf_sam,
        'fused': conf_fused,
        'difference': conf_diff
    }


def plot_fusion_quality_map(
    pred_clip,
    pred_sam,
    pred_fused,
    gt,
    hsi_shape=None,
    save_path=None
):
    """
    Visualize fusion quality map showing where fusion improves/degrades predictions.
    - Green: Fusion improved (was wrong, now correct)
    - Red: Fusion degraded (was correct, now wrong)
    - Orange: Individual branches agree but fusion disagrees
    - Gray: No change
    
    Args:
        pred_clip: CLIP branch predictions (1-indexed)
        pred_sam: SAM branch predictions (1-indexed)
        pred_fused: Fused predictions (1-indexed)
        gt: Ground truth labels (1-indexed)
        hsi_shape: Original HSI shape (H, W, C) for reshaping
        save_path: Path to save figure
    
    Returns:
        fusion_map: RGB fusion quality map
    """
    # Ensure flat arrays
    pred_clip = pred_clip.flatten()
    pred_sam = pred_sam.flatten()
    pred_fused = pred_fused.flatten()
    gt = gt.flatten()
    
    # Calculate correctness
    clip_correct = (pred_clip == gt)
    sam_correct = (pred_sam == gt)
    fused_correct = (pred_fused == gt)
    individual_max_correct = np.logical_or(clip_correct, sam_correct)
    
    # Create fusion quality map
    fusion_map = np.zeros((len(gt), 3), dtype=np.uint8)
    
    for i in range(len(gt)):
        if fused_correct[i] and not individual_max_correct[i]:
            # Fusion improved - Green
            fusion_map[i] = [0, 255, 0]
        elif not fused_correct[i] and individual_max_correct[i]:
            # Fusion degraded - Red
            fusion_map[i] = [255, 0, 0]
        elif fused_correct[i] and individual_max_correct[i]:
            # Both fusion and at least one branch correct - Light Green
            fusion_map[i] = [144, 238, 144]
        elif not fused_correct[i] and not individual_max_correct[i]:
            # Both fusion and all branches wrong - Light Gray
            fusion_map[i] = [200, 200, 200]
        else:
            # Branches agree but fusion disagrees - Orange
            fusion_map[i] = [255, 165, 0]
    
    # Reshape to 2D if original shape is provided
    if hsi_shape is not None:
        H, W, _ = hsi_shape
        fusion_map = fusion_map.reshape(H, W, 3)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Fusion quality map
    if len(fusion_map.shape) == 3 and fusion_map.shape[2] == 3:
        axes[0].imshow(fusion_map)
    else:
        axes[0].imshow(fusion_map.reshape(-1))
    axes[0].set_title("Fusion Quality Map", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Fusion Improved'),
        Patch(facecolor='red', edgecolor='black', label='Fusion Degraded'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Consistent Correct'),
        Patch(facecolor='orange', edgecolor='black', label='Branch Disagreement'),
        Patch(facecolor='lightgray', edgecolor='black', label='Consistent Wrong')
    ]
    axes[1].axis("off")
    axes[1].legend(handles=legend_elements, loc="center", fontsize=11, frameon=True)
    axes[1].set_title("Legend", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Fusion quality map saved to: {save_path}")
    
    plt.show()
    
    return fusion_map


def plot_tcdm_layer_activation_changes(
    layer_activation_data,
    save_path=None
):
    """
    Visualize layer-wise activation changes from TCDM modulation.
    
    Args:
        layer_activation_data: Dict with keys:
            - 'layer_names': List of layer names/indices
            - 'before_activation_mean': [n_layers] mean activation before TCDM
            - 'after_activation_mean': [n_layers] mean activation after TCDM
            - 'before_activation_std': [n_layers] std activation before TCDM
            - 'after_activation_std': [n_layers] std activation after TCDM
            - 'modulation_factor': [n_layers] magnitude of gamma * feat changes
            - 'modulation_bias': [n_layers] magnitude of beta shifts
        save_path: Path to save figure
    
    Returns:
        fig: matplotlib figure object
    """
    layer_names = layer_activation_data['layer_names']
    n_layers = len(layer_names)
    
    before_mean = layer_activation_data['before_activation_mean']
    after_mean = layer_activation_data['after_activation_mean']
    before_std = layer_activation_data['before_activation_std']
    after_std = layer_activation_data['after_activation_std']
    modulation_factor = layer_activation_data['modulation_factor']
    modulation_bias = layer_activation_data['modulation_bias']
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    x = np.arange(n_layers)
    width = 0.35
    
    # =====================================================================
    # 1. Mean Activation Before/After TCDM
    # =====================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x - width/2, before_mean, width, label='Before TCDM', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, after_mean, width, label='After TCDM', color='coral', alpha=0.8)
    ax1.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Activation', fontsize=11, fontweight='bold')
    ax1.set_title('Mean Activation: Before vs After TCDM', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # =====================================================================
    # 2. Activation Std Before/After TCDM
    # =====================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    bars3 = ax2.bar(x - width/2, before_std, width, label='Before TCDM', color='lightblue', alpha=0.8)
    bars4 = ax2.bar(x + width/2, after_std, width, label='After TCDM', color='lightsalmon', alpha=0.8)
    ax2.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Activation Std Dev', fontsize=11, fontweight='bold')
    ax2.set_title('Activation Std: Before vs After TCDM', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_names)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # =====================================================================
    # 3. Modulation Factor (Gamma Contribution)
    # =====================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    colors_gamma = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_layers))
    bars5 = ax3.bar(x, modulation_factor, color=colors_gamma, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Modulation Factor (γ)', fontsize=11, fontweight='bold')
    ax3.set_title('TCDM Scale Modulation (γ) per Layer', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(layer_names)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars5):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # =====================================================================
    # 4. Modulation Bias (Beta Contribution)
    # =====================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    colors_beta = plt.cm.viridis(np.linspace(0.2, 0.8, n_layers))
    bars6 = ax4.bar(x, modulation_bias, color=colors_beta, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Modulation Bias (β)', fontsize=11, fontweight='bold')
    ax4.set_title('TCDM Shift Modulation (β) per Layer', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(layer_names)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars6):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # =====================================================================
    # 5. Relative Change in Mean Activation
    # =====================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    relative_change = ((after_mean - before_mean) / (np.abs(before_mean) + 1e-8)) * 100
    colors_change = ['green' if x > 0 else 'red' for x in relative_change]
    bars7 = ax5.bar(x, relative_change, color=colors_change, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax5.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Relative Change (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Relative Change in Mean Activation After TCDM', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(layer_names)
    ax5.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars7):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')
    
    # =====================================================================
    # 6. Summary Statistics Table
    # =====================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for i in range(n_layers):
        summary_data.append([
            f'Layer {layer_names[i]}',
            f'{before_mean[i]:.4f}',
            f'{after_mean[i]:.4f}',
            f'{modulation_factor[i]:.4f}',
            f'{modulation_bias[i]:.4f}'
        ])
    
    table = ax6.table(
        cellText=summary_data,
        colLabels=['Layer', 'μ(before)', 'μ(after)', 'γ', 'β'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('TCDM Layer-wise Activation Changes Analysis', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ TCDM layer activation changes saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_lora_weight_matrices(
    lora_data_dict,
    save_path=None
):
    """
    Visualize LoRA weight matrices (A and B) across different layers/modules.
    
    Args:
        lora_data_dict: Dict with LoRA module information:
            - 'modules': List of module names with LoRA
            - 'lora_A_shapes': List of LoRA_A matrix shapes
            - 'lora_B_shapes': List of LoRA_B matrix shapes
            - 'lora_A_norms': List of Frobenius norms of LoRA_A
            - 'lora_B_norms': List of Frobenius norms of LoRA_B
            - 'lora_A_svd': List of (rank,) singular values for each module
            - 'lora_B_svd': List of (rank,) singular values for each module
            - 'lora_rank': LoRA rank value
            - 'alpha': LoRA alpha scaling factor
        save_path: Path to save figure
    
    Returns:
        fig: matplotlib figure object
    """
    modules = lora_data_dict['modules']
    lora_A_norms = lora_data_dict['lora_A_norms']
    lora_B_norms = lora_data_dict['lora_B_norms']
    lora_A_svd = lora_data_dict['lora_A_svd']
    lora_B_svd = lora_data_dict['lora_B_svd']
    lora_rank = lora_data_dict.get('lora_rank', None)
    alpha = lora_data_dict.get('alpha', None)
    
    n_modules = len(modules)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    x = np.arange(n_modules)
    width = 0.35
    
    # =====================================================================
    # 1. LoRA_A and LoRA_B Frobenius Norms
    # =====================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x - width/2, lora_A_norms, width, label='LoRA_A Norm', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, lora_B_norms, width, label='LoRA_B Norm', color='coral', alpha=0.8)
    ax1.set_xlabel('Module Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frobenius Norm', fontsize=11, fontweight='bold')
    ax1.set_title('LoRA Matrix Frobenius Norms', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'M{i}' for i in range(n_modules)])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # =====================================================================
    # 2. Singular Value Distribution - LoRA_A
    # =====================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    for i, svs in enumerate(lora_A_svd):
        ax2.plot(range(len(svs)), svs, marker='o', label=f'M{i}', linewidth=2, markersize=5)
    ax2.set_xlabel('Singular Value Index', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Singular Value', fontsize=11, fontweight='bold')
    ax2.set_title('LoRA_A Singular Values', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # =====================================================================
    # 3. Singular Value Distribution - LoRA_B
    # =====================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    for i, svs in enumerate(lora_B_svd):
        ax3.plot(range(len(svs)), svs, marker='s', label=f'M{i}', linewidth=2, markersize=5)
    ax3.set_xlabel('Singular Value Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Singular Value', fontsize=11, fontweight='bold')
    ax3.set_title('LoRA_B Singular Values', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # =====================================================================
    # 4. Heatmap of LoRA_A Norms (Module-wise)
    # =====================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    heatmap_A = np.array(lora_A_norms).reshape(-1, 1)
    im4 = ax4.imshow(heatmap_A, cmap='YlOrRd', aspect='auto')
    ax4.set_yticks(range(n_modules))
    ax4.set_yticklabels([f'M{i}' for i in range(n_modules)])
    ax4.set_xticks([0])
    ax4.set_xticklabels(['LoRA_A'])
    ax4.set_title('LoRA_A Norms Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Norm Value')
    
    # Add text annotations
    for i in range(n_modules):
        ax4.text(0, i, f'{lora_A_norms[i]:.3f}', ha='center', va='center', 
                color='white' if lora_A_norms[i] > max(lora_A_norms) * 0.5 else 'black', fontweight='bold')
    
    # =====================================================================
    # 5. Heatmap of LoRA_B Norms (Module-wise)
    # =====================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    heatmap_B = np.array(lora_B_norms).reshape(-1, 1)
    im5 = ax5.imshow(heatmap_B, cmap='Blues', aspect='auto')
    ax5.set_yticks(range(n_modules))
    ax5.set_yticklabels([f'M{i}' for i in range(n_modules)])
    ax5.set_xticks([0])
    ax5.set_xticklabels(['LoRA_B'])
    ax5.set_title('LoRA_B Norms Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=ax5, label='Norm Value')
    
    # Add text annotations
    for i in range(n_modules):
        ax5.text(0, i, f'{lora_B_norms[i]:.3f}', ha='center', va='center',
                color='white' if lora_B_norms[i] > max(lora_B_norms) * 0.5 else 'black', fontweight='bold')
    
    # =====================================================================
    # 6. Effective Rank Estimation (via SVD)
    # =====================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    effective_ranks_A = []
    effective_ranks_B = []
    
    for i in range(n_modules):
        svs_A = lora_A_svd[i]
        svs_B = lora_B_svd[i]
        
        # Effective rank = sum(sv^2) / max(sv)^2 (normalized rank)
        rank_A = np.sum(svs_A) / (np.max(svs_A) + 1e-8) if len(svs_A) > 0 else 1
        rank_B = np.sum(svs_B) / (np.max(svs_B) + 1e-8) if len(svs_B) > 0 else 1
        
        effective_ranks_A.append(rank_A)
        effective_ranks_B.append(rank_B)
    
    bars5 = ax6.bar(x - width/2, effective_ranks_A, width, label='LoRA_A', color='seagreen', alpha=0.8)
    bars6 = ax6.bar(x + width/2, effective_ranks_B, width, label='LoRA_B', color='mediumpurple', alpha=0.8)
    ax6.set_xlabel('Module Index', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Effective Rank', fontsize=11, fontweight='bold')
    ax6.set_title('Effective Rank per Module', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'M{i}' for i in range(n_modules)])
    ax6.legend(fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    
    # =====================================================================
    # 7. LoRA Contribution Analysis
    # =====================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    contribution = np.array(lora_A_norms) * np.array(lora_B_norms)
    contribution = contribution / (np.max(contribution) + 1e-8)  # Normalize
    colors_contrib = plt.cm.viridis(contribution)
    bars7 = ax7.bar(x, contribution, color=colors_contrib, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax7.set_xlabel('Module Index', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Contribution (Normalized)', fontsize=11, fontweight='bold')
    ax7.set_title('LoRA Contribution Impact (A_norm × B_norm)', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels([f'M{i}' for i in range(n_modules)])
    ax7.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars7, contribution):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # =====================================================================
    # 8. Summary Statistics Table
    # =====================================================================
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    summary_data = []
    for i in range(n_modules):
        summary_data.append([
            f'M{i}',
            f'{lora_A_norms[i]:.4f}',
            f'{lora_B_norms[i]:.4f}',
            f'{contribution[i]:.4f}',
            f'{len(lora_A_svd[i])}',
            f'{effective_ranks_A[i]:.2f}',
            f'{effective_ranks_B[i]:.2f}'
        ])
    
    # Add summary row
    summary_data.append([
        'TOTAL',
        f'{sum(lora_A_norms):.4f}',
        f'{sum(lora_B_norms):.4f}',
        f'{sum(contribution):.4f}',
        f'{lora_rank if lora_rank else "—"}',
        f'{np.mean(effective_ranks_A):.2f}',
        f'{np.mean(effective_ranks_B):.2f}'
    ])
    
    table = ax8.table(
        cellText=summary_data,
        colLabels=['Module', '||A||', '||B||', 'Contribution', 'Rank', 'E.Rank(A)', 'E.Rank(B)'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors and highlight total row
    for i in range(1, len(summary_data) + 1):
        for j in range(7):
            if i == len(summary_data):  # Total row
                table[(i, j)].set_facecolor('#FFA726')
                table[(i, j)].set_text_props(weight='bold')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    ax8.set_title('LoRA Weight Statistics Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Add hyperparameters text
    hp_text = f'LoRA Rank: {lora_rank}, Alpha: {alpha}'
    fig.text(0.5, 0.01, hp_text, ha='center', fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('LoRA Weight Matrices Analysis', fontsize=14, fontweight='bold', y=0.995)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ LoRA weight matrices visualization saved to: {save_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# Model Output Extraction Utilities
# ============================================================================
# HyperPrompt models have different output structures. Use these utilities to
# extract branch logits consistently across all model types.
# ============================================================================

def extract_branch_logits_from_model_output(model_outputs: tuple, model_name: str):
    """
    Extract per-branch logits from model output.
    
    All HyperPrompt models return per-branch logits in their outputs
    along with corresponding image features (img_feat1, img_feat2).
    
    Args:
        model_outputs: Tuple returned by model.forward()
        model_name: Name of the HyperPrompt model
    
    Returns:
        Tuple of (logits_branch1, logits_branch2, logits_fused)
        
    Raises:
        ValueError: If model type is not recognized or output structure is invalid
    """
    model_name_lower = model_name.lower()
    
    if len(model_outputs) < 7:
        raise ValueError(f"Model output is too short ({len(model_outputs)} elements). HyperPrompt models should return at least 7 elements.")
    
    # =====================================================================
    # 9-element models: logits1 at [0], logits2 at [3], fused at [6]
    # Structure: (logits1, img_feat1, text_feat1, logits2, img_feat2, text_feat2, logits_fused, recon_rgb, rgb_sam)
    # =====================================================================
    if model_name_lower in [
        'hyperprompt_promptsrc', 'hyperprompt_promptkd',
        'hyperprompt_coop',
        'hyperprompt_maple', 'hyperprompt_mmrl',
    ]:
        logits_branch1 = model_outputs[0]
        logits_branch2 = model_outputs[3]
        logits_fused = model_outputs[6]
        return (logits_branch1, logits_branch2, logits_fused)

    # =====================================================================
    # 11-element models: logits1 at [0], logits2 at [4], fused at [8]
    # KgCoOp: (logits1, img_feat1, text_feat1, kloss1, logits2, img_feat2, text_feat2, kloss2, logits_fused, recon_rgb, rgb_sam)
    # =====================================================================
    elif model_name_lower in [
        'hyperprompt_kgcoop',
    ]:
        if len(model_outputs) < 9:
            raise ValueError(f"11-element model {model_name} has insufficient output elements ({len(model_outputs)}).")
        logits_branch1 = model_outputs[0]
        logits_branch2 = model_outputs[4]
        logits_fused = model_outputs[8]
        return (logits_branch1, logits_branch2, logits_fused)
    
    else:
        raise ValueError(f"Unknown model type: {model_name}. Cannot extract per-branch logits.")


def extract_fused_logits_from_model_output(model_outputs: tuple, model_name: str):
    """
    Extract only the fused logits from model output.
    This works for all supported HyperPrompt models.
    
    Args:
        model_outputs: Tuple returned by model.forward()
        model_name: Name of the model
    
    Returns:
        Tensor of fused logits [B, C]
        
    Raises:
        ValueError: If model type is not recognized or output structure is invalid
    """
    model_name_lower = model_name.lower()
    
    # 9-element models: fused logits at index 6
    if model_name_lower in [
        'hyperprompt_coop', 'hyperprompt_maple',
        'hyperprompt_promptsrc', 'hyperprompt_promptkd',
        'hyperprompt_mmrl'
    ]:
        if len(model_outputs) >= 7:
            return model_outputs[6]

    # 11-element models: fused logits at index 8
    elif model_name_lower in ['hyperprompt_kgcoop']:
        if len(model_outputs) >= 9:
            return model_outputs[8]
    
    else:
        raise ValueError(f"Unknown model type: {model_name}. Cannot extract fused logits.")
    
    raise ValueError(f"Model output too short. Expected sufficient elements for {model_name}.")


def get_model_output_info(model_name: str):
    """
    Get information about a model's output structure.
    
    All HyperPrompt models return per-branch logits, image features, and text features.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dict with:
            - 'output_length': Expected number of elements in tuple
            - 'has_branch_logits': Whether per-branch logits are returned (always True now)
            - 'has_img_features': Whether image features are returned (always True now)
            - 'fused_logits_index': Index of fused logits in tuple
            - 'branch_logits_indices': Tuple of (branch1_idx, branch2_idx)
            - 'img_feat_indices': Tuple of (img_feat1_idx, img_feat2_idx)
            - 'description': Text description of output structure
    """
    model_name_lower = model_name.lower()
    
    info_map = {
        # =====================================================================
        # 9-element models with per-branch logits AND image features
        # Structure: (logits1, img_feat1, text_feat1, logits2, img_feat2, text_feat2, logits_fused, recon_rgb, rgb_sam)
        # =====================================================================
        'hyperprompt_mmrl': {
            'output_length': 9,
            'has_branch_logits': True,
            'has_img_features': True,
            'fused_logits_index': 6,
            'branch_logits_indices': (0, 3),
            'img_feat_indices': (1, 4),
            'description': '(logits1, img_feat1, text_feat1, logits2, img_feat2, text_feat2, logits_fused, recon_rgb, rgb_sam)'
        },
        'hyperprompt_promptkd': {
            'output_length': 9,
            'has_branch_logits': True,
            'has_img_features': True,
            'fused_logits_index': 6,
            'branch_logits_indices': (0, 3),
            'img_feat_indices': (1, 4),
            'description': '(logits1, img_feat1, teacher_logits1, logits2, img_feat2, teacher_logits2, logits_fused, recon_rgb, rgb_sam)'
        },
        'hyperprompt_coop': {
            'output_length': 9,
            'has_branch_logits': True,
            'has_img_features': True,
            'fused_logits_index': 6,
            'branch_logits_indices': (0, 3),
            'img_feat_indices': (1, 4),
            'description': '(logits1, img_feat1, text_feat1, logits2, img_feat2, text_feat2, logits_fused, recon_rgb, rgb_sam)'
        },
        'hyperprompt_maple': {
            'output_length': 9,
            'has_branch_logits': True,
            'has_img_features': True,
            'fused_logits_index': 6,
            'branch_logits_indices': (0, 3),
            'img_feat_indices': (1, 4),
            'description': '(logits1, img_feat1, text_feat1, logits2, img_feat2, text_feat2, logits_fused, recon_rgb, rgb_sam)'
        },
        
        # =====================================================================
        # 11-element models: per-branch logits, image features, mu/sigma or text
        # =====================================================================
        'hyperprompt_kgcoop': {
            'output_length': 11,
            'has_branch_logits': True,
            'has_img_features': True,
            'fused_logits_index': 8,
            'branch_logits_indices': (0, 4),
            'img_feat_indices': (1, 5),
            'description': '(logits1, img_feat1, text_feat1, knowledge_loss1, logits2, img_feat2, text_feat2, knowledge_loss2, logits_fused, recon_rgb, rgb_sam)'
        },
    }
    
    if model_name_lower not in info_map:
        raise ValueError(f"Unknown model type: {model_name}")
    
    return info_map[model_name_lower]
