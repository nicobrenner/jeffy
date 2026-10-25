"""Class-conditional loss distribution classifier.

Uses the existing pretrained model to compute per-class loss statistics,
then classifies test images by finding which class distribution they're
most likely under (Mahalanobis distance).
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path

from train_rope_byte_2d import RoPE2DByteLM, MNISTByteDataset


def load_model(ckpt_path, bidirectional, device):
    model = RoPE2DByteLM(vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                         height=28, width=28, bidirectional=bidirectional).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


@torch.no_grad()
def compute_loss_patterns(model, loader, device, max_batches=None):
    """Compute shifted per-pixel loss for all images. Returns (N, 783), labels (N,)."""
    all_losses, all_labels = [], []
    for i, (imgs, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        imgs = imgs.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, 256),
            imgs[:, 1:].contiguous().view(-1).long(),
            reduction='none'
        ).reshape(imgs.shape[0], -1)
        all_losses.append(loss.cpu())
        all_labels.append(labels)
    return torch.cat(all_losses).numpy(), np.concatenate(all_labels)


def mahalanobis_classify(train_losses, train_labels, test_losses, use_shared_cov=False):
    """Classify via Mahalanobis distance to class-conditional Gaussians."""
    n_classes = 10
    dim = train_losses.shape[1]
    means = []
    inv_covs = []

    if use_shared_cov:
        shared_cov = np.cov(train_losses.T) + 1e-5 * np.eye(dim)
        shared_inv = np.linalg.inv(shared_cov)

    for c in range(n_classes):
        mask = train_labels == c
        class_data = train_losses[mask]
        means.append(class_data.mean(axis=0))
        if not use_shared_cov:
            cov = np.cov(class_data.T) + 1e-5 * np.eye(dim)
            inv_covs.append(np.linalg.inv(cov))

    preds = []
    for x in test_losses:
        dists = []
        for c in range(n_classes):
            diff = x - means[c]
            inv_c = shared_inv if use_shared_cov else inv_covs[c]
            d = diff @ inv_c @ diff
            dists.append(d)
        preds.append(np.argmin(dists))
    return np.array(preds)


def pca_mahalanobis_classify(train_losses, train_labels, test_losses, n_components=64):
    """PCA reduce then Mahalanobis."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_losses)
    test_s = scaler.transform(test_losses)

    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_s)
    test_pca = pca.transform(test_s)

    return mahalanobis_classify(train_pca, train_labels, test_pca, use_shared_cov=True)


def centroid_classify(train_losses, train_labels, test_losses, metric='euclidean'):
    """Classify by nearest class centroid."""
    means = []
    for c in range(10):
        means.append(train_losses[train_labels == c].mean(axis=0))
    means = np.array(means)

    if metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(test_losses, means)
        return dists.argmin(axis=1)
    elif metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(test_losses, means)
        return sims.argmax(axis=1)


def main():
    device = 'mps'
    batch_size = 128

    mnist_train = torchvision.datasets.MNIST(root='/tmp/mnist', train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root='/tmp/mnist', train=False, download=True)
    train_labels_all = np.array([l for _, l in mnist_train])
    test_labels_all = np.array([l for _, l in mnist_test])

    train_ds = MNISTByteDataset('pile_runs/custom_vocab/byte_mnist/train.bytes', train_labels_all, 784)
    test_ds = MNISTByteDataset('pile_runs/custom_vocab/byte_mnist/val.bytes', test_labels_all, 784)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)

    models = {}
    bidir_path = Path('pile_runs/rope_byte/mnist_2d_d64_bidirectional_tp1800/best_model.pt')
    causal_path = Path('pile_runs/rope_byte/mnist_2d_d64_causal_tp1800/best_model.pt')
    if bidir_path.exists():
        models['bidirectional'] = load_model(bidir_path, True, device)
    if causal_path.exists():
        models['causal'] = load_model(causal_path, False, device)

    for name, model in models.items():
        print(f'\n{"="*60}')
        print(f'  MODEL: {name.upper()}')
        print(f'{"="*60}')

        print('  Computing train loss patterns...')
        train_losses, train_labels = compute_loss_patterns(model, train_loader, device)
        print(f'  Train: {train_losses.shape}, mean loss/px: {train_losses.mean():.6f}')

        print('  Computing test loss patterns...')
        test_losses, test_labels = compute_loss_patterns(model, test_loader, device)
        print(f'  Test: {test_losses.shape}')

        # Method 1: Nearest centroid (euclidean)
        preds = centroid_classify(train_losses, train_labels, test_losses, 'euclidean')
        acc = (preds == test_labels).mean()
        print(f'\n  Nearest centroid (euclidean):  {acc*100:.1f}%')

        # Method 2: Nearest centroid (cosine)
        preds = centroid_classify(train_losses, train_labels, test_losses, 'cosine')
        acc = (preds == test_labels).mean()
        print(f'  Nearest centroid (cosine):     {acc*100:.1f}%')

        # Method 3: PCA + shared-cov Mahalanobis
        for nc in [32, 64, 128]:
            preds = pca_mahalanobis_classify(train_losses, train_labels, test_losses, n_components=nc)
            acc = (preds == test_labels).mean()
            print(f'  PCA({nc}) + Mahalanobis:        {acc*100:.1f}%')

        # Method 4: Per-class Mahalanobis on PCA features
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        tr_s = scaler.fit_transform(train_losses)
        te_s = scaler.transform(test_losses)
        pca = PCA(n_components=64, random_state=42)
        tr_pca = pca.fit_transform(tr_s)
        te_pca = pca.transform(te_s)
        preds = mahalanobis_classify(tr_pca, train_labels, te_pca, use_shared_cov=False)
        acc = (preds == test_labels).mean()
        print(f'  PCA(64) + per-class Mahalanobis: {acc*100:.1f}%')

        # Method 5: LDA (linear discriminant analysis) - optimal linear class separator
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis()
        lda.fit(tr_pca, train_labels)
        acc = lda.score(te_pca, test_labels)
        print(f'  PCA(64) + LDA:                 {acc*100:.1f}%')

        # Method 6: QDA (quadratic discriminant analysis) - per-class covariance
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        try:
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(tr_pca, train_labels)
            acc = qda.score(te_pca, test_labels)
            print(f'  PCA(64) + QDA:                 {acc*100:.1f}%')
        except Exception as e:
            print(f'  PCA(64) + QDA:                 failed ({e})')


if __name__ == '__main__':
    main()
