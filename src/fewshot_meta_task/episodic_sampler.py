"""N-way K-shot episodic sampler over an ImageFolder-style dataset.

No learn2learn dependency: episode sampling is just "group indices by
class once, then draw shot+query distinct indices per class per episode
and split them" -- simple enough to not need the pinned old
torch1.12/learn2learn environment the original meta-tasks code used.
"""
import random
from collections import defaultdict

import torch


class EpisodicSampler:
    """Samples N-way K-shot episodes from a `torchvision.datasets.ImageFolder`.

    Support and query are disjoint by construction: for each sampled class,
    `shot + query` distinct indices are drawn without replacement and then
    split, so a sample can never appear in both sets within the same episode.
    """

    def __init__(self, dataset, seed=None):
        self.dataset = dataset
        self.by_class = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            self.by_class[label].append(idx)
        self.classes = sorted(self.by_class)
        self.rng = random.Random(seed)
        # Memoizes decoded+transformed images: with only a few thousand
        # images total and hundreds of episodes resampling the same pool,
        # re-decoding from disk every time is pure waste (and, on a slow
        # filesystem, the actual bottleneck).
        self._image_cache = {}

    def reset_rng(self, seed):
        """Reseeds episode sampling without discarding the image cache or
        rebuilding by_class -- use this to replay the same episode sequence
        against a different checkpoint while still reusing cached images."""
        self.rng = random.Random(seed)

    def _get_image(self, idx):
        cached = self._image_cache.get(idx)
        if cached is None:
            cached, _ = self.dataset[idx]
            self._image_cache[idx] = cached
        return cached

    def sample_episode(self, n_way, shot, query, device=None):
        if n_way > len(self.classes):
            raise ValueError(f"n_way={n_way} exceeds available classes ({len(self.classes)})")

        classes = self.rng.sample(self.classes, n_way)
        support_imgs, support_labels = [], []
        query_imgs, query_labels = [], []

        for way_idx, cls in enumerate(classes):
            pool = self.by_class[cls]
            need = shot + query
            if need > len(pool):
                raise ValueError(
                    f"class {cls} has {len(pool)} samples, need {need} "
                    f"for shot={shot} + query={query}"
                )
            chosen = self.rng.sample(pool, need)
            support_idx, query_idx = chosen[:shot], chosen[shot:]

            for idx in support_idx:
                support_imgs.append(self._get_image(idx))
                support_labels.append(way_idx)
            for idx in query_idx:
                query_imgs.append(self._get_image(idx))
                query_labels.append(way_idx)

        support_set = torch.stack(support_imgs)
        query_set = torch.stack(query_imgs)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_labels = torch.tensor(query_labels, dtype=torch.long)

        if device is not None:
            support_set = support_set.to(device)
            query_set = query_set.to(device)
            support_labels = support_labels.to(device)
            query_labels = query_labels.to(device)

        return support_set, support_labels, query_set, query_labels
