"""
Dual-Stream DataLoader for Multi-Dataset Training (EXPERIMENT_12).
Synchronizes:
  - L3D Landmark Loader (690 training frames, WeightedRandomSampler)
  - CholecSeg8k Scene Loader (8,080 frames, Infinite Cyclical Stream)
"""
from torch.utils.data import DataLoader


class DualStreamDataLoader:
    """
    Pairs each L3D landmark batch with a synchronized CholecSeg8k scene batch.
    The epoch length is governed by L3D, ensuring standard epoch logging and scheduling,
    while CholecSeg8k continuously supplies non-repeating diverse surgical scenes.
    """
    def __init__(self, l3d_loader: DataLoader, cholec_loader: DataLoader):
        self.l3d_loader = l3d_loader
        self.cholec_loader = cholec_loader
        self._cholec_iter = self._infinite_iterator(cholec_loader)

    @staticmethod
    def _infinite_iterator(loader):
        while True:
            for batch in loader:
                yield batch

    def __iter__(self):
        for l3d_batch in self.l3d_loader:
            cholec_batch = next(self._cholec_iter)
            yield l3d_batch, cholec_batch

    def __len__(self):
        return len(self.l3d_loader)
