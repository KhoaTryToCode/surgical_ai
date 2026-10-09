"""
Patient-Stratified Clip Sampler for EXPERIMENT_15.
Ensures that every patient cohort has equal probability of being sampled per iteration,
preventing large patient cohorts (Patient 38, Patient 50) from drowning out smaller ones.
"""
from collections import defaultdict
import random
import torch
from torch.utils.data import Sampler

class PatientStratifiedClipSampler(Sampler):
    def __init__(self, dataset, samples_per_epoch=None, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        
        # Group clip indices by patient
        self.patient_to_indices = defaultdict(list)
        for idx, clip in enumerate(dataset.clips):
            self.patient_to_indices[clip['patient']].append(idx)
            
        self.patients = sorted(list(self.patient_to_indices.keys()))
        self.samples_per_epoch = samples_per_epoch if samples_per_epoch is not None else len(dataset)

    def __iter__(self):
        indices = []
        for _ in range(self.samples_per_epoch):
            # 1. Pick a patient uniformly at random
            chosen_pat = random.choice(self.patients)
            # 2. Pick a random clip from that patient's available pool
            chosen_clip_idx = random.choice(self.patient_to_indices[chosen_pat])
            indices.append(chosen_clip_idx)
            
        return iter(indices)

    def __len__(self):
        return self.samples_per_epoch
