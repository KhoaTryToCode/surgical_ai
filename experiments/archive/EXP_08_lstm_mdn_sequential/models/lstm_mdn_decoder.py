import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMMDNDecoder(nn.Module):
    """
    LSTM + Mixture Density Network decoder for sequential polyline prediction.
    
    Architecture (per autoregressive step t):
    ──────────────────────────────────────────────────────────────────────────
    1. INIT TOKEN:     Learnable <INIT> embedding replaces p_{-1} at t=0
    2. LOCAL FEATURE:  F.grid_sample(feature_map, p_{t-1}) → f_local ∈ R^D
    3. INPUT FUSION:   x_t = Linear([p_{t-1}(2); f_local(D)]) → R^D
    4. LSTM STEP:      (h_t, c_t) = LSTM(x_t, (h_{t-1}, c_{t-1}))
    5. MDN HEAD:       y_t = Linear(h_t) → (π, μ_x, μ_y, σ_x, σ_y)×M + eos
    ──────────────────────────────────────────────────────────────────────────
    
    Initialization:
        h_0 = tanh(W_h [GAP(F); e_cls] + b_h)
        c_0 = tanh(W_c [GAP(F); e_cls] + b_c)
    
    Where e_cls is a learned embedding for the landmark class (Ridge, Silhouette, etc.)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        D = config.lstm_hidden_dim       # 256
        M = config.mdn_num_components    # 10
        K = config.num_points            # 20
        C_cls = config.class_embed_dim   # 32
        
        self.D = D
        self.M = M
        self.K = K
        
        # ──── Class Embedding (injected into LSTM initial state) ────
        # num_classes + 1 to handle class 0 (background/unused)
        self.class_embedding = nn.Embedding(config.num_classes + 1, C_cls)
        
        # ──── Learnable <INIT> Token ────
        # This replaces p_{-1} and f_local at t=0 since there is no previous point.
        # It is a single learnable vector that acts as the "start of sequence" signal.
        self.init_token = nn.Parameter(torch.randn(1, D) * 0.02)
        
        # ──── LSTM Initial State Projections ────
        # Maps [v_global(D) ; e_cls(C_cls)] → h_0, c_0
        init_input_dim = D + C_cls
        self.init_h_proj = nn.Linear(init_input_dim, D)
        self.init_c_proj = nn.Linear(init_input_dim, D)
        
        # ──── Step Input Fusion ────
        # At each step t ≥ 1: fuse [p_{t-1}(2-dim) ; f_local(D-dim)] → D
        # At step t = 0: the <INIT> token is already D-dimensional, bypass this
        self.input_fusion = nn.Sequential(
            nn.Linear(2 + D, D),
            nn.LayerNorm(D),
            nn.GELU()
        )
        
        # ──── LSTM Core ────
        self.lstm = nn.LSTM(
            input_size=D,
            hidden_size=D,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            batch_first=False  # We process step-by-step manually
        )
        
        # ──── MDN Output Head ────
        # Per step: 5*M params (π̂, μ_x, μ_y, σ̂_x, σ̂_y per component) + 1 eos logit
        mdn_output_dim = 5 * M + 1
        self.mdn_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, mdn_output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MDN head with small weights to prevent initial sigma explosion."""
        for m in self.mdn_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize sigma biases to log(0.05) ≈ -3.0 for tight initial predictions
        # The last linear layer outputs [π̂(M), μ_x(M), μ_y(M), σ̂_x(M), σ̂_y(M), eos(1)]
        final_linear = self.mdn_head[-1]
        with torch.no_grad():
            # σ̂_x occupies indices [3*M : 4*M], σ̂_y occupies [4*M : 5*M]
            final_linear.bias[3 * self.M: 5 * self.M] = math.log(0.05)
    
    def _sample_local_feature(self, feature_map: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """
        Bilinear grid_sample to extract local visual features at a continuous coordinate.
        
        Args:
            feature_map: (B, D, H_f, W_f) or (D, H_f, W_f) for single instance
            point: (2,) normalized coordinate (u, v) in [0, 1]
            
        Returns:
            f_local: (D,) local feature vector
        """
        # Convert [0,1] → [-1,1] for grid_sample
        grid_point = point * 2.0 - 1.0  # (2,)
        
        # grid_sample expects (B, C, H, W) input and (B, H_out, W_out, 2) grid
        if feature_map.dim() == 3:
            feat = feature_map.unsqueeze(0)  # (1, D, H, W)
        else:
            feat = feature_map
        
        # Create grid: (1, 1, 1, 2) — sampling a single point
        grid = grid_point.view(1, 1, 1, 2)  # (1, 1, 1, 2) → (x, y) order
        
        # F.grid_sample expects grid in (x, y) order which matches (u, v)
        sampled = F.grid_sample(
            feat, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (1, D, 1, 1)
        
        return sampled.squeeze()  # (D,)
    
    def _parse_mdn_output(self, raw: torch.Tensor):
        """
        Parse raw MDN head output into activated parameters.
        
        Args:
            raw: (5*M + 1,) raw output vector
            
        Returns:
            dict with keys:
                pi:      (M,) mixture weights (softmax)
                mu_x:    (M,) x-coordinate means (raw, clamped at inference)
                mu_y:    (M,) y-coordinate means (raw, clamped at inference)
                sigma_x: (M,) x standard deviations (exp activation)
                sigma_y: (M,) y standard deviations (exp activation)
                eos:     scalar end-of-sequence probability (sigmoid)
        """
        M = self.M
        
        pi_logits = raw[0:M]               # (M,)
        mu_x      = raw[M:2*M]             # (M,)
        mu_y      = raw[2*M:3*M]           # (M,)
        sigma_x_raw = raw[3*M:4*M]         # (M,)
        sigma_y_raw = raw[4*M:5*M]         # (M,)
        eos_logit = raw[5*M]               # scalar
        
        pi = F.softmax(pi_logits, dim=-1)
        sigma_x = torch.exp(sigma_x_raw).clamp(min=1e-5, max=1.0)
        sigma_y = torch.exp(sigma_y_raw).clamp(min=1e-5, max=1.0)
        eos = torch.sigmoid(eos_logit)
        
        return {
            "pi": pi,
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "eos": eos,
            "pi_logits": pi_logits,
            "eos_logit": eos_logit,
            "raw": raw
        }
    
    def _get_expected_point(self, mdn_params: dict) -> torch.Tensor:
        """
        Compute the expected (mean) point from MDN mixture: p̂ = Σ_j π_j [μ_x_j, μ_y_j]
        
        Returns:
            (2,) expected coordinate
        """
        mu = torch.stack([mdn_params["mu_x"], mdn_params["mu_y"]], dim=-1)  # (M, 2)
        return (mdn_params["pi"].unsqueeze(-1) * mu).sum(dim=0)  # (2,)
    
    def forward_teacher_forced(
        self,
        feature_map: torch.Tensor,
        cls_id: int,
        gt_polyline: torch.Tensor
    ) -> dict:
        """
        Teacher-forced forward pass for a single landmark instance.
        The LSTM receives GT coordinates as input at each step (no autoregression).
        
        Args:
            feature_map: (D, H_f, W_f) backbone feature map for this image
            cls_id: integer class ID for this landmark instance
            gt_polyline: (K, 2) ground truth polyline in [0, 1]^2
            
        Returns:
            dict with:
                mdn_params: list of K dicts, each containing parsed MDN params
                expected_points: (K, 2) expected coordinates from MDN
                raw_outputs: (K, 5*M+1) raw MDN outputs for loss computation
        """
        device = feature_map.device
        K = gt_polyline.shape[0]
        
        # ──── 1. Compute LSTM initial state ────
        # Global Average Pool the feature map
        v_global = feature_map.mean(dim=(-2, -1))  # (D,)
        
        # Class embedding
        cls_tensor = torch.tensor([cls_id], device=device, dtype=torch.long)
        e_cls = self.class_embedding(cls_tensor).squeeze(0)  # (C_cls,)
        
        # Project to initial LSTM states
        init_input = torch.cat([v_global, e_cls], dim=-1)  # (D + C_cls,)
        h_0 = torch.tanh(self.init_h_proj(init_input)).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        c_0 = torch.tanh(self.init_c_proj(init_input)).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        
        h_t, c_t = h_0, c_0
        
        # Storage
        all_mdn_params = []
        all_expected_points = []
        all_raw_outputs = []
        
        for t in range(K):
            # ──── 2. Construct step input ────
            if t == 0:
                # First step: use learnable <INIT> token
                x_t = self.init_token.to(device)  # (1, D)
            else:
                # Subsequent steps: teacher-forced GT coordinate + local feature
                p_prev = gt_polyline[t - 1]  # (2,) GT coordinate from previous step
                f_local = self._sample_local_feature(feature_map, p_prev)  # (D,)
                
                # Fuse: [p_prev(2) ; f_local(D)] → D
                fused_input = torch.cat([p_prev, f_local], dim=-1)  # (2 + D,)
                x_t = self.input_fusion(fused_input).unsqueeze(0)  # (1, D)
            
            # ──── 3. LSTM step ────
            lstm_out, (h_t, c_t) = self.lstm(x_t.unsqueeze(0), (h_t, c_t))
            # lstm_out: (1, 1, D)
            
            # ──── 4. MDN head ────
            raw_output = self.mdn_head(lstm_out.squeeze(0).squeeze(0))  # (5*M + 1,)
            mdn_params = self._parse_mdn_output(raw_output)
            expected_pt = self._get_expected_point(mdn_params)  # (2,)
            
            all_mdn_params.append(mdn_params)
            all_expected_points.append(expected_pt)
            all_raw_outputs.append(raw_output)
        
        expected_points = torch.stack(all_expected_points, dim=0)  # (K, 2)
        raw_outputs = torch.stack(all_raw_outputs, dim=0)          # (K, 5*M+1)
        
        return {
            "mdn_params": all_mdn_params,
            "expected_points": expected_points,
            "raw_outputs": raw_outputs
        }
    
    @torch.no_grad()
    def forward_autoregressive(
        self,
        feature_map: torch.Tensor,
        cls_id: int,
        max_steps: int = None
    ) -> dict:
        """
        Autoregressive inference: the model feeds its own predicted coordinates back as input.
        
        Args:
            feature_map: (D, H_f, W_f) backbone feature map
            cls_id: integer class ID
            max_steps: maximum steps before forced termination (default: K + 5)
            
        Returns:
            dict with:
                predicted_points: (T, 2) predicted polyline coordinates
                eos_probs: (T,) end-of-sequence probabilities per step
                mdn_params: list of T dicts with MDN parameters
        """
        device = feature_map.device
        K = self.K
        if max_steps is None:
            max_steps = K + 5  # Allow some extra steps for EOS detection
        
        # Initial state
        v_global = feature_map.mean(dim=(-2, -1))
        cls_tensor = torch.tensor([cls_id], device=device, dtype=torch.long)
        e_cls = self.class_embedding(cls_tensor).squeeze(0)
        
        init_input = torch.cat([v_global, e_cls], dim=-1)
        h_t = torch.tanh(self.init_h_proj(init_input)).unsqueeze(0).unsqueeze(0)
        c_t = torch.tanh(self.init_c_proj(init_input)).unsqueeze(0).unsqueeze(0)
        
        predicted_points = []
        eos_probs = []
        all_mdn_params = []
        
        for t in range(max_steps):
            if t == 0:
                x_t = self.init_token.to(device)
            else:
                p_prev = predicted_points[-1].detach()  # Use own prediction
                # Clamp to valid range for grid_sample
                p_prev_clamped = p_prev.clamp(0.0, 1.0)
                f_local = self._sample_local_feature(feature_map, p_prev_clamped)
                fused_input = torch.cat([p_prev_clamped, f_local], dim=-1)
                x_t = self.input_fusion(fused_input).unsqueeze(0)
            
            lstm_out, (h_t, c_t) = self.lstm(x_t.unsqueeze(0), (h_t, c_t))
            raw_output = self.mdn_head(lstm_out.squeeze(0).squeeze(0))
            mdn_params = self._parse_mdn_output(raw_output)
            expected_pt = self._get_expected_point(mdn_params)
            
            predicted_points.append(expected_pt)
            eos_probs.append(mdn_params["eos"])
            all_mdn_params.append(mdn_params)
            
            # Early stop if EOS probability exceeds threshold
            if mdn_params["eos"].item() > 0.5 and t >= K - 1:
                break
        
        return {
            "predicted_points": torch.stack(predicted_points, dim=0),  # (T, 2)
            "eos_probs": torch.stack(eos_probs, dim=0),               # (T,)
            "mdn_params": all_mdn_params
        }
