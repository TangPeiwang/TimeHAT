import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.encoder import EncoderLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layers = configs.e_layers
        self.keep_ratio = configs.keep_ratio
        self.patch_len = configs.patch_len
        self.forecast_mode = getattr(configs, 'forecast_mode', 'parallel')
        if self.forecast_mode not in ('parallel', 'recursive'):
            raise ValueError('forecast_mode must be parallel or recursive')
        self.recursive_readout = getattr(configs, 'recursive_readout', 'future_token')
        if self.recursive_readout not in ('future_token', 'last_token'):
            raise ValueError('recursive_readout must be future_token or last_token')
        self.recursive_window = getattr(configs, 'recursive_window', 'fixed')
        if self.recursive_window not in ('fixed', 'growing'):
            raise ValueError('recursive_window must be fixed or growing')
        if self.patch_len <= 0 or self.pred_len <= 0:
            raise ValueError('patch_len/pred_len must be positive')
        self.full_atten_layers = configs.full_atten_layers

        if self.task_name != 'long_term_forecast':
            self.pred_len = configs.seq_len

        self.enc_embedding = nn.Linear(self.patch_len, configs.d_model, bias=False)
        self.emb_dropout = nn.Dropout(configs.dropout)

        self.forecast_token = nn.Parameter(torch.randn(1, 1, configs.d_model) * 0.02)

        full_layers = [EncoderLayer(configs, layer_type="bi") for i in range(self.full_atten_layers)]
        mix_layers = [EncoderLayer(configs, layer_type="mix") for i in range(self.layers - self.full_atten_layers)]
        self.encoder = nn.ModuleList(full_layers + mix_layers)

        if self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.patch_len * configs.enc_in, configs.num_class)
        else:
            self.decoder_output_len = (self.pred_len
                                       if self.task_name == 'long_term_forecast' and self.forecast_mode == 'recursive'
                                       else self.patch_len)
            self.decoder = nn.Linear(configs.d_model, self.decoder_output_len)


    def mask_point_patch(self, xb):
        bs, l = xb.shape
        mask_ratio = 1 - self.keep_ratio
        P = self.patch_len
        x = xb.view(bs, -1, P)
        bs, p_l, p = x.shape
        device = x.device

        # ---------- 0. Task Split and Quota Allocation ----------
        total_mask_points = int(l * mask_ratio)
        patch_mask_points = int(total_mask_points * 0.8) # Allocate 80% quota to Patch masking
        total_masked_patches = max(1, patch_mask_points // P)
        total_masked_patches = min(total_masked_patches, p_l - 1)  # Keep at least 1 visible patch

        # Random task assignment: 40% for forecasting (extrapolation), 60% for reconstruction (Auto-Encoding)
        is_tail = torch.rand(bs, device=device) < 0.4

        # ---------- 1. Determine learnable [FORECAST] replacement regions ----------
        # For the forecasting group, randomly select k patches after the truncation point
        k = torch.randint(1, p_l - 7, (bs,), device=device)

        pos = torch.arange(p_l, device=device).unsqueeze(0)
        tail_f_mask = pos >= (p_l - k.unsqueeze(1)) # Forecasting group: last k patches
        ae_f_mask = pos == (p_l - 1)                # AE group: only the last patch

        # Merge to generate the final forecast_mask [bs, p_l]
        forecast_mask = torch.where(is_tail.unsqueeze(1), tail_f_mask, ae_f_mask)

        # ---------- 2. Generate physical mask matrix for AE group (ae_mask_matrix) ----------
        # Assign remaining patch masks for AE (excluding the last anchor patch)
        ae_patches_to_assign = total_masked_patches - 1

        scores = torch.rand(bs, p_l, device=device)
        # Strictly forbid AE mask on the last patch (reserved for forecast_mask)
        scores[:, -1] = -100.0

        ids_shuffle = torch.argsort(scores, dim=-1, descending=True)
        ae_mask_matrix = torch.zeros(bs, p_l, dtype=torch.bool, device=device)
        ae_mask_matrix.scatter_(1, ids_shuffle[:, :ae_patches_to_assign], True)

        # Core protection: The forecasting group (is_tail) strictly cannot have historical patch masks
        ae_mask_matrix[is_tail] = False

        # ---------- 3. Precisely assign AE filling types (Zero / Mean / Shuffle) ----------
        zero_cnt = max(0, int(ae_patches_to_assign * 0.8))
        mean_cnt = int(ae_patches_to_assign * 0.1)
        shuffle_cnt = ae_patches_to_assign - zero_cnt - mean_cnt

        assign_scores = torch.rand(bs, p_l, device=device)
        assign_scores[~ae_mask_matrix] = -1.0 # Compete only within AE mask regions
        _, sorted_idx = torch.sort(assign_scores, dim=-1, descending=True)

        types_seq = torch.arange(p_l, device=device).unsqueeze(0)
        types = torch.where(types_seq < zero_cnt, torch.tensor(0, device=device),
                torch.where(types_seq < zero_cnt + mean_cnt, torch.tensor(1, device=device),
                            torch.tensor(2, device=device))).expand(bs, -1)

        patch_mask_types = torch.full((bs, p_l), -1, dtype=torch.long, device=device)
        patch_mask_types.scatter_(1, sorted_idx, types)

        # ---------- 4. Safely compute visible patch mean (Prevent Data Leakage) ----------
        # Visible patches: neither in the forecast region nor in the AE masked region
        visible_mask = ~(forecast_mask | ae_mask_matrix)
        visible_sum = (x * visible_mask.unsqueeze(-1)).sum(dim=(1,2), keepdim=True)
        visible_count = visible_mask.sum(dim=1, keepdim=True).unsqueeze(-1) * p
        global_mean = visible_sum / visible_count.clamp(min=1)

        # ---------- 5. Apply Physical Patch Transformations ----------
        x_masked = x.clone()

        # 5.0 Clear forecast region (prevent interference, will be replaced by Embeddings later)
        x_masked[forecast_mask] = 0.0

        # 5.1 AE: Zero padding
        x_masked[(patch_mask_types == 0) & ae_mask_matrix] = 0.0

        # 5.2 AE: Mean padding
        mean_mask = ((patch_mask_types == 1) & ae_mask_matrix).unsqueeze(-1)
        x_masked = torch.where(mean_mask, global_mean, x_masked)

        # 5.3 AE: Intra-patch shuffle
        shuffle_mask = (patch_mask_types == 2) & ae_mask_matrix
        if shuffle_mask.any():
            b_idx, p_idx = shuffle_mask.nonzero(as_tuple=True)
            rand_shuffle = torch.rand(len(b_idx), p, device=device).argsort(dim=-1)
            x_masked[b_idx, p_idx] = torch.gather(x[b_idx, p_idx], 1, rand_shuffle)

        # =====================================================================
        # 5.4 Simulating downstream Left-Padding (Only for Forecasting group)
        # Goal: Enhance Zero-shot / Few-shot robustness for arbitrary history lengths
        # =====================================================================
        # 1. Randomly select 20% of forecasting samples to simulate indivisible lengths
        is_padded = is_tail & (torch.rand(bs, device=device) < 0.2)

        if is_padded.any():
            # 2. Generate random truncation points (1 to P-1) to simulate missing remainders
            padding_lens = torch.randint(1, P, (bs,), device=device)

            # 3. Construct point-level mask [bs, P]
            point_indices = torch.arange(P, device=device).unsqueeze(0)
            pad_mask = point_indices < padding_lens.unsqueeze(1)

            # 4. Get the first valid value (simulating mode='replicate')
            first_valid_vals = torch.gather(x[:, 0, :], 1, padding_lens.unsqueeze(1))

            # 5. Apply replicate padding simulation
            # Only affects the first patch of padded sequences
            pad_mask = pad_mask & is_padded.unsqueeze(1)
            x_masked[:, 0, :] = torch.where(pad_mask, first_valid_vals, x_masked[:, 0, :])
        # =====================================================================

        # ---------- 6. Point-level Fallback Masking ----------
        # Summarize all physically altered and reserved regions
        change_mask = (forecast_mask | ae_mask_matrix).unsqueeze(-1).expand(-1, -1, p).clone()
        already_masked = change_mask.sum(dim=(1,2))

        # Remaining mask points to fulfill
        remain = (total_mask_points - already_masked).clamp(min=0)

        # Core protection: Forecasting group strictly cannot have point-level masks (keep history 100% pure)
        remain[is_tail] = 0

        available = (~change_mask).sum(dim=(1,2))
        point_mask_ratio = (remain / available.clamp(min=1)).clamp(max=1.0).view(bs, 1, 1)

        point_mask = (torch.rand(bs, p_l, p, device=device) < point_mask_ratio) & (~change_mask)
        x_masked[point_mask] = 0.0
        change_mask[point_mask] = True

        return x_masked.reshape(bs, l), change_mask.reshape(bs, l), forecast_mask


    def patch_embedding(self, x_flat):
        """
        Optimized: Directly accepts flattened Channel Independence (CI) data [B*V, L]
        Avoids permute operations to reduce overhead.
        """
        x_patches = x_flat.view(x_flat.size(0), -1, self.patch_len)
        x_emb = self.enc_embedding(x_patches)
        x_emb = self.emb_dropout(x_emb)
        return x_emb

    def pre_training(self, x_enc):
        """
        Unified dual-objective pre-training forward pass (Reconstruction + Forecasting)
        """
        B, L, V = x_enc.shape

        # ========== 1. Data Alignment and Normalization (RevIN) ==========
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Convert to Channel Independence (CI) view
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # [B, V, L]
        x_enc_flat = x_enc.view(B * V, L)            # [B*V, L]
        x_ori_flat = x_enc_flat.clone()              # Keep Ground Truth [B*V, L]

        # ========== 2. Mixed Mask Preparation ==========
        # forecast_mask indicates where to insert learnable parameters
        x_masked_flat, change_mask_flat, forecast_mask = self.mask_point_patch(x_enc_flat)

        # ========== 3. Feature Mapping and Probe Injection ==========
        x_emb = self.patch_embedding(x_masked_flat)

        # Expand the unique forecast_token to align with the batch
        num_patches = x_emb.size(1)
        token_embeds = self.forecast_token.expand(B * V, num_patches, -1)

        # Replace physical 0 values with predictive probes at locations specified by forecast_mask
        x_input = torch.where(forecast_mask.unsqueeze(-1), token_embeds, x_emb)

        # ========== 4. Forward through Backbone ==========
        # Pass forecast_mask to build attention barriers in bidirectional/mixed attention layers
        for mod in self.encoder:
            x_input, _ = mod(x_input, forecast_mask=forecast_mask)

        # ========== 5. Decoding and Unified Loss Calculation ==========
        # Reduce dimension back to physical space [B*V, num_patches, P] -> [B*V, L]
        dec_out = self.decoder(x_input).view(B * V, L)

        # Calculate base MSE loss
        raw_loss = (dec_out - x_ori_flat) ** 2

        # Expand patch-level forecast_mask to point-level, aligned to [B*V, L]
        f_mask_points = forecast_mask.unsqueeze(-1).expand(-1, -1, self.patch_len).reshape(B * V, L)

        # Target A: Auto-Encoding reconstruction loss (L_Recon)
        # Calculated only in physically perturbed and non-forecasting regions
        ae_loss_mask = change_mask_flat & (~f_mask_points)
        loss_ae = (raw_loss * ae_loss_mask).sum() / ae_loss_mask.sum().clamp(min=1.0)

        # Target B: Forecasting extrapolation loss (L_Forecast)
        # Calculated only in forecasting probe regions (enables Zero-shot capability)
        loss_forecast = (raw_loss * f_mask_points).sum() / f_mask_points.sum().clamp(min=1.0)

        # Total loss (forecasting weight can be adjusted)
        total_loss = loss_ae + 1.0 * loss_forecast

        return total_loss


    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """One direct training/validation pass using the selected recursive readout."""
        return self._forecast_parallel(x_enc, self.pred_len)

    @torch.no_grad()
    def predict(self, x, horizon=None):
        """Predict with the selected readout and a fixed or growing recursive history."""
        if self.training:
            raise RuntimeError('Call model.eval() before predict()')
        horizon = self.pred_len if horizon is None else horizon
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError('horizon must be a positive integer')
        if self.forecast_mode == 'parallel':
            return self._forecast_parallel(x, horizon)
        history = x
        window = x.size(1)
        outputs = []
        for _ in range(0, horizon, self.pred_len):
            chunk = self._forecast_parallel(history, self.pred_len)
            outputs.append(chunk)
            history = torch.cat([history, chunk], dim=1)
            if self.recursive_window == 'fixed':
                history = history[:, -window:, :]
        return torch.cat(outputs, dim=1)[:, :horizon, :]

    def _forecast_parallel(self, x_enc, pred_len):
        """
        Prefix-LM parallel downstream forecasting.
        Supports arbitrary history and prediction lengths (Zero-shot extrapolation).
        """
        # ================= 1. Instance Normalization (RevIN) =================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        b, l, v = x_enc.shape
        P = self.patch_len

        # ================= 2. Handle Indivisible History Lengths =================
        # Calculate padding length. If exactly divisible, pad_len is 0
        pad_len = (P - (l % P)) % P
        if pad_len > 0:
            # Use left edge replicate padding to protect the latest rightmost data
            x_enc = F.pad(x_enc, pad=(0, 0, pad_len, 0), mode='replicate')

        # ================= 3. Data Projection and Embedding =================
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc_flat = torch.reshape(x_enc, (b * v, -1))

        x_emb = self.patch_embedding(x_enc_flat)
        num_hist_patches = x_emb.size(1)

        # Last-history-token readout uses no forecast probes or forecast mask.
        if self.forecast_mode == 'recursive' and self.recursive_readout == 'last_token':
            enc_out = x_emb
            for mod in self.encoder:
                enc_out, _ = mod(enc_out)
            dec_out = self.decoder(enc_out[:, -1:, :])
            dec_out = dec_out.reshape(b, v, self.decoder_output_len).permute(0, 2, 1)
            return dec_out[:, :pred_len, :] * stdev + means

        # ================= 4. Dynamic Probe Injection =================
        # Calculate required probes to cover the target prediction length
        output_len = self.decoder_output_len
        k = (pred_len + output_len - 1) // output_len

        # Duplicate the pre-trained global token k times
        forecast_tokens = self.forecast_token.expand(b * v, k, -1)

        # Append to the history sequence; RoPE will assign future coordinates
        x_input = torch.cat([x_emb, forecast_tokens], dim=1)

        # ================= 5. Build Prefix-LM Attention Mask =================
        # Mask indicates history (False) and probes (True).
        # Mix Attention automatically applies causal masks for these probes.
        forecast_mask = torch.zeros(b * v, num_hist_patches + k, dtype=torch.bool, device=x_input.device)
        forecast_mask[:, -k:] = True

        # ================= 6. Forward through Backbone =================
        for mod in self.encoder:
            x_input, _ = mod(x_input, forecast_mask=forecast_mask)

        # ================= 7. Extract, Decode, and Truncate =================
        # 1. Extract only the last k probes corresponding to the future
        enc_out = x_input[:, -k:, :]

        # 2. Map back to physical space
        dec_out = self.decoder(enc_out)

        # 3. Flatten and truncate to exact prediction length
        dec_out = dec_out.view(b * v, k * output_len)
        dec_out = dec_out[:, :pred_len]

        # 4. Restore to standard 3D tensor
        dec_out = dec_out.view(b, v, pred_len).permute(0, 2, 1).contiguous()

        # ================= 8. Denormalization (De-RevIN) =================
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))

        return dec_out


    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        B, L, V = x_enc.shape

        # ================= 1. Masked Instance Normalization (Masked RevIN) =================
        # Mean and variance computation must ignore points where mask == 0
        means = torch.sum(x_enc, dim=1, keepdim=True) / torch.sum(mask == 1, dim=1, keepdim=True).clamp(min=1)
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1, keepdim=True) /
                           torch.sum(mask == 1, dim=1, keepdim=True).clamp(min=1) + 1e-5)
        x_enc /= stdev

        # ================= 2. Convert to CI View =================
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc_flat = x_enc.view(B * V, L)

        # ================= 3. Embedding & Encoder =================
        x_emb = self.patch_embedding(x_enc_flat)
        for mod in self.encoder:
            x_emb, _ = mod(x_emb)

        # ================= 4. Decoding =================
        dec_out = self.decoder(x_emb)
        dec_out = dec_out.view(B, V, L).permute(0, 2, 1).contiguous()

        # ================= 5. Denormalization (De-RevIN) =================
        dec_out = dec_out * stdev + means
        return dec_out


    def anomaly_detection(self, x_enc):
        B, L, V = x_enc.shape

        # ================= 1. Instance Normalization (RevIN) =================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # ================= 2. Convert to CI View =================
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc_flat = x_enc.view(B * V, L)

        # ================= 3. Embedding & Encoder =================
        x_emb = self.patch_embedding(x_enc_flat)
        for mod in self.encoder:
            x_emb, _ = mod(x_emb)

        # ================= 4. Decoding =================
        # Note: Anomaly detection is a reconstruction task, so it uses the decoder.
        dec_out = self.decoder(x_emb)
        dec_out = dec_out.view(B, V, L).permute(0, 2, 1).contiguous()

        # ================= 5. Denormalization (De-RevIN) =================
        dec_out = dec_out * stdev + means
        return dec_out


    def classification(self, x_enc, x_mark_enc):
        B, L, V = x_enc.shape

        # ================= 1. Instance Normalization (RevIN) =================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # ================= 2. Convert to CI View =================
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc_flat = x_enc.view(B * V, L)

        # ================= 3. Embedding & Encoder =================
        x_emb = self.patch_embedding(x_enc_flat)
        for mod in self.encoder:
            x_emb, _ = mod(x_emb)

        # ================= 4. Extract [CLS] Equivalent Feature =================
        # Extract only the last patch as the global semantic feature
        cls_token = x_emb[:, -1, :]

        # ================= 5. Feature Flattening and Projection =================
        # Concatenate CLS features across channels
        output = cls_token.view(B, -1)

        output = self.dropout(output)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        if self.task_name == 'pre_training':
            loss = self.pre_training(x_enc)
            return loss
        else:
            if self.task_name == 'long_term_forecast':
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
            if self.task_name == 'imputation':
                dec_out = self.imputation(
                    x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
                return dec_out
            if self.task_name == 'anomaly_detection':
                dec_out = self.anomaly_detection(x_enc)
                return dec_out
            if self.task_name == 'classification':
                dec_out = self.classification(x_enc, x_mark_enc)
                return dec_out
        return None
