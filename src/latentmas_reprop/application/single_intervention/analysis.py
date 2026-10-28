"""Post-hoc research analysis utilities using ResearchDataReader without re-running inference."""

from typing import Any

import torch
import torch.nn.functional as F

from ...infrastructure.storage import ResearchDataReader
from .decoder import get_lm_head


class SingleInterventionAnalyzer:
    """Provides UC-04 post-hoc analysis capabilities over saved experiment data."""

    def __init__(
        self,
        reader: ResearchDataReader | None = None,
        model_module: torch.nn.Module | None = None,
    ) -> None:
        self.reader = reader or ResearchDataReader()
        self.model = model_module
        self._lm_head = get_lm_head(model_module) if model_module else None

    def load_common_state(
        self, run_id: str, sample_key: str
    ) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        """Load common hidden trajectory, KV snapshot, and metadata."""
        hidden = self.reader.load_tensor(run_id, "hidden", sample_key=sample_key)
        kv = self.reader.load_tensor(run_id, "kv", sample_key=sample_key)
        meta = self.reader.load_artifact(
            run_id, f"samples/{sample_key}/shared/result.json"
        )
        return hidden, kv, meta

    def load_condition_state(
        self, run_id: str, sample_key: str, condition: str
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Load condition hidden trajectory and result record."""
        hidden = self.reader.load_tensor(
            run_id, "hidden", sample_key=sample_key, condition=condition
        )
        meta = self.reader.load_artifact(
            run_id, f"samples/{sample_key}/{condition}/result.json"
        )
        return hidden, meta

    def compute_latent_step_logits_and_entropy(
        self,
        latent_hidden: torch.Tensor,
        lm_head: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vocabulary logits and Shannon entropy across latent step trajectory."""
        head = lm_head or self._lm_head
        if head is None:
            raise ValueError(
                "An lm_head module or output embeddings is required to compute logits."
            )
        logits = head(latent_hidden)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        return logits, entropy

    def analyze_trajectory_state_deltas(
        self, trajectory_hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute Euclidean step distance and cosine similarity along the hidden trajectory."""
        if trajectory_hidden.shape[0] < 2:
            return {
                "step_distances": trajectory_hidden.new_empty(0),
                "step_cosine_similarities": trajectory_hidden.new_empty(0),
            }
        diffs = trajectory_hidden[1:] - trajectory_hidden[:-1]
        distances = torch.linalg.norm(diffs, dim=-1)
        norms = torch.linalg.norm(trajectory_hidden, dim=-1)
        dots = torch.sum(trajectory_hidden[1:] * trajectory_hidden[:-1], dim=-1)
        cos_sims = dots / (norms[1:] * norms[:-1] + 1e-12)
        return {
            "step_distances": distances,
            "step_cosine_similarities": cos_sims,
        }

    def compare_transition_distributions(
        self,
        hiddens_dict: dict[str, torch.Tensor],
        lm_head: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute next-token probability distribution comparisons right after transition point."""
        head = lm_head or self._lm_head
        if head is None:
            raise ValueError("An lm_head module is required to compute distributions.")
        probs_dict = {}
        for name, h in hiddens_dict.items():
            probe = h[0] if h.dim() == 2 and h.shape[0] > 0 else h
            logits = head(probe)
            probs_dict[name] = F.softmax(logits, dim=-1)
        return probs_dict

    def run_additional_intervention(
        self,
        run_id: str,
        sample_key: str,
        intervention_tokens: torch.Tensor,
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[Any, torch.Tensor]:
        """UC-04: Perform a novel intervention experiment reusing the saved branch KV cache."""
        _, kv_snapshot, _ = self.load_common_state(run_id, sample_key)
        from ...domain.services.kv_cache import (
            clone_past_kv,
            get_past_kv_sequence_length,
            move_past_kv,
        )

        kv = move_past_kv(clone_past_kv(kv_snapshot), device)
        past_len = get_past_kv_sequence_length(kv)
        mask = torch.ones(
            (1, past_len + intervention_tokens.shape[1]),
            dtype=torch.long,
            device=device,
        )
        out = model(
            input_ids=intervention_tokens.to(device),
            attention_mask=mask,
            past_key_values=kv,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.past_key_values, out.hidden_states[-1][0].detach().cpu()
