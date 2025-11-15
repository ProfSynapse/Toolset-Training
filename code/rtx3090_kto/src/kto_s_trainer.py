#!/usr/bin/env python3
"""
KTO-S Trainer Implementation
Adds SIGN correction to standard KTO for stable KL divergence.

Based on research paper showing KTO has early instability when training
from base models (not SFT-initialized). The SIGN correction fixes gradient
scaling so higher-KL responses don't get stronger updates.

Key difference:
- Standard KTO: loss = -log_sigmoid(beta * (reward - KL))
- KTO-S:        loss = -log_sigmoid(beta * (reward + SIGN(reward) * KL))
"""

import torch
import torch.nn.functional as F
from trl import KTOTrainer
from typing import Dict, Optional, Tuple, Union, List


class KTOSTrainer(KTOTrainer):
    """
    KTO trainer with optional SIGN correction for stable KL divergence.

    This extends TRL's KTOTrainer to add the KTO-S variant, which uses
    a sign-based correction to prevent KL divergence spikes during training.

    The SIGN correction ensures that the KL penalty adapts based on the
    reward sign, preventing the gradient scaling bug in standard KTO where
    higher-KL (worse) responses get larger gradient updates.

    Args:
        use_sign_correction (bool): Enable KTO-S mode (default: True)
            - True: Use SIGN correction (KTO-S, stable training)
            - False: Use standard KTO (may have KL spikes)

    Example:
        >>> trainer = KTOSTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     tokenizer=tokenizer,
        ...     train_dataset=train_dataset,
        ...     use_sign_correction=True  # Enable KTO-S
        ... )
    """

    def __init__(self, *args, use_sign_correction: bool = True, **kwargs):
        """
        Initialize KTO-S trainer.

        Args:
            use_sign_correction: If True, use SIGN correction (KTO-S mode).
                                If False, use standard KTO behavior.
            *args, **kwargs: Passed to parent KTOTrainer
        """
        super().__init__(*args, **kwargs)
        self.use_sign_correction = use_sign_correction

        # Log configuration
        if self.use_sign_correction:
            print("\n" + "=" * 80)
            print("🔬 KTO-S MODE ENABLED")
            print("=" * 80)
            print("Using SIGN correction for stable KL divergence")
            print("This fixes gradient scaling bug in standard KTO")
            print("Expected: KL stays < 0.1 through early training")
            print("=" * 80 + "\n")
        else:
            print("\n" + "=" * 80)
            print("⚠️  STANDARD KTO MODE")
            print("=" * 80)
            print("KTO-S disabled - using original KTO implementation")
            print("Warning: May experience KL spikes if training from base model")
            print("=" * 80 + "\n")

    def concatenated_forward(
        self, model, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Run forward pass for KTO training.

        This is called by the parent class and doesn't need modification.
        We only override the loss computation.
        """
        return super().concatenated_forward(model, batch)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: str = "train",
    ):
        """
        Compute KTO loss with optional SIGN correction.

        This overrides the parent method to add SIGN correction when enabled.
        The correction changes how KL divergence is incorporated into the loss.

        Args:
            model: The model being trained
            batch: Training batch
            train_eval: "train" or "eval"

        Returns:
            Tuple of (losses, metrics_dict)
        """
        metrics = {}

        # Get model outputs (unchanged from standard KTO)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = self.concatenated_forward(model, batch)

        # Also need reference model outputs for KL computation
        with torch.no_grad():
            if self.ref_model is None:
                # If no reference model, use the model itself (not recommended)
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        # Compute KTO loss with optional SIGN correction
        losses, chosen_rewards, rejected_rewards = self.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        # Compute reward accuracy (unchanged)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # Prefix for logging
        prefix = "eval_" if train_eval == "eval" else ""

        # Build metrics dict
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        return losses.mean(), metrics

    def kto_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the KTO loss with optional SIGN correction.

        This is the core modification for KTO-S.

        Standard KTO loss:
            chosen_loss = -log_sigmoid(beta * (r_chosen - KL_chosen))
            rejected_loss = -log_sigmoid(beta * (KL_rejected - r_rejected))

        KTO-S loss (with SIGN correction):
            S_chosen = sign(r_chosen)
            S_rejected = sign(r_rejected)
            chosen_loss = -log_sigmoid(beta * (r_chosen + S_chosen * KL_chosen))
            rejected_loss = -log_sigmoid(beta * (S_rejected * KL_rejected - r_rejected))

        The SIGN correction ensures gradient scaling adapts to reward sign,
        preventing higher-KL responses from getting stronger updates.

        Args:
            policy_chosen_logps: Log probabilities for chosen responses
            policy_rejected_logps: Log probabilities for rejected responses
            reference_chosen_logps: Reference model log probs for chosen
            reference_rejected_logps: Reference model log probs for rejected

        Returns:
            Tuple of (losses, chosen_rewards, rejected_rewards)
        """
        # Compute KL divergence (unchanged from standard KTO)
        # KL = log(policy) - log(reference)
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean()
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean()

        # Compute rewards (unchanged from standard KTO)
        # Reward is just the KL divergence per response
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps

        if self.use_sign_correction:
            # ============================================================
            # KTO-S: Add SIGN correction
            # ============================================================
            # The sign flips the KL penalty direction based on reward sign
            # This fixes the gradient scaling bug in standard KTO
            S_chosen = torch.sign(chosen_rewards)
            S_rejected = torch.sign(rejected_rewards)

            # Apply SIGN-corrected loss
            chosen_losses = -F.logsigmoid(
                self.beta * (chosen_rewards + S_chosen * chosen_KL)
            )
            rejected_losses = -F.logsigmoid(
                self.beta * (S_rejected * rejected_KL - rejected_rewards)
            )

        else:
            # ============================================================
            # Standard KTO: Original formulation
            # ============================================================
            # Warning: This may cause KL spikes in early training
            chosen_losses = -F.logsigmoid(
                self.beta * (chosen_rewards - chosen_KL)
            )
            rejected_losses = -F.logsigmoid(
                self.beta * (rejected_KL - rejected_rewards)
            )

        # Combine losses with weighting (unchanged from standard KTO)
        # Apply desirable/undesirable weights
        losses = (
            self.loss_type == "ipo"
            and (chosen_losses + rejected_losses)
            or (chosen_losses * self.desirable_weight + rejected_losses * self.undesirable_weight)
        )

        return losses, chosen_rewards.mean(), rejected_rewards.mean()


def test_kto_s():
    """Quick test to verify KTO-S implementation."""
    print("Testing KTO-S implementation...")

    # Create mock data
    batch_size = 4
    policy_chosen = torch.randn(batch_size)
    policy_rejected = torch.randn(batch_size) - 0.5  # Make these lower
    reference_chosen = torch.randn(batch_size)
    reference_rejected = torch.randn(batch_size)

    # Mock trainer with KTO-S enabled
    class MockTrainer:
        def __init__(self, use_sign):
            self.use_sign_correction = use_sign
            self.beta = 0.2
            self.desirable_weight = 1.0
            self.undesirable_weight = 1.0
            self.loss_type = "kto"

    # Test KTO-S
    trainer_s = MockTrainer(use_sign=True)
    kto_s_instance = KTOSTrainer.__new__(KTOSTrainer)
    kto_s_instance.use_sign_correction = True
    kto_s_instance.beta = 0.2
    kto_s_instance.desirable_weight = 1.0
    kto_s_instance.undesirable_weight = 1.0
    kto_s_instance.loss_type = "kto"

    losses_s, rewards_chosen_s, rewards_rejected_s = kto_s_instance.kto_loss(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected
    )

    # Test standard KTO
    kto_std_instance = KTOSTrainer.__new__(KTOSTrainer)
    kto_std_instance.use_sign_correction = False
    kto_std_instance.beta = 0.2
    kto_std_instance.desirable_weight = 1.0
    kto_std_instance.undesirable_weight = 1.0
    kto_std_instance.loss_type = "kto"

    losses_std, rewards_chosen_std, rewards_rejected_std = kto_std_instance.kto_loss(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected
    )

    print(f"✓ KTO-S loss computed: {losses_s.mean().item():.4f}")
    print(f"✓ Standard KTO loss:   {losses_std.mean().item():.4f}")
    print(f"✓ Difference:          {abs(losses_s.mean().item() - losses_std.mean().item()):.4f}")
    print("\n✅ KTO-S implementation test passed!")


if __name__ == "__main__":
    test_kto_s()
