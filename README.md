# Adaptive-GRPO-Algorithm-Thesis

## Adaptive Group-Relative Policy Optimization (Adaptive GRPO): A Dynamic Approach to Preference-Based Policy Optimization

**Abstract**

Reinforcement learning from human feedback (RLHF) has become a crucial paradigm for aligning large-scale language models with human preferences. Recent advancements, such as Group-Relative Policy Optimization (GRPO), have introduced effective methods for fine-tuning policies based on ranked feedback without requiring an explicit reward model. However, GRPO relies on static update mechanisms that may not fully leverage the complexity of multi-level preference data. This thesis introduces **Adaptive GRPO (A-GRPO)**, a novel extension that dynamically adjusts optimization strategies based on uncertainty, feedback quality, and policy divergence. By integrating an adaptive loss weighting mechanism and hierarchical preference modeling, **A-GRPO** enhances stability, efficiency, and generalization in preference-based policy optimization.


### 1. Introduction

Modern large-scale language models are often trained using supervised learning and further refined through Reinforcement Learning from Human Feedback (RLHF). Traditional RLHF methods rely on Proximal Policy Optimization (PPO) with reward models, but recent developments such as Direct Preference Optimization (DPO) and Group-Relative Policy Optimization (GRPO) have eliminated the need for explicit reward modeling. GRPO improves upon DPO by leveraging group-level comparisons rather than pairwise ranking, thereby capturing more nuanced preference structures.

Despite its advantages, GRPO suffers from three main limitations:
- Static Optimization Weights – The relative contribution of group preference constraints is fixed, leading to potential under-optimization in uncertain cases.
- Limited Adaptability to Policy Divergence – GRPO applies the same update strength across all training stages, which may cause inefficiencies or overfitting.
- Lack of Uncertainty Awareness – GRPO does not explicitly adjust for the reliability of human feedback, which can vary in quality.

This thesis proposes **Adaptive GRPO (A-GRPO)**, an extension that dynamically adjusts optimization parameters based on feedback confidence, policy divergence, and group preference strength.


### 2. Adaptive GRPO Algorithm

<img width="660" alt="Adaptive GRPO Algorithm" src="https://github.com/user-attachments/assets/51853fd5-c7ea-47ca-bce5-2b7db4167b93" />


**2.4 Algorithm Pseudocode**
```
for each training iteration:
    batch_data = collect_preference_data()
    
    for x, ranked_outputs in batch_data:
        rewards = {}
        for y in ranked_outputs:
            rewards[y] = log(pi_theta(y|x)) - log(pi_ref(y|x))
        
        L_group = 0
        for (y_i, y_j) in group_rankings(ranked_outputs):
            w_ij = compute_confidence(y_i, y_j)
            L_group += -w_ij * log(sigmoid(rewards[y_i] - rewards[y_j]))

        L_policy = compute_policy_loss()
        
        # Adaptive loss weighting
        uncertainty = compute_uncertainty(batch_data)
        lambda_t = lambda_t + eta * (uncertainty - tau)
        
        L_total = lambda_t * L_group + (1 - lambda_t) * L_policy

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
    
    # Adjust clipping threshold based on KL divergence
    epsilon_t = adjust_clipping_threshold()

```


### 3. Experimental Results

**A-GRPO** is evaluated against GRPO and PPO-based RLHF methods on tasks such as text summarization, chatbot fine-tuning, and ranking-based reward modeling. Preliminary results indicate:

Higher stability: A-GRPO avoids the sharp policy oscillations observed in GRPO.
Better reward alignment: Models fine-tuned with A-GRPO generate more preferred outputs than GRPO-trained counterparts.
Faster convergence: Adaptive mechanisms reduce redundant updates, leading to a more efficient training process.


### 4. Conclusion and Future Work

A-GRPO extends GRPO by introducing adaptive weighting mechanisms, hierarchical preference modeling, and policy clipping based on divergence. These enhancements enable more stable, efficient, and interpretable fine-tuning of language models using human preferences.

**Future research directions include:**
- Extending A-GRPO to multi-turn conversational AI settings.
- Integrating reinforcement learning with uncertainty-aware reward shaping.
- Investigating meta-learning techniques to dynamically adjust optimization parameters across tasks.

**Key Contributions of A-GRPO:**
- Dynamic Loss Weighting: Balances between group constraints and policy updates.
- Uncertainty-Aware Ranking: Weights feedback based on confidence levels.
- Adaptive Clipping: Prevents overcorrection in policy updates.
- Superior Performance: More stable training and better reward alignment than GRPO.

A-GRPO represents a significant step forward in preference-based optimization for RLHF, providing a robust framework for fine-tuning AI models in human-aligned tasks.
