"""PPO RLHF with a reward model. Simplified."""
import torch, torch.nn.functional as F
from model import MaxTransformer, MaxConfig
import yaml, copy

class RewardModel(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = torch.nn.Linear(base.args.dim, 1)
    def forward(self, ids):
        h = self.base.embed(ids)
        freqs = self.base.freqs_cis[:ids.size(1)]
        for l in self.base.layers: h = l(h, freqs)
        return self.head(self.base.norm(h)[:, -1])

def ppo_step(policy, ref, reward_model, prompts, optim, kl_coef=0.1, clip=0.2):
    # generate
    with torch.no_grad():
        gen = policy_generate(policy, prompts, max_new=256)
        rewards = reward_model(gen).squeeze()
        ref_lp = compute_logprobs(ref, gen)
    pol_lp = compute_logprobs(policy, gen)
    ratio = (pol_lp - ref_lp.detach()).exp()
    adv = rewards - rewards.mean()
    s1 = ratio * adv
    s2 = torch.clamp(ratio, 1-clip, 1+clip) * adv
    policy_loss = -torch.min(s1, s2).mean()
    kl = (pol_lp - ref_lp).mean()
    loss = policy_loss + kl_coef * kl
    loss.backward(); optim.step(); optim.zero_grad()
    return loss.item()

def policy_generate(model, ids, max_new):
    for _ in range(max_new):
        logits, _ = model(ids)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
    return ids

def compute_logprobs(model, ids):
    logits, _ = model(ids)
    lp = F.log_softmax(logits[:, :-1], -1)
    return lp.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)

# Main loop omitted — wire up your prompt dataset
