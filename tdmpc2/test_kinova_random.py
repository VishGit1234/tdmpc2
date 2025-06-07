
import torch
from types import SimpleNamespace

from envs import make_env

class DummyCfg(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


cfg = DummyCfg(
    # ---------- make_env requirements ----------
    num_envs=8,                  # fewer envs → faster test
    obs="state",
    # ---------- KinovaEnv physical / reward ----------
    episode_length_s=5,
    init_joint_angles=[
        6.9761, 1.1129, 1.7474, -2.2817, 7.5884, -1.1489,
        1.6530, 0.8213, 0.8200, 0.8209, 0.8208, 0.8217, 0.8210
    ],
    init_quat=[0, 0, 0, 1],
    bracelet_link_height=0.25,
    init_box_pos=[0.2, 0.2, 0.002],
    box_size=[0.08, 0.08, 0.02],
    clip_actions=0.01,
    termination_if_cube_goal_dist_less_than=0.01,
    cube_goal_dist_rew_scale=10,
    cube_arm_dist_rew_scale=10,
    success_reward=1,
    target_displacement=0.3,
    action_scale=0.01,
    # ---------- placeholders that make_env will overwrite ----------
    obs_shape=None,
    action_dim=None,
    episode_length=None,
    seed_steps=None,
)


env = make_env(cfg)     # returns a KinovaEnv instance

print(f"\n✓ KinovaEnv created with {len(env.domain_set)} predefined domains.\n")

N_RESETS = 50
hist = torch.zeros(len(env.domain_set), dtype=torch.int32)

for i in range(N_RESETS):
    env.reset()                              # triggers random domain choice
    doms = env.domain_idx.cpu()              # vector length == cfg.num_envs
    print(f"Reset {i:02d}: {doms.tolist()}")
    for d in doms:
        hist[d] += 1
        

print("\n── Domain-usage histogram ──")
for d, cnt in enumerate(hist.tolist()):
    print(f"Domain {d}: {cnt} samples")

assert (hist > 0).all(), (
    "❌  Some domains were never selected – check your randomization logic!"
)
print("\n✅  Domain randomization appears to be working.")
