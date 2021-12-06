# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 11:18:59 2021

@author: Schorsch
"""

import matplotlib.pyplot as plt
import pandas as pd

AGENTS = ['simple_20211107-112443_10000',
          'ppo_20211106-165607_24000',
          'ppo_20211106-212841_8000',
          'ppo_20211107-113147_18000',
          'ppo_20211107-113126_18000']


fig, ax = plt.subplots(figsize=(8,6))

for agent in AGENTS:
    df = pd.read_csv(fr'checkpoints\{agent}.csv')
    ax.plot(df['episode'], df['avg_rewards'], label=agent, alpha=0.5)

ax.set_xlim(left=0, right=16_000)
ax.grid(alpha=0.7)
ax.set_ylabel('reward / stroke / ep')
ax.set_xlabel('episodes')
ax.legend()
plt.tight_layout()
plt.savefig(r'checkpoints\comparison.png', dpi=600)

