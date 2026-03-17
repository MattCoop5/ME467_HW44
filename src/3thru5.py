import random
from mdp_agent import (
    ACTIONS,
    ARROWS,
    GOAL,
    HAZARD,
    HEIGHT,
    STATES,
    TERMINALS,
    WIDTH,
    WALL,
    extract_policy,
    reward,
    run_episode,
    value_iteration,
)


def greedy_policy_action(state):
    """Move toward the goal, ignoring hazards."""
    gx, gy = GOAL
    sx, sy = state
    if sx < gx:
        return "East"
    elif sx > gx:
        return "West"
    elif sy < gy:
        return "North"
    else:
        return "South"


greedy_policy = {s: greedy_policy_action(s) for s in STATES if s not in TERMINALS}
greedy_policy[GOAL] = None
greedy_policy[HAZARD] = None

random.seed(42)
V, _ = value_iteration(gamma=0.99)
optimal_policy = extract_policy(V, gamma=0.99)
opt_results = [run_episode(optimal_policy) for _ in range(1000)]
opt_outcomes = [r[2] for r in opt_results]
opt_rewards = [r[1] for r in opt_results]
greedy_results = [run_episode(greedy_policy) for _ in range(1000)]
greedy_outcomes = [r[2] for r in greedy_results]
greedy_rewards = [r[1] for r in greedy_results]
print("Optimal policy:")
print(f"  Goal reached:  {opt_outcomes.count('goal') / 1000:.3f}")
print(f"  Hazard hit:    {opt_outcomes.count('hazard') / 1000:.3f}")
print(f"  Avg reward:    {sum(opt_rewards) / 1000:.3f}")
print("\nGreedy policy:")
print(f"  Goal reached:  {greedy_outcomes.count('goal') / 1000:.3f}")
print(f"  Hazard hit:    {greedy_outcomes.count('hazard') / 1000:.3f}")
print(f"  Avg reward:    {sum(greedy_rewards) / 1000:.3f}")

random.seed(42)
print("Discount factor experiment (living reward = -0.04):\n")
for gamma in [0.1, 0.5, 0.9, 0.99]:
    V, _ = value_iteration(gamma=gamma)
    policy = extract_policy(V, gamma=gamma)
    results = [run_episode(policy) for _ in range(1000)]
    outcomes = [r[2] for r in results]
    rewards = [r[1] for r in results]
    goal_rate = outcomes.count("goal") / 1000
    hazard_rate = outcomes.count("hazard") / 1000
    avg_reward = sum(rewards) / 1000
    print(
        f"gamma={gamma:.2f}  goal={goal_rate:.3f}  "
        f"hazard={hazard_rate:.3f}  avg_reward={avg_reward:.3f}"
    )

random.seed(42)
TERMINALS[(2, 3)] = -1.0
V, _ = value_iteration(gamma=0.99)
policy = extract_policy(V, gamma=0.99)
print("Policy with second hazard at (2,3):\n")
for y in range(HEIGHT, 0, -1):
    row = []
    for x in range(1, WIDTH + 1):
        s = (x, y)
        if s == GOAL:
            row.append(" GOAL")
        elif s in TERMINALS and TERMINALS[s] < 0:
            row.append(" HAZD")
        elif s not in STATES:
            row.append(" WALL")
        else:
            row.append(f"  {ARROWS[policy[s]]} ")
    print("".join(row))
results = [run_episode(policy) for _ in range(1000)]
outcomes = [r[2] for r in results]
rewards = [r[1] for r in results]
print(f"\nGoal reached:  {outcomes.count('goal') / 1000:.3f}")
print(f"Hazard hit:    {outcomes.count('hazard') / 1000:.3f}")
print(f"Avg reward:    {sum(rewards) / 1000:.3f}")
