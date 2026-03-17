import MDP_ENV as env
import random


def run_episode(policy, start=(1, 1), max_steps=100):
    """Simulate one episode following the given policy.
    Returns:
        trajectory: list of states visited (including start)
        total_reward: sum of R(s) over all visited states
        outcome: "goal", "hazard", or "timeout"
    """
    state = start
    trajectory = [state]
    total_reward = env.reward(state)
    for step in range(max_steps):
        if state in env.TERMINALS:
            break
        action = policy[state]
        state = env.simulate_step(state, action)
        trajectory.append(state)
        total_reward += env.reward(state)
    if state == env.GOAL:
        outcome = "goal"
    elif state == env.HAZARD:
        outcome = "hazard"
    else:
        outcome = "timeout"
    return trajectory, total_reward, outcome


random.seed(42)
V, num_iters = env.value_iteration(gamma=0.99)
optimal_policy = env.extract_policy(V, gamma=0.99)
print(f"Value iteration converged in {num_iters} iterations.\n")
results = [run_episode(optimal_policy) for _ in range(1000)]
outcomes = [r[2] for r in results]
rewards = [r[1] for r in results]
goal_rate = outcomes.count("goal") / 1000
hazard_rate = outcomes.count("hazard") / 1000
avg_reward = sum(rewards) / 1000
print(f"Goal reached:  {goal_rate:.3f}")
print(f"Hazard hit:    {hazard_rate:.3f}")
print(f"Avg reward:    {avg_reward:.3f}")
