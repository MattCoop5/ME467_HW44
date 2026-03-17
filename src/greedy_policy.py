import contextlib
import importlib
import io
import random

# Import MDP_ENV silently so its top-level demo prints do not appear
with contextlib.redirect_stdout(io.StringIO()):
    env = importlib.import_module("MDP_ENV")

from agent_loop import run_episode


def greedy_action_toward_goal(state, goal=env.GOAL):
    """Pick an action that moves directly toward the goal, ignoring hazards.

    Priority is horizontal first, then vertical:
    - East/West if x differs
    - North/South if y differs
    """
    x, y = state
    gx, gy = goal

    if x < gx:
        return "East"
    if x > gx:
        return "West"
    if y < gy:
        return "North"
    if y > gy:
        return "South"
    return None


def build_greedy_policy(goal=env.GOAL):
    """Create a deterministic greedy policy for all states."""
    policy = {}
    for s in env.STATES:
        if s in env.TERMINALS:
            policy[s] = None
        else:
            policy[s] = greedy_action_toward_goal(s, goal=goal)
    return policy


def run_greedy_trials(num_episodes=1000, start=(1, 1), max_steps=100, seed=42):
    """Run 1000 episodes (or specified count) under the greedy policy."""
    random.seed(seed)

    policy = build_greedy_policy(goal=env.GOAL)

    goal_count = 0
    hazard_count = 0
    total_reward_sum = 0.0

    for _ in range(num_episodes):
        _, total_reward, outcome = run_episode(policy, start=start, max_steps=max_steps)
        total_reward_sum += total_reward
        if outcome == "goal":
            goal_count += 1
        elif outcome == "hazard":
            hazard_count += 1

    goal_fraction = goal_count / num_episodes
    hazard_fraction = hazard_count / num_episodes
    avg_total_reward = total_reward_sum / num_episodes

    print("Greedy goal-directed policy (ignores hazard)")
    print(f"Episodes run: {num_episodes}")
    print(f"(a) Fraction reach goal : {goal_fraction:.4f}")
    print(f"(b) Fraction hit hazard : {hazard_fraction:.4f}")
    print(f"(c) Average total reward: {avg_total_reward:.4f}")

    return goal_fraction, hazard_fraction, avg_total_reward


if __name__ == "__main__":
    run_greedy_trials(num_episodes=1000)


# The MDP optimal policy compares quite differently to the greedy policy.
# The greedy policy reaches the goal 1.4% of the time, and hits the hazard
# 98.6% of the time. Conversely, the optimal policy reaches the goal 100%
# of the time, and hits the hazard 0% of the time.
