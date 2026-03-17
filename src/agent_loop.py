import random

import MDP_ENV as env


def simulate_step(state, action):
    """Sample one stochastic transition from (state, action).

    Returns
    -------
    next_state : tuple[int, int]
    reward     : float
    done       : bool
    """
    if state in env.TERMINALS:
        return state, env.reward(state), True

    outcomes = env.transitions(state, action)
    r = random.random()
    cumulative = 0.0
    next_state = state
    for s_next, prob in outcomes.items():
        cumulative += prob
        if r <= cumulative:
            next_state = s_next
            break

    return next_state, env.reward(next_state), next_state in env.TERMINALS


def run_episode(policy, start=(1, 1), max_steps=100):
    """Simulate one full episode following a deterministic policy.

    Parameters
    ----------
    policy : dict
            Mapping from state -> action, i.e., pi*(s).
    start : tuple[int, int]
            Start state for the episode.
    max_steps : int
            Maximum number of transitions before timeout.

    Returns
    -------
    trajectory : list[tuple[int, int]]
            States visited (includes start and each next state).
    total_reward : float
            Undiscounted sum of step rewards from simulate_step.
    outcome : str
            One of: "goal", "hazard", "timeout".
    """
    state = start
    trajectory = [state]
    total_reward = 0.0

    for _ in range(max_steps):
        if state in env.TERMINALS:
            break

        action = policy.get(state)
        if action is None:
            return trajectory, total_reward, "timeout"

        next_state, step_reward, done = simulate_step(state, action)
        total_reward += step_reward
        trajectory.append(next_state)
        state = next_state

        if done:
            break

    if state == env.GOAL:
        outcome = "goal"
    elif state == env.HAZARD:
        outcome = "hazard"
    else:
        outcome = "timeout"

    return trajectory, total_reward, outcome


def run_optimal_policy_trials(num_episodes=1000, start=(1, 1), max_steps=100, seed=42):
    """Run many episodes with the optimal policy from MDP_ENV.

    Reports and returns:
      (a) fraction reaching goal
      (b) fraction hitting hazard
      (c) average total (undiscounted) reward
    """
    random.seed(seed)

    V_star, _ = env.value_iteration(gamma=0.99)
    optimal_policy = env.extract_policy(V_star, gamma=0.99)

    goal_count = 0
    hazard_count = 0
    total_reward_sum = 0.0

    for _ in range(num_episodes):
        _, total_reward, outcome = run_episode(
            optimal_policy,
            start=start,
            max_steps=max_steps,
        )
        total_reward_sum += total_reward
        if outcome == "goal":
            goal_count += 1
        elif outcome == "hazard":
            hazard_count += 1

    goal_fraction = goal_count / num_episodes
    hazard_fraction = hazard_count / num_episodes
    avg_total_reward = total_reward_sum / num_episodes

    print(f"Episodes run: {num_episodes}")
    print(f"(a) Fraction reach goal : {goal_fraction:.4f}")
    print(f"(b) Fraction hit hazard : {hazard_fraction:.4f}")
    print(f"(c) Average total reward: {avg_total_reward:.4f}")

    return goal_fraction, hazard_fraction, avg_total_reward


if __name__ == "__main__":
    run_optimal_policy_trials(num_episodes=1000)
