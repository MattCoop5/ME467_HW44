import contextlib
import importlib
import io
import random

# Import silently because MDP_ENV has top-level print code.
with contextlib.redirect_stdout(io.StringIO()):
    env = importlib.import_module("MDP_ENV")


def simulate_step(state, action):
    """Sample one stochastic transition from (state, action)."""
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

    done = next_state in env.TERMINALS
    return next_state, env.reward(next_state), done


def run_episode(policy, start=(1, 1), max_steps=100):
    """Run one episode and return (total_reward, outcome)."""
    state = start
    total_reward = 0.0

    for _ in range(max_steps):
        if state in env.TERMINALS:
            break

        action = policy.get(state)
        if action is None:
            return total_reward, "timeout"

        next_state, reward, done = simulate_step(state, action)
        total_reward += reward
        state = next_state

        if done:
            break

    if state == env.GOAL:
        return total_reward, "goal"
    if state == env.HAZARD:
        return total_reward, "hazard"
    return total_reward, "timeout"


def evaluate_gamma(gamma, num_episodes=1000, start=(1, 1), max_steps=100):
    """Compute optimal policy for gamma, then evaluate via simulation."""
    env.LIVING_REWARD = -0.04
    V_star, _ = env.value_iteration(gamma=gamma)
    policy = env.extract_policy(V_star, gamma=gamma)

    goal_count = 0
    hazard_count = 0
    total_reward_sum = 0.0

    for _ in range(num_episodes):
        ep_reward, outcome = run_episode(policy, start=start, max_steps=max_steps)
        total_reward_sum += ep_reward
        if outcome == "goal":
            goal_count += 1
        elif outcome == "hazard":
            hazard_count += 1

    goal_rate = goal_count / num_episodes
    hazard_rate = hazard_count / num_episodes
    avg_reward = total_reward_sum / num_episodes
    return goal_rate, hazard_rate, avg_reward


def main():
    random.seed(42)
    gammas = [0.1, 0.5, 0.9, 0.99]
    results = []

    print("Discount-factor study (living reward = -0.04)")
    print("=" * 58)
    print(" gamma | goal_rate | hazard_rate | avg_total_reward")
    print("-" * 58)

    for gamma in gammas:
        goal_rate, hazard_rate, avg_reward = evaluate_gamma(gamma, num_episodes=1000)
        results.append((gamma, goal_rate, hazard_rate, avg_reward))
        print(
            f" {gamma:>5.2f} |"
            f"   {goal_rate:>6.3f}  |"
            f"    {hazard_rate:>6.3f}   |"
            f"     {avg_reward:>8.4f}"
        )

    # "Consistently avoid" interpreted as zero hazard hits in all 1000 episodes.
    first_safe_gamma = None
    for gamma, _, hazard_rate, _ in results:
        if hazard_rate == 0.0:
            first_safe_gamma = gamma
            break

    print("-" * 58)
    if first_safe_gamma is None:
        print("First gamma that consistently avoids hazard: none in tested set")
    else:
        print(
            "First gamma that consistently avoids hazard "
            f"(0 hazard hits / 1000): {first_safe_gamma}"
        )


if __name__ == "__main__":
    main()


# From the terminal:
# ==========================================================
# gamma | goal_rate | hazard_rate | avg_total_reward
# ----------------------------------------------------------
#  0.10 |    1.000  |     0.000   |       0.7192
#  0.50 |    0.979  |     0.021   |       0.6966
#  0.90 |    0.980  |     0.020   |       0.7018
#  0.99 |    1.000  |     0.000   |       0.7196
# ----------------------------------------------------------
# First gamma that consistently avoids hazard (0 hazard hits / 1000): 0.1
