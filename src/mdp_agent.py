import random
from collections import Counter

# Grid dimensions
WIDTH, HEIGHT = 4, 3
# Terminal states and their rewards
GOAL = (4, 3)
HAZARD = (4, 2)
TERMINALS = {GOAL: +1.0, HAZARD: -1.0}
# Living reward for non-terminal states
LIVING_REWARD = -0.04
# Wall position
WALL = (2, 2)
# All valid states: every grid cell except the wall
STATES = [
    (x, y) for x in range(1, WIDTH + 1) for y in range(1, HEIGHT + 1) if (x, y) != WALL
]
# Actions and their (dx, dy) displacements
ACTIONS = {
    "North": (0, 1),
    "South": (0, -1),
    "East": (1, 0),
    "West": (-1, 0),
}
ARROWS = {"North": "\u2191", "South": "\u2193", "East": "\u2192", "West": "\u2190"}


def reward(state):
    """R(s): immediate reward for being in state s."""
    if state in TERMINALS:
        return TERMINALS[state]
    return LIVING_REWARD


def get_perpendicular(action):
    """Return the two actions perpendicular to the given action."""
    if action in ("North", "South"):
        return ["West", "East"]
    else:
        return ["North", "South"]


def attempt_move(state, action):
    """Return the state that results from attempting to move in the
    given direction. If the move would leave the grid or hit a wall,
    return the original state."""
    dx, dy = ACTIONS[action]
    nx, ny = state[0] + dx, state[1] + dy
    if 1 <= nx <= WIDTH and 1 <= ny <= HEIGHT and (nx, ny) != WALL:
        return (nx, ny)
    return state


def transitions(state, action):
    """T(s' | s, a): return a dict {s': probability}."""
    if state in TERMINALS:
        return {}
    outcomes = {}
    intended = attempt_move(state, action)
    outcomes[intended] = outcomes.get(intended, 0) + 0.8
    for perp in get_perpendicular(action):
        drifted = attempt_move(state, perp)
        outcomes[drifted] = outcomes.get(drifted, 0) + 0.1
    return outcomes


def value_iteration(gamma=0.99, epsilon=1e-6):
    """Run value iteration and return the converged value function
    and the number of iterations."""
    V = {s: 0.0 for s in STATES}
    iteration = 0
    while True:
        V_new = {}
        delta = 0
        for s in STATES:
            if s in TERMINALS:
                V_new[s] = reward(s)
                continue
            best_value = float("-inf")
            for a in ACTIONS:
                expected = sum(
                    prob * V[s_next] for s_next, prob in transitions(s, a).items()
                )
                best_value = max(best_value, expected)
            V_new[s] = reward(s) + gamma * best_value
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        iteration += 1
        if delta < epsilon:
            break
    return V, iteration


def extract_policy(V, gamma=0.99):
    """Compute the optimal policy from a converged value function."""
    policy = {}
    for s in STATES:
        if s in TERMINALS:
            policy[s] = None
            continue
        best_action = None
        best_value = float("-inf")
        for a in ACTIONS:
            expected = sum(
                prob * V[s_next] for s_next, prob in transitions(s, a).items()
            )
            value = reward(s) + gamma * expected
            if value > best_value:
                best_value = value
                best_action = a
        policy[s] = best_action
    return policy


def simulate_step(state, action):
    """Sample a next state from T(s' | s, a)."""
    dist = transitions(state, action)
    states = list(dist.keys())
    probs = list(dist.values())
    return random.choices(states, weights=probs, k=1)[0]


random.seed(42)
counts = Counter()
for _ in range(10_000):
    counts[simulate_step((3, 1), "North")] += 1
print("Empirical transition frequencies from (3,1), action North:")
for s, c in sorted(counts.items()):
    print(f"  {s}: {c / 10_000:.3f}")


def run_episode(policy, start=(1, 1), max_steps=100):
    """Simulate one episode following the given policy.
    Returns:
        trajectory: list of states visited (including start)
        total_reward: sum of R(s) over all visited states
        outcome: "goal", "hazard", or "timeout"
    """
    state = start
    trajectory = [state]
    total_reward = reward(state)
    for step in range(max_steps):
        if state in TERMINALS:
            break
        action = policy[state]
        state = simulate_step(state, action)
        trajectory.append(state)
        total_reward += reward(state)
    if state == GOAL:
        outcome = "goal"
    elif state == HAZARD:
        outcome = "hazard"
    else:
        outcome = "timeout"
    return trajectory, total_reward, outcome


random.seed(42)
V, num_iters = value_iteration(gamma=0.99)
optimal_policy = extract_policy(V, gamma=0.99)
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
