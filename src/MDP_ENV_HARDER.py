import random

# Grid dimensions
WIDTH, HEIGHT = 4, 4
# Terminal states and their rewards
GOAL = (4, 4)
HAZARD = (4, 3)
HAZARD2 = (2, 3)  # second hazard
TERMINALS = {GOAL: +1.0, HAZARD: -1.0, HAZARD2: -1.0}
# Living reward for non-terminal states
LIVING_REWARD = -0.04
# All states: every grid cell
STATES = [(x, y) for x in range(1, WIDTH + 1) for y in range(1, HEIGHT + 1)]
# Actions and their (dx, dy) displacements
ACTIONS = {
    "North": (0, 1),
    "South": (0, -1),
    "East": (1, 0),
    "West": (-1, 0),
}


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
    given direction. If the move would leave the grid, return the
    original state (the robot stays put)."""
    dx, dy = ACTIONS[action]
    nx, ny = state[0] + dx, state[1] + dy
    if 1 <= nx <= WIDTH and 1 <= ny <= HEIGHT:
        return (nx, ny)
    return state  # hit a wall, stay in place


def transitions(state, action):
    """T(s' | s, a): return a dict {s': probability} for all reachable
    states s' when taking action a in state s."""
    if state in TERMINALS:
        return {}  # no transitions from terminal states
    # Intended direction (80%) and two perpendicular drifts (10% each)
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
    # Initialize V(s) = 0 for all states
    V = {s: 0.0 for s in STATES}
    iteration = 0
    while True:
        V_new = {}
        delta = 0  # track the largest change
        for s in STATES:
            if s in TERMINALS:
                V_new[s] = reward(s)  # terminal values are fixed
                continue
            # Bellman update: V(s) = R(s) + gamma * max_a sum_s' T(s'|s,a) V(s')
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
            policy[s] = None  # no action in terminal states
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


ARROWS = {"North": "\u2191", "South": "\u2193", "East": "\u2192", "West": "\u2190"}


def display_grid(V, policy):
    """Print the value function and policy as grids."""
    print("Value function V*:")
    for y in range(HEIGHT, 0, -1):
        row = []
        for x in range(1, WIDTH + 1):
            s = (x, y)
            row.append(f"{V[s]:7.3f}")
        print("  ".join(row))
    print("\nOptimal policy:")
    for y in range(HEIGHT, 0, -1):
        row = []
        for x in range(1, WIDTH + 1):
            s = (x, y)
            if s == GOAL:
                row.append("  GOAL ")
            elif s == HAZARD:
                row.append("  HAZD ")
            elif s == HAZARD2:
                row.append("  HZD2 ")
            else:
                row.append(f"   {ARROWS[policy[s]]}   ")
        print("".join(row))


def simulate_episodes(
    policy, num_episodes=1000, max_steps=200, gamma=0.99, start_state=(1, 1)
):
    """Simulate *num_episodes* episodes from start_state following policy.

    Returns
    -------
    avg_return   : float   – mean discounted return across episodes
    goal_count   : int     – number of episodes that reached GOAL
    hazard_count : int     – number of episodes that reached a HAZARD
    step_counts  : list    – steps taken per episode
    """
    total_return = 0.0
    goal_count = 0
    hazard_count = 0
    step_counts = []
    for _ in range(num_episodes):
        state = start_state
        ep_return = 0.0
        discount = 1.0
        steps = 0
        while state not in TERMINALS and steps < max_steps:
            ep_return += discount * reward(state)
            action = policy[state]
            trans = transitions(state, action)
            r_val = random.random()
            cumulative = 0.0
            next_state = state
            for s_next, prob in trans.items():
                cumulative += prob
                if r_val <= cumulative:
                    next_state = s_next
                    break
            state = next_state
            discount *= gamma
            steps += 1
        if state in TERMINALS:
            ep_return += discount * reward(state)
            if state == GOAL:
                goal_count += 1
            else:
                hazard_count += 1
        total_return += ep_return
        step_counts.append(steps)
    avg_return = total_return / num_episodes
    return avg_return, goal_count, hazard_count, step_counts


for r in [-2.0, -0.4, -0.04, -0.01, +0.01]:
    LIVING_REWARD = r
    V, iters = value_iteration(gamma=0.99)
    policy = extract_policy(V, gamma=0.99)
    print(f"\nLiving reward = {r:+.2f} (converged in {iters} iterations)")
    # Show just the policy
    print("Policy:")
    for y in range(HEIGHT, 0, -1):
        row = []
        for x in range(1, WIDTH + 1):
            s = (x, y)
            if s == GOAL:
                row.append(" GOAL")
            elif s == HAZARD:
                row.append(" HAZD")
            elif s == HAZARD2:
                row.append(" HZD2")
            else:
                row.append(f"  {ARROWS[policy[s]]} ")
        print("".join(row))


V, num_iterations = value_iteration(gamma=0.99)
policy = extract_policy(V, gamma=0.99)
print(f"Converged in {num_iterations} iterations.\n")
display_grid(V, policy)

# ── 1000-episode Monte-Carlo evaluation from (1, 1) ──────────────────────────
random.seed(42)
avg_return, goal_count, hazard_count, step_counts = simulate_episodes(
    policy, num_episodes=1000
)
print("\n" + "=" * 50)
print("1000-Episode Simulation  (start = (1,1))")
print("=" * 50)
print(f"  Avg discounted return : {avg_return:.4f}")
print(f"  Goal reached          : {goal_count:4d} / 1000  ({goal_count / 10:.1f}%)")
print(f"  Hazard hit            : {hazard_count:4d} / 1000  ({hazard_count / 10:.1f}%)")
print(f"  Avg steps per episode : {sum(step_counts) / len(step_counts):.1f}")
print("=" * 50)

# The agent's behavior did not change with the addition of a second hazard.
# While that is the case, as the values for the living reward change, and
# the double hazard state causes much higher iteration values when compared
# to the single hazard state. The agent's behavior is still optimal, but the
# agent is more likely to take a longer path to the goal in order to avoid
# the hazards.
