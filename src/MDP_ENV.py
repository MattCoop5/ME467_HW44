# Grid dimensions
WIDTH, HEIGHT = 4, 4
# Terminal states and their rewards
GOAL = (4, 4)
HAZARD = (4, 3)
TERMINALS = {GOAL: +1.0, HAZARD: -1.0}
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
            else:
                row.append(f"   {ARROWS[policy[s]]}   ")
        print("".join(row))


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
            else:
                row.append(f"  {ARROWS[policy[s]]} ")
        print("".join(row))


V, num_iterations = value_iteration(gamma=0.99)
policy = extract_policy(V, gamma=0.99)
print(f"Converged in {num_iterations} iterations.\n")
display_grid(V, policy)
