import os
import json
import random
import math
from typing import Dict, List, Tuple

import streamlit as st

from crewai import Agent, Task, Crew, Process, LLM


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Delivery Agent",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef6ff 50%,
            #ffffff 100%
        );
    }

    /* Main container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Header */
    .hero {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #edf6ff 100%
        );
        border: 1px solid #d8e7f5;
        border-radius: 24px;
        padding: 30px 35px;
        margin-bottom: 25px;
        box-shadow: 0 10px 35px rgba(35, 90, 140, 0.08);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #12344d;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #557086;
    }

    /* Pipeline */
    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin: 20px 0 30px 0;
    }

    .pipeline-item {
        background: white;
        border: 1px solid #d9e8f5;
        border-radius: 14px;
        padding: 12px 17px;
        color: #23445b;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(40, 80, 120, 0.06);
    }

    .arrow {
        color: #4b8dcc;
        font-size: 22px;
        font-weight: bold;
    }

    /* Cards */
    .card {
        background: white;
        border: 1px solid #dceaf5;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 6px 20px rgba(30, 80, 120, 0.06);
    }

    .card-title {
        font-size: 20px;
        font-weight: 800;
        color: #173f59;
        margin-bottom: 10px;
    }

    /* Agent cards */
    .agent-card {
        background: linear-gradient(
            135deg,
            #ffffff,
            #f5faff
        );
        border: 1px solid #d5e7f5;
        border-radius: 16px;
        padding: 17px;
        margin-bottom: 10px;
    }

    .agent-name {
        font-weight: 800;
        color: #174766;
        font-size: 16px;
    }

    .agent-status {
        color: #5e7585;
        font-size: 13px;
        margin-top: 4px;
    }

    /* Metrics */
    .metric-card {
        background: white;
        border: 1px solid #dceaf5;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 5px 18px rgba(30, 80, 120, 0.05);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #1769aa;
    }

    .metric-label {
        color: #657b8b;
        font-size: 13px;
        margin-top: 4px;
    }

    /* Route node */
    .route-node {
        display: inline-block;
        background: #ffffff;
        border: 2px solid #6aa9d8;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 4px;
        color: #19435d;
        font-weight: 700;
    }

    .route-arrow {
        color: #5194c9;
        font-size: 20px;
        font-weight: bold;
    }

    /* Status */
    .success-box {
        background: #eefaf3;
        border: 1px solid #b9e3ca;
        color: #1d6b3d;
        border-radius: 14px;
        padding: 15px;
    }

    .info-box {
        background: #eef7ff;
        border: 1px solid #c5dff5;
        color: #245b83;
        border-radius: 14px;
        padding: 15px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 45px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f8fbfe;
        border-right: 1px solid #dceaf5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DELIVERY ENVIRONMENT
# ============================================================

LOCATIONS = {
    "Warehouse": (0, 0),
    "A": (1, 0),
    "B": (2, 0),
    "C": (3, 0),
    "D": (0, 1),
    "E": (1, 1),
    "F": (2, 1),
    "G": (3, 1),
    "H": (0, 2),
    "I": (1, 2),
    "J": (2, 2),
    "K": (3, 2),
}

NEIGHBORS = {
    "Warehouse": ["A", "D"],
    "A": ["Warehouse", "B", "E"],
    "B": ["A", "C", "F"],
    "C": ["B", "G"],
    "D": ["Warehouse", "E", "H"],
    "E": ["D", "A", "F", "I"],
    "F": ["E", "B", "G", "J"],
    "G": ["F", "C", "K"],
    "H": ["D", "I"],
    "I": ["H", "E", "J"],
    "J": ["I", "F", "K"],
    "K": ["J", "G"],
}

PACKAGES = {
    "P1": {
        "destination": "C",
        "priority": "Critical",
        "deadline": 18,
    },
    "P2": {
        "destination": "I",
        "priority": "High",
        "deadline": 28,
    },
    "P3": {
        "destination": "K",
        "priority": "Normal",
        "deadline": 40,
    },
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def distance(a: str, b: str) -> float:
    x1, y1 = LOCATIONS[a]
    x2, y2 = LOCATIONS[b]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def traffic_factor(level: str) -> float:
    return {
        "Low": 1.0,
        "Medium": 1.25,
        "High": 1.6,
    }[level]


def package_priority_value(priority: str) -> float:
    return {
        "Critical": 3.0,
        "High": 2.0,
        "Normal": 1.0,
    }[priority]


# ============================================================
# Q-LEARNING DELIVERY AGENT
# ============================================================

class DeliveryEnvironment:

    def __init__(
        self,
        time_weight=0.4,
        cost_weight=0.2,
        priority_weight=0.3,
        risk_weight=0.1,
        traffic="Medium",
    ):

        self.time_weight = time_weight
        self.cost_weight = cost_weight
        self.priority_weight = priority_weight
        self.risk_weight = risk_weight
        self.traffic = traffic

        self.reset()

    def reset(self):

        self.location = "Warehouse"
        self.time = 0
        self.cost = 0
        self.distance_travelled = 0

        self.delivered = []

        return self.get_state()

    def get_state(self):

        remaining = tuple(
            p for p in PACKAGES
            if p not in self.delivered
        )

        return (
            self.location,
            remaining,
        )

    def available_actions(self):

        actions = list(NEIGHBORS[self.location])

        for package_id, package in PACKAGES.items():

            if (
                package["destination"] == self.location
                and package_id not in self.delivered
            ):
                actions.append(f"DELIVER_{package_id}")

        return actions

    def step(self, action):

        reward = 0
        done = False

        # Delivery action
        if action.startswith("DELIVER_"):

            package_id = action.replace("DELIVER_", "")

            if package_id in self.delivered:
                reward -= 10

            elif PACKAGES[package_id]["destination"] != self.location:
                reward -= 20

            else:

                self.delivered.append(package_id)

                package = PACKAGES[package_id]

                priority_bonus = (
                    package_priority_value(
                        package["priority"]
                    )
                    * 15
                )

                reward += (
                    60
                    + priority_bonus
                )

                if self.time <= package["deadline"]:
                    reward += 30
                else:
                    reward -= 30

        # Movement action
        else:

            if action not in NEIGHBORS[self.location]:

                reward -= 25

            else:

                step_distance = distance(
                    self.location,
                    action
                )

                factor = traffic_factor(
                    self.traffic
                )

                travel_time = step_distance * factor

                travel_cost = step_distance * 2

                self.location = action

                self.time += travel_time
                self.cost += travel_cost
                self.distance_travelled += step_distance

                # Time penalty
                reward -= (
                    travel_time
                    * self.time_weight
                )

                # Cost penalty
                reward -= (
                    travel_cost
                    * self.cost_weight
                )

                # Traffic / risk penalty
                if self.traffic == "High":
                    reward -= 2 * self.risk_weight

        # Completion
        if len(self.delivered) == len(PACKAGES):

            reward += 100
            done = True

        # Maximum time
        if self.time > 70:

            reward -= 80
            done = True

        return self.get_state(), reward, done


# ============================================================
# Q-LEARNING
# ============================================================

def choose_action(q_table, state, actions, epsilon):

    if not actions:
        return None

    if random.random() < epsilon:
        return random.choice(actions)

    values = [
        q_table.get((state, action), 0.0)
        for action in actions
    ]

    max_value = max(values)

    best = [
        action
        for action, value in zip(actions, values)
        if value == max_value
    ]

    return random.choice(best)


def train_q_learning(
    time_weight,
    cost_weight,
    priority_weight,
    risk_weight,
    traffic,
    episodes=700,
):

    q_table = {}

    learning_rate = 0.15
    discount = 0.90

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.992

    episode_rewards = []

    for episode in range(episodes):

        env = DeliveryEnvironment(
            time_weight=time_weight,
            cost_weight=cost_weight,
            priority_weight=priority_weight,
            risk_weight=risk_weight,
            traffic=traffic,
        )

        state = env.reset()

        total_reward = 0

        for _ in range(150):

            actions = env.available_actions()

            if not actions:
                break

            action = choose_action(
                q_table,
                state,
                actions,
                epsilon,
            )

            next_state, reward, done = env.step(
                action
            )

            old_q = q_table.get(
                (state, action),
                0.0
            )

            next_actions = env.available_actions()

            if next_actions:

                max_next_q = max(
                    q_table.get(
                        (next_state, next_action),
                        0.0
                    )
                    for next_action in next_actions
                )

            else:
                max_next_q = 0.0

            new_q = old_q + learning_rate * (
                reward
                + discount * max_next_q
                - old_q
            )

            q_table[(state, action)] = new_q

            state = next_state

            total_reward += reward

            if done:
                break

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        episode_rewards.append(
            total_reward
        )

    return q_table, episode_rewards


# ============================================================
# RUN LEARNED POLICY
# ============================================================

def run_policy(
    q_table,
    time_weight,
    cost_weight,
    priority_weight,
    risk_weight,
    traffic,
):

    env = DeliveryEnvironment(
        time_weight=time_weight,
        cost_weight=cost_weight,
        priority_weight=priority_weight,
        risk_weight=risk_weight,
        traffic=traffic,
    )

    state = env.reset()

    route = ["Warehouse"]
    decisions = []

    total_reward = 0

    for _ in range(100):

        actions = env.available_actions()

        if not actions:
            break

        # Exploit learned policy
        action = choose_action(
            q_table,
            state,
            actions,
            epsilon=0
        )

        q_value = q_table.get(
            (state, action),
            0
        )

        previous_location = env.location

        next_state, reward, done = env.step(
            action
        )

        total_reward += reward

        decisions.append({
            "location": previous_location,
            "action": action,
            "q_value": q_value,
            "reward": reward,
        })

        if not action.startswith("DELIVER_"):

            if env.location != route[-1]:
                route.append(env.location)

        else:

            package_id = action.replace(
                "DELIVER_",
                ""
            )

            route.append(
                f"📦 {package_id} delivered"
            )

        state = next_state

        if done:
            break

    return {
        "route": route,
        "decisions": decisions,
        "reward": total_reward,
        "time": env.time,
        "cost": env.cost,
        "distance": env.distance_travelled,
        "delivered": env.delivered,
        "q_table": q_table,
    }


# ============================================================
# GROQ / CREWAI
# ============================================================

def get_crewai_llm():

    return LLM(
        model="groq/openai/gpt-oss-120b",
        temperature=0.2,
        max_tokens=1000,
    )


def run_planner_agent(user_instruction):

    llm = get_crewai_llm()

    planner = Agent(
        role="Delivery Mission Planner",
        goal=(
            "Convert a human delivery instruction into "
            "clear optimization priorities."
        ),
        backstory=(
            "You are an intelligent logistics planner. "
            "You translate human delivery goals into "
            "simple numerical priorities."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
        Analyze this delivery instruction:

        "{user_instruction}"

        Return ONLY valid JSON with these fields:

        {{
          "time_weight": number between 0 and 1,
          "cost_weight": number between 0 and 1,
          "priority_weight": number between 0 and 1,
          "risk_weight": number between 0 and 1,
          "summary": "short explanation"
        }}

        The four weights should add up to approximately 1.0.

        Interpret the user's priorities.
        If they emphasize speed, increase time_weight.
        If they emphasize cost, increase cost_weight.
        If they emphasize urgent packages, increase priority_weight.
        If they emphasize safety/reliability, increase risk_weight.
        """,
        expected_output="Valid JSON object only.",
        agent=planner,
    )

    crew = Crew(
        agents=[planner],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    text = str(result)

    # Extract JSON safely
    try:
        start = text.find("{")
        end = text.rfind("}") + 1

        data = json.loads(
            text[start:end]
        )

    except Exception:

        # Safe fallback
        data = {
            "time_weight": 0.4,
            "cost_weight": 0.2,
            "priority_weight": 0.3,
            "risk_weight": 0.1,
            "summary": (
                "Balanced delivery optimization."
            ),
        }

    # Normalize weights
    weights = [
        float(data.get("time_weight", 0.4)),
        float(data.get("cost_weight", 0.2)),
        float(data.get("priority_weight", 0.3)),
        float(data.get("risk_weight", 0.1)),
    ]

    total = sum(weights)

    if total <= 0:
        weights = [0.4, 0.2, 0.3, 0.1]
        total = 1

    weights = [
        value / total
        for value in weights
    ]

    data["time_weight"] = weights[0]
    data["cost_weight"] = weights[1]
    data["priority_weight"] = weights[2]
    data["risk_weight"] = weights[3]

    return data


def run_explanation_agent(
    instruction,
    objective,
    result,
):

    llm = get_crewai_llm()

    explainer = Agent(
        role="Delivery Decision Explainer",
        goal=(
            "Explain reinforcement learning delivery "
            "decisions clearly to a human."
        ),
        backstory=(
            "You are an AI logistics analyst. "
            "You explain what the RL agent learned "
            "without pretending the RL system is an LLM."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
        Explain this delivery decision.

        Human instruction:
        {instruction}

        Objective:
        {json.dumps(objective, indent=2)}

        RL results:
        Route:
        {result["route"]}

        Total reward:
        {result["reward"]:.2f}

        Time:
        {result["time"]:.2f} minutes

        Distance:
        {result["distance"]:.2f} km

        Cost:
        {result["cost"]:.2f}

        Delivered:
        {result["delivered"]}

        Explain in 2-4 short paragraphs:

        1. What the human wanted.
        2. How the objective influenced the RL reward.
        3. Why the learned route was selected.
        4. What the final result means.

        Do not claim that the LLM itself selected the route.
        The route was selected by the Q-learning policy.
        """,
        expected_output="A concise human-readable explanation.",
        agent=explainer,
    )

    crew = Crew(
        agents=[explainer],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result_text = crew.kickoff()

    return str(result_text)


# ============================================================
# SESSION STATE
# ============================================================

if "objective" not in st.session_state:
    st.session_state.objective = None

if "q_table" not in st.session_state:
    st.session_state.q_table = None

if "training_rewards" not in st.session_state:
    st.session_state.training_rewards = []

if "delivery_result" not in st.session_state:
    st.session_state.delivery_result = None

if "explanation" not in st.session_state:
    st.session_state.explanation = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🚚 Smart Delivery Agent
        </div>

        <div class="hero-subtitle">
            LLM-Guided Reinforcement Learning for
            Multi-Objective Delivery Optimization
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="pipeline">

        <div class="pipeline-item">
            👤 Human Instruction
        </div>

        <div class="arrow">→</div>

        <div class="pipeline-item">
            🤖 CrewAI + Groq
        </div>

        <div class="arrow">→</div>

        <div class="pipeline-item">
            🎯 Objective
        </div>

        <div class="arrow">→</div>

        <div class="pipeline-item">
            🧠 Q-Learning
        </div>

        <div class="arrow">→</div>

        <div class="pipeline-item">
            🌍 Environment
        </div>

        <div class="arrow">→</div>

        <div class="pipeline-item">
            📦 Delivery
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎛️ Mission Control")

    st.markdown(
        "Configure the delivery environment."
    )

    traffic = st.selectbox(
        "🚦 Traffic Level",
        ["Low", "Medium", "High"],
        index=1,
    )

    episodes = st.slider(
        "🧠 Training Episodes",
        min_value=200,
        max_value=1500,
        value=700,
        step=100,
    )

    st.divider()

    st.markdown("### 📦 Packages")

    for package_id, package in PACKAGES.items():

        st.write(
            f"**{package_id}** → "
            f"{package['destination']}  "
            f"({package['priority']})"
        )

    st.divider()

    st.markdown(
        """
        **RL Parameters**

        Learning Rate: `0.15`

        Discount Factor: `0.90`

        Exploration: `ε-greedy`

        Algorithm: `Q-Learning`
        """
    )


# ============================================================
# HUMAN INSTRUCTION
# ============================================================

st.markdown(
    '<div class="card-title">👤 Human Mission</div>',
    unsafe_allow_html=True,
)

instruction = st.text_area(
    "Tell the delivery agent what you want:",
    value=(
        "Deliver the packages as quickly as possible. "
        "The critical package should receive the highest priority, "
        "but keep the delivery cost reasonable."
    ),
    height=100,
    label_visibility="collapsed",
)


if st.button(
    "🤖 Understand Mission with CrewAI",
    use_container_width=True,
    type="primary",
):

    if not os.environ.get("GROQ_API_KEY"):

        st.error(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Cloud Secrets."
        )

    else:

        with st.spinner(
            "CrewAI is translating your instruction into an objective..."
        ):

            try:

                objective = run_planner_agent(
                    instruction
                )

                st.session_state.objective = objective
                st.session_state.q_table = None
                st.session_state.delivery_result = None
                st.session_state.explanation = None

                st.success(
                    "Mission successfully converted into an optimization objective."
                )

            except Exception as e:

                st.error(
                    f"Groq/CrewAI error: {str(e)}"
                )


# ============================================================
# OBJECTIVE
# ============================================================

if st.session_state.objective:

    objective = st.session_state.objective

    st.markdown(
        '<div class="card-title">🎯 LLM-Generated Objective</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)

    metrics = [
        (
            "⏱️ Time",
            objective["time_weight"],
        ),
        (
            "💰 Cost",
            objective["cost_weight"],
        ),
        (
            "📦 Priority",
            objective["priority_weight"],
        ),
        (
            "🛡️ Risk",
            objective["risk_weight"],
        ),
    ]

    for col, (label, value) in zip(
        cols,
        metrics
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {value:.2f}
                    </div>

                    <div class="metric-label">
                        {label} Weight
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info(
        f"🧠 **CrewAI Mission Planner:** "
        f"{objective.get('summary', 'Balanced optimization.')}"
    )


# ============================================================
# TRAINING
# ============================================================

st.markdown(
    '<div class="card-title">🧠 Reinforcement Learning</div>',
    unsafe_allow_html=True,
)

if st.button(
    "🚀 Train Q-Learning Delivery Agent",
    use_container_width=True,
):

    if not st.session_state.objective:

        st.warning(
            "First ask CrewAI to understand the mission."
        )

    else:

        objective = st.session_state.objective

        progress = st.progress(0)

        status = st.empty()

        # Train in chunks to show progress
        q_table = {}
        rewards = []

        # Full training function
        q_table, rewards = train_q_learning(
            time_weight=objective["time_weight"],
            cost_weight=objective["cost_weight"],
            priority_weight=objective["priority_weight"],
            risk_weight=objective["risk_weight"],
            traffic=traffic,
            episodes=episodes,
        )

        progress.progress(100)

        status.success(
            f"Training completed: {episodes} episodes."
        )

        st.session_state.q_table = q_table
        st.session_state.training_rewards = rewards

        st.session_state.delivery_result = None
        st.session_state.explanation = None


# ============================================================
# TRAINING GRAPH
# ============================================================

if st.session_state.training_rewards:

    st.markdown(
        "### 📈 Learning Progress"
    )

    rewards = st.session_state.training_rewards

    # Moving average
    window = 30

    moving_avg = []

    for i in range(len(rewards)):

        start = max(
            0,
            i - window + 1
        )

        values = rewards[start:i + 1]

        moving_avg.append(
            sum(values) / len(values)
        )

    chart_data = {
        "Episode Reward": rewards,
        "Moving Average": moving_avg,
    }

    st.line_chart(
        chart_data,
        height=320,
    )


# ============================================================
# DELIVERY
# ============================================================

if st.session_state.q_table:

    st.markdown(
        '<div class="card-title">🚚 Run Learned Delivery Policy</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "📦 Execute Learned Route",
        use_container_width=True,
        type="primary",
    ):

        objective = st.session_state.objective

        result = run_policy(
            q_table=st.session_state.q_table,
            time_weight=objective["time_weight"],
            cost_weight=objective["cost_weight"],
            priority_weight=objective["priority_weight"],
            risk_weight=objective["risk_weight"],
            traffic=traffic,
        )

        st.session_state.delivery_result = result
        st.session_state.explanation = None


# ============================================================
# RESULTS
# ============================================================

if st.session_state.delivery_result:

    result = st.session_state.delivery_result

    st.markdown(
        '<div class="card-title">📊 Delivery Evaluation</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)

    metrics = [
        (
            "🏆 Reward",
            f"{result['reward']:.1f}",
        ),
        (
            "⏱️ Time",
            f"{result['time']:.1f} min",
        ),
        (
            "📍 Distance",
            f"{result['distance']:.1f} km",
        ),
        (
            "💰 Cost",
            f"{result['cost']:.1f}",
        ),
        (
            "📦 Delivered",
            f"{len(result['delivered'])}/{len(PACKAGES)}",
        ),
    ]

    for col, (label, value) in zip(
        cols,
        metrics
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    st.markdown(
        "### 🗺️ Learned Route"
    )

    route_html = ""

    for index, node in enumerate(
        result["route"]
    ):

        route_html += (
            f'<span class="route-node">{node}</span>'
        )

        if index < len(result["route"]) - 1:

            route_html += (
                '<span class="route-arrow"> → </span>'
            )

    st.markdown(
        route_html,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # DECISION LOG
    # --------------------------------------------------------

    with st.expander(
        "🧠 View RL Decision Log"
    ):

        for index, decision in enumerate(
            result["decisions"],
            start=1,
        ):

            st.write(
                f"**Step {index}** | "
                f"Location: `{decision['location']}` | "
                f"Action: `{decision['action']}` | "
                f"Q-value: `{decision['q_value']:.2f}` | "
                f"Reward: `{decision['reward']:.2f}`"
            )


    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    if st.button(
        "💡 Explain Why the Agent Chose This Route",
        use_container_width=True,
    ):

        if not os.environ.get("GROQ_API_KEY"):

            st.error(
                "GROQ_API_KEY is missing."
            )

        else:

            with st.spinner(
                "Explanation Agent is analyzing the learned policy..."
            ):

                try:

                    explanation = run_explanation_agent(
                        instruction=instruction,
                        objective=st.session_state.objective,
                        result=result,
                    )

                    st.session_state.explanation = explanation

                except Exception as e:

                    st.error(
                        f"Explanation error: {str(e)}"
                    )


    if st.session_state.explanation:

        st.markdown(
            '<div class="card-title">🤖 AI Explanation</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-box">

            {st.session_state.explanation}

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# AGENT ARCHITECTURE
# ============================================================

st.divider()

st.markdown(
    "### 🤖 Agent Architecture"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="agent-card">

        <div class="agent-name">
        🤖 Mission Planner Agent
        </div>

        <div class="agent-status">
        CrewAI + Groq
        </div>

        <br>

        Converts human language into
        optimization weights.

        <br><br>

        <b>Output:</b><br>
        Time / Cost / Priority / Risk

        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        """
        <div class="agent-card">

        <div class="agent-name">
        🧠 RL Delivery Agent
        </div>

        <div class="agent-status">
        Q-Learning
        </div>

        <br>

        Learns which action produces
        higher long-term reward.

        <br><br>

        <b>Output:</b><br>
        Learned delivery route

        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        """
        <div class="agent-card">

        <div class="agent-name">
        💡 Explanation Agent
        </div>

        <div class="agent-status">
        CrewAI + Groq
        </div>

        <br>

        Explains the learned RL
        decision to humans.

        <br><br>

        <b>Output:</b><br>
        Natural-language explanation

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>

    <div style="
        text-align:center;
        color:#718596;
        font-size:13px;
        padding:20px;
    ">

    Smart Delivery Agent ·
    LLM + CrewAI + Reinforcement Learning

    <br>

    Human Instruction → Objective → Learning → Decision

    </div>
    """,
    unsafe_allow_html=True,
)
