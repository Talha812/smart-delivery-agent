import os
import json
import random
import re
from collections import defaultdict

import streamlit as st
from groq import Groq

# CrewAI is used to define the agent roles/orchestration concept.
# Actual LLM calls are made directly through the Groq SDK.
from crewai import Agent


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

st.html(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(37, 99, 235, 0.08), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(124, 58, 237, 0.08), transparent 30%),
            #f8fafc;
    }

    .main {
        padding-top: 1rem;
    }

    h1, h2, h3 {
        color: #0f172a;
    }

    p, li, label {
        color: #334155;
    }

    /* ---------- Hero ---------- */

    .hero {
        background:
            linear-gradient(
                135deg,
                #eff6ff 0%,
                #ffffff 48%,
                #f5f3ff 100%
            );
        border: 1px solid #dbeafe;
        border-radius: 24px;
        padding: 34px 38px;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.07);
    }

    .hero-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1d4ed8;
        border-radius: 999px;
        padding: 7px 14px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 42px;
        line-height: 1.08;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #475569;
        line-height: 1.6;
        max-width: 950px;
    }

    /* ---------- Pipeline ---------- */

    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 9px;
        flex-wrap: wrap;
        margin: 20px 0 30px 0;
    }

    .pipeline-item {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 12px 15px;
        font-size: 13px;
        font-weight: 700;
        color: #334155;
        box-shadow: 0 5px 15px rgba(15, 23, 42, 0.05);
    }

    .pipeline-arrow {
        color: #64748b;
        font-size: 20px;
        font-weight: 700;
    }

    /* ---------- Cards ---------- */

    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.05);
    }

    .info-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }

    .info-title {
        color: #0f172a;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .info-text {
        color: #64748b;
        font-size: 14px;
        line-height: 1.55;
    }

    /* ---------- Metrics ---------- */

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
    }

    .metric-value {
        color: #1d4ed8;
        font-size: 30px;
        font-weight: 800;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        margin-top: 3px;
    }

    /* ---------- Route ---------- */

    .route-box {
        background: linear-gradient(135deg, #eff6ff, #f5f3ff);
        border: 1px solid #c7d2fe;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        margin: 15px 0;
    }

    .route-text {
        font-size: 24px;
        font-weight: 800;
        color: #1e3a8a;
        letter-spacing: 0.5px;
    }

    /* ---------- Decision ---------- */

    .decision-card {
        background: #ffffff;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }

    .decision-title {
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 5px;
    }

    .decision-text {
        color: #64748b;
        font-size: 14px;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 30px 0 10px 0;
        font-size: 13px;
    }

    </style>
    """
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# CREWAI AGENT DEFINITIONS
# ============================================================

planner_agent = Agent(
    role="Delivery Mission Planner",
    goal=(
        "Interpret human delivery instructions and convert them into "
        "clear optimization priorities for a reinforcement learning agent."
    ),
    backstory=(
        "You are an AI logistics planner. You understand delivery "
        "deadlines, package priority, travel time, cost and risk."
    ),
    verbose=False,
    allow_delegation=False,
)

explainer_agent = Agent(
    role="RL Decision Explainer",
    goal=(
        "Explain how a Q-learning delivery agent learned and selected "
        "a route in clear language."
    ),
    backstory=(
        "You explain reinforcement learning decisions to students, "
        "developers and business users."
    ),
    verbose=False,
    allow_delegation=False,
)


# ============================================================
# GROQ DIRECT CALL
# ============================================================

def call_groq(prompt, temperature=0.2, max_tokens=1200):

    client = get_groq_client()

    if client is None:
        return None, "GROQ_API_KEY is not configured."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL_NAME,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = chat_completion.choices[0].message.content

        return content, None

    except Exception as e:
        return None, str(e)


# ============================================================
# DELIVERY MAP
# ============================================================

LOCATIONS = {
    "Warehouse": (0, 0),
    "A": (0, 1),
    "B": (0, 2),
    "C": (0, 3),
    "D": (1, 0),
    "E": (1, 1),
    "F": (1, 2),
    "G": (1, 3),
    "H": (2, 0),
    "I": (2, 1),
    "J": (2, 2),
    "K": (2, 3),
}


def build_neighbors():
    neighbors = defaultdict(list)

    for location, (r, c) in LOCATIONS.items():

        for other, (orow, ocol) in LOCATIONS.items():

            if location == other:
                continue

            distance = abs(r - orow) + abs(c - ocol)

            if distance == 1:
                neighbors[location].append(other)

    return dict(neighbors)


NEIGHBORS = build_neighbors()


# ============================================================
# PACKAGES
# ============================================================

PACKAGES = {
    "P1": {
        "destination": "C",
        "priority": "Critical",
        "deadline": 18,
        "priority_value": 4,
    },
    "P2": {
        "destination": "I",
        "priority": "High",
        "deadline": 28,
        "priority_value": 3,
    },
    "P3": {
        "destination": "K",
        "priority": "Normal",
        "deadline": 40,
        "priority_value": 1,
    },
}


PRIORITY_VALUES = {
    "Critical": 4,
    "High": 3,
    "Normal": 1,
}


# ============================================================
# LLM MISSION PLANNER
# ============================================================

def plan_mission(user_instruction):

    prompt = f"""
You are the Delivery Mission Planner.

Human instruction:

"{user_instruction}"

Convert this instruction into optimization priorities for a Q-learning
delivery robot.

Return ONLY valid JSON.

Use this exact structure:

{{
    "time_weight": 0.0,
    "cost_weight": 0.0,
    "priority_weight": 0.0,
    "risk_weight": 0.0,
    "traffic": "Low",
    "summary": "short explanation"
}}

Rules:

1. All four weights must be between 0 and 1.
2. The four weights should approximately add up to 1.
3. traffic must be one of:
   Low
   Medium
   High
4. Critical deliveries should increase priority importance.
5. Urgent/deadline language should increase time importance.
6. Cheap/economical language should increase cost importance.
7. Safe/reliable language should increase risk importance.
8. Do not include Markdown.
"""

    response, error = call_groq(
        prompt,
        temperature=0.1,
        max_tokens=700,
    )

    if error:
        return {
            "time_weight": 0.40,
            "cost_weight": 0.20,
            "priority_weight": 0.30,
            "risk_weight": 0.10,
            "traffic": "Medium",
            "summary": "Default logistics priorities were used because the LLM was unavailable.",
        }, error

    try:
        cleaned = response.strip()

        cleaned = re.sub(
            r"^```json\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"^```\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        data = json.loads(cleaned)

        time_weight = float(data.get("time_weight", 0.4))
        cost_weight = float(data.get("cost_weight", 0.2))
        priority_weight = float(data.get("priority_weight", 0.3))
        risk_weight = float(data.get("risk_weight", 0.1))

        total = (
            time_weight
            + cost_weight
            + priority_weight
            + risk_weight
        )

        if total <= 0:
            raise ValueError("Invalid weights")

        # Normalize weights.
        time_weight /= total
        cost_weight /= total
        priority_weight /= total
        risk_weight /= total

        traffic = str(
            data.get("traffic", "Medium")
        ).title()

        if traffic not in ["Low", "Medium", "High"]:
            traffic = "Medium"

        return {
            "time_weight": time_weight,
            "cost_weight": cost_weight,
            "priority_weight": priority_weight,
            "risk_weight": risk_weight,
            "traffic": traffic,
            "summary": str(
                data.get(
                    "summary",
                    "Mission priorities generated by the LLM.",
                )
            ),
        }, None

    except Exception as e:

        return {
            "time_weight": 0.40,
            "cost_weight": 0.20,
            "priority_weight": 0.30,
            "risk_weight": 0.10,
            "traffic": "Medium",
            "summary": "Could not parse the LLM response, so default weights were used.",
        }, f"LLM parsing error: {e}"


# ============================================================
# RL ENVIRONMENT
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

        self.position = "Warehouse"

        self.delivered = set()

        self.time = 0

        self.total_cost = 0

        self.total_reward = 0

        self.done = False

        return self.get_state()

    def get_state(self):

        delivered_state = tuple(
            sorted(self.delivered)
        )

        return (
            self.position,
            delivered_state,
        )

    def get_actions(self):

        return NEIGHBORS.get(
            self.position,
            [],
        )

    def get_move_time(self, destination):

        traffic_multiplier = {
            "Low": 1.0,
            "Medium": 1.25,
            "High": 1.6,
        }

        return (
            2
            * traffic_multiplier.get(
                self.traffic,
                1.25,
            )
        )

    def step(self, action):

        if self.done:

            return (
                self.get_state(),
                0,
                True,
                {},
            )

        if action not in self.get_actions():

            return (
                self.get_state(),
                -50,
                False,
                {
                    "invalid": True
                },
            )

        previous_position = self.position

        self.position = action

        move_time = self.get_move_time(action)

        move_cost = 1.0

        self.time += move_time

        self.total_cost += move_cost

        reward = 0

        # Base movement penalty.
        reward -= (
            self.time_weight
            * move_time
        )

        reward -= (
            self.cost_weight
            * move_cost
        )

        # Risk penalty.
        if self.traffic == "High":

            reward -= (
                self.risk_weight
                * 2
            )

        # Check package deliveries.
        delivered_package = None

        for package_id, package in PACKAGES.items():

            if (
                package["destination"] == self.position
                and package_id not in self.delivered
            ):

                delivered_package = package_id

                self.delivered.add(package_id)

                reward += 60

                reward += (
                    self.priority_weight
                    * package["priority_value"]
                    * 15
                )

                if self.time <= package["deadline"]:

                    reward += 30

                else:

                    reward -= 30

        # Mission complete.
        if len(self.delivered) == len(PACKAGES):

            reward += 100

            self.done = True

        # Time limit.
        if self.time > 70:

            reward -= 80

            self.done = True

        self.total_reward += reward

        info = {
            "from": previous_position,
            "to": self.position,
            "time": self.time,
            "cost": self.total_cost,
            "reward": reward,
            "delivered_package": delivered_package,
        }

        return (
            self.get_state(),
            reward,
            self.done,
            info,
        )


# ============================================================
# Q-LEARNING
# ============================================================

def choose_action(
    q_table,
    state,
    actions,
    epsilon,
):

    if not actions:
        return None

    if random.random() < epsilon:

        return random.choice(actions)

    values = [
        q_table[state][action]
        for action in actions
    ]

    maximum = max(values)

    best_actions = [
        action
        for action, value in zip(actions, values)
        if value == maximum
    ]

    return random.choice(best_actions)


def train_q_learning(
    env,
    episodes=700,
    learning_rate=0.15,
    gamma=0.90,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.992,
):

    q_table = defaultdict(
        lambda: defaultdict(float)
    )

    rewards_history = []

    epsilon_history = []

    best_reward = float("-inf")

    best_route = []

    for episode in range(episodes):

        state = env.reset()

        episode_reward = 0

        route = ["Warehouse"]

        for step in range(150):

            actions = env.get_actions()

            if not actions:
                break

            action = choose_action(
                q_table,
                state,
                actions,
                epsilon,
            )

            next_state, reward, done, info = env.step(
                action
            )

            episode_reward += reward

            next_actions = env.get_actions()

            if next_actions:

                next_max = max(
                    q_table[next_state][a]
                    for a in next_actions
                )

            else:

                next_max = 0

            old_value = q_table[state][action]

            new_value = (
                old_value
                + learning_rate
                * (
                    reward
                    + gamma * next_max
                    - old_value
                )
            )

            q_table[state][action] = new_value

            state = next_state

            route.append(action)

            if done:
                break

        rewards_history.append(
            episode_reward
        )

        epsilon_history.append(
            epsilon
        )

        if episode_reward > best_reward:

            best_reward = episode_reward

            best_route = route.copy()

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay,
        )

    return (
        q_table,
        rewards_history,
        epsilon_history,
        best_route,
    )


# ============================================================
# EXTRACT LEARNED ROUTE
# ============================================================

def get_learned_route(
    env,
    q_table,
    max_steps=100,
):

    env.reset()

    route = ["Warehouse"]

    decision_log = []

    for step in range(max_steps):

        state = env.get_state()

        actions = env.get_actions()

        if not actions:
            break

        action = choose_action(
            q_table,
            state,
            actions,
            epsilon=0,
        )

        before = env.position

        next_state, reward, done, info = env.step(
            action
        )

        decision_log.append(
            {
                "Step": step + 1,
                "From": before,
                "Action": action,
                "Reward": round(reward, 2),
                "Time": round(env.time, 2),
                "Cost": round(env.total_cost, 2),
                "Delivered": info.get(
                    "delivered_package"
                )
                or "-",
            }
        )

        route.append(action)

        if done:
            break

    return (
        route,
        decision_log,
        env,
    )


# ============================================================
# RL EXPLANATION
# ============================================================

def explain_result(
    mission,
    plan,
    route,
    env,
    decision_log,
):

    delivered = ", ".join(
        sorted(env.delivered)
    )

    prompt = f"""
You are the RL Decision Explainer.

Explain the result of a Smart Delivery Agent demonstration.

Human mission:
{mission}

LLM-generated objective:
{json.dumps(plan, indent=2)}

Learned route:
{" → ".join(route)}

Final time:
{env.time:.2f}

Final cost:
{env.total_cost:.2f}

Total reward:
{env.total_reward:.2f}

Packages delivered:
{delivered}

Decision steps:
{json.dumps(decision_log[:12], indent=2)}

Important architecture:

Human Instruction
→ LLM
→ Objective / Weights
→ Q-Learning Agent
→ Delivery Environment
→ Learned Route
→ Evaluation
→ Explanation

Explain in simple language.

Make it clear that:

1. The LLM does NOT directly choose every movement.
2. The LLM interprets the human objective.
3. Q-learning learns action values from environment rewards.
4. The learned Q-table determines the final route.
5. Reward represents the optimization objective.
6. Training and final route selection are different stages.

Keep the explanation under 300 words.
"""

    response, error = call_groq(
        prompt,
        temperature=0.3,
        max_tokens=1000,
    )

    if error:

        return (
            "The explanation model could not be reached.\n\n"
            f"Route: {' → '.join(route)}\n\n"
            f"Total reward: {env.total_reward:.2f}\n"
            f"Time: {env.time:.2f}\n"
            f"Cost: {env.total_cost:.2f}"
        )

    return response


# ============================================================
# SESSION STATE
# ============================================================

if "trained" not in st.session_state:

    st.session_state.trained = False

if "q_table" not in st.session_state:

    st.session_state.q_table = None

if "training_rewards" not in st.session_state:

    st.session_state.training_rewards = []

if "route" not in st.session_state:

    st.session_state.route = []

if "decision_log" not in st.session_state:

    st.session_state.decision_log = []

if "environment" not in st.session_state:

    st.session_state.environment = None

if "plan" not in st.session_state:

    st.session_state.plan = None

if "explanation" not in st.session_state:

    st.session_state.explanation = ""


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-badge">
            🤖 LLM + REINFORCEMENT LEARNING
        </div>

        <div class="hero-title">
            Smart Delivery Agent
        </div>

        <div class="hero-subtitle">
            A live demonstration of how a Large Language Model can
            translate human instructions into an optimization objective,
            while a Q-learning agent learns the actual delivery policy
            through interaction with an environment.
        </div>

    </div>
    """
)


# ============================================================
# PIPELINE
# ============================================================

st.html(
    """
    <div class="pipeline">

        <div class="pipeline-item">
            👤 Human Instruction
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-item">
            🧠 LLM
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-item">
            🎯 Objective
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-item">
            🎓 Q-Learning
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-item">
            🌍 Environment
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-item">
            🗺️ Learned Route
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-item">
            📊 Evaluation
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-item">
            💬 Explanation
        </div>

    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Mission Control")

    st.caption(
        "Configure the reinforcement learning experiment."
    )

    episodes = st.slider(
        "Training Episodes",
        min_value=100,
        max_value=1500,
        value=700,
        step=100,
    )

    learning_rate = st.slider(
        "Learning Rate",
        min_value=0.05,
        max_value=0.50,
        value=0.15,
        step=0.05,
    )

    gamma = st.slider(
        "Discount Factor",
        min_value=0.50,
        max_value=0.99,
        value=0.90,
        step=0.01,
    )

    epsilon_decay = st.slider(
        "Exploration Decay",
        min_value=0.95,
        max_value=0.999,
        value=0.992,
        step=0.001,
    )

    st.divider()

    st.subheader("📦 Packages")

    for package_id, package in PACKAGES.items():

        st.write(
            f"**{package_id}** → {package['destination']}  \n"
            f"{package['priority']} priority · "
            f"Deadline {package['deadline']}"
        )

    st.divider()

    st.caption(
        "LLM: Groq / GPT-OSS 120B"
    )

    st.caption(
        "RL: Q-Learning"
    )

    st.caption(
        "Agent Layer: CrewAI"
    )


# ============================================================
# MAIN INPUT
# ============================================================

st.subheader("1️⃣ Give the Agent a Human Instruction")

default_instruction = (
    "Deliver all packages as quickly as possible. "
    "P1 is critical and must reach its destination before the deadline. "
    "Avoid unnecessary travel cost and consider traffic risk."
)

user_instruction = st.text_area(
    "Mission instruction",
    value=default_instruction,
    height=110,
)


# ============================================================
# INFORMATION CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.html(
        """
        <div class="info-card">

            <div class="info-icon">🧠</div>

            <div class="info-title">
                LLM Planner
            </div>

            <div class="info-text">
                Understands the human instruction and converts
                natural language into optimization weights.
            </div>

        </div>
        """
    )


with col2:

    st.html(
        """
        <div class="info-card">

            <div class="info-icon">🎓</div>

            <div class="info-title">
                Q-Learning Agent
            </div>

            <div class="info-text">
                Learns which movement actions produce higher
                long-term rewards through repeated interaction.
            </div>

        </div>
        """
    )


with col3:

    st.html(
        """
        <div class="info-card">

            <div class="info-icon">🌍</div>

            <div class="info-title">
                Delivery Environment
            </div>

            <div class="info-text">
                Simulates locations, traffic, time, cost,
                package priorities and deadlines.
            </div>

        </div>
        """
    )


st.write("")


# ============================================================
# RUN BUTTON
# ============================================================

run_button = st.button(
    "🚀 Understand Mission & Train Delivery Agent",
    type="primary",
    use_container_width=True,
)


# ============================================================
# EXECUTION
# ============================================================

if run_button:

    if not user_instruction.strip():

        st.warning(
            "Please provide a delivery instruction."
        )

        st.stop()

    # --------------------------------------------------------
    # Step 1: LLM
    # --------------------------------------------------------

    with st.status(
        "🧠 LLM is interpreting the human instruction...",
        expanded=True,
    ) as status:

        plan, planner_error = plan_mission(
            user_instruction
        )

        st.write(
            "Mission priorities generated."
        )

        if planner_error:

            st.warning(
                planner_error
            )

        status.update(
            label="✅ Mission interpreted",
            state="complete",
        )

    # --------------------------------------------------------
    # Show objective
    # --------------------------------------------------------

    st.subheader("2️⃣ LLM → Optimization Objective")

    objective_cols = st.columns(4)

    with objective_cols[0]:

        st.metric(
            "⏱️ Time",
            f"{plan['time_weight']:.2f}",
        )

    with objective_cols[1]:

        st.metric(
            "💰 Cost",
            f"{plan['cost_weight']:.2f}",
        )

    with objective_cols[2]:

        st.metric(
            "⭐ Priority",
            f"{plan['priority_weight']:.2f}",
        )

    with objective_cols[3]:

        st.metric(
            "🛡️ Risk",
            f"{plan['risk_weight']:.2f}",
        )

    st.info(
        f"**LLM interpretation:** {plan['summary']}  \n"
        f"**Traffic assumption:** {plan['traffic']}"
    )

    # --------------------------------------------------------
    # Step 2: RL Environment
    # --------------------------------------------------------

    environment = DeliveryEnvironment(
        time_weight=plan["time_weight"],
        cost_weight=plan["cost_weight"],
        priority_weight=plan["priority_weight"],
        risk_weight=plan["risk_weight"],
        traffic=plan["traffic"],
    )

    # --------------------------------------------------------
    # Step 3: Q-Learning
    # --------------------------------------------------------

    with st.status(
        "🎓 Q-learning agent is training...",
        expanded=True,
    ) as status:

        (
            q_table,
            rewards_history,
            epsilon_history,
            best_training_route,
        ) = train_q_learning(
            environment,
            episodes=episodes,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
        )

        st.write(
            f"Completed {episodes:,} training episodes."
        )

        status.update(
            label="✅ Q-learning training completed",
            state="complete",
        )

    # --------------------------------------------------------
    # Step 4: Learned route
    # --------------------------------------------------------

    (
        route,
        decision_log,
        final_environment,
    ) = get_learned_route(
        environment,
        q_table,
    )

    # --------------------------------------------------------
    # Step 5: Explanation
    # --------------------------------------------------------

    with st.status(
        "💬 Generating decision explanation...",
        expanded=True,
    ) as status:

        explanation = explain_result(
            user_instruction,
            plan,
            route,
            final_environment,
            decision_log,
        )

        status.update(
            label="✅ Explanation generated",
            state="complete",
        )

    # --------------------------------------------------------
    # Save state
    # --------------------------------------------------------

    st.session_state.trained = True

    st.session_state.q_table = q_table

    st.session_state.training_rewards = (
        rewards_history
    )

    st.session_state.route = route

    st.session_state.decision_log = (
        decision_log
    )

    st.session_state.environment = (
        final_environment
    )

    st.session_state.plan = plan

    st.session_state.explanation = (
        explanation
    )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.trained:

    plan = st.session_state.plan

    environment = st.session_state.environment

    route = st.session_state.route

    decision_log = st.session_state.decision_log

    rewards_history = (
        st.session_state.training_rewards
    )

    # ========================================================
    # TRAINING RESULTS
    # ========================================================

    st.divider()

    st.subheader(
        "3️⃣ Q-Learning Training"
    )

    chart_data = {
        "Episode Reward": rewards_history
    }

    st.line_chart(
        chart_data,
        height=280,
    )

    if rewards_history:

        recent_count = min(
            50,
            len(rewards_history),
        )

        recent_average = sum(
            rewards_history[-recent_count:]
        ) / recent_count

        st.caption(
            f"Average reward over the last "
            f"{recent_count} episodes: "
            f"**{recent_average:.2f}**"
        )

    # ========================================================
    # LEARNED ROUTE
    # ========================================================

    st.subheader(
        "4️⃣ Learned Delivery Route"
    )

    route_string = "  →  ".join(
        route
    )

    st.html(
        f"""
        <div class="route-box">

            <div class="route-text">
                {route_string}
            </div>

        </div>
        """
    )

    # ========================================================
    # METRICS
    # ========================================================

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {len(route) - 1}
                </div>

                <div class="metric-label">
                    Movement Steps
                </div>

            </div>
            """
        )

    with metric2:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {environment.time:.1f}
                </div>

                <div class="metric-label">
                    Total Time
                </div>

            </div>
            """
        )

    with metric3:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {environment.total_cost:.1f}
                </div>

                <div class="metric-label">
                    Travel Cost
                </div>

            </div>
            """
        )

    with metric4:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {environment.total_reward:.1f}
                </div>

                <div class="metric-label">
                    Total Reward
                </div>

            </div>
            """
        )

    # ========================================================
    # DELIVERY STATUS
    # ========================================================

    st.subheader(
        "5️⃣ Delivery Evaluation"
    )

    evaluation_cols = st.columns(
        len(PACKAGES)
    )

    for column, (package_id, package) in zip(
        evaluation_cols,
        PACKAGES.items(),
    ):

        delivered = (
            package_id
            in environment.delivered
        )

        with column:

            if delivered:

                st.success(
                    f"✅ {package_id}\n\n"
                    f"Destination: {package['destination']}\n\n"
                    f"Priority: {package['priority']}"
                )

            else:

                st.error(
                    f"❌ {package_id}\n\n"
                    f"Destination: {package['destination']}\n\n"
                    f"Priority: {package['priority']}"
                )

    # ========================================================
    # DECISION LOG
    # ========================================================

    st.subheader(
        "6️⃣ Agent Decision Log"
    )

    if decision_log:

        st.dataframe(
            decision_log,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # Q TABLE
    # ========================================================

    st.subheader(
        "7️⃣ Learned Q-Table"
    )

    q_table = st.session_state.q_table

    q_rows = []

    for state, actions in q_table.items():

        position = state[0]

        delivered_state = (
            ", ".join(state[1])
            if state[1]
            else "-"
        )

        for action, value in actions.items():

            q_rows.append(
                {
                    "Position": position,
                    "Delivered": delivered_state,
                    "Action": action,
                    "Q-Value": round(
                        value,
                        3,
                    ),
                }
            )

    if q_rows:

        q_rows = sorted(
            q_rows,
            key=lambda x: x["Q-Value"],
            reverse=True,
        )

        st.dataframe(
            q_rows[:80],
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # EXPLANATION
    # ========================================================

    st.subheader(
        "8️⃣ AI Explanation"
    )

    st.info(
        st.session_state.explanation
    )

    # ========================================================
    # ARCHITECTURE
    # ========================================================

    st.subheader(
        "9️⃣ System Architecture"
    )

    architecture_cols = st.columns(5)

    architecture = [
        (
            "👤",
            "Human",
            "Provides the natural-language delivery objective.",
        ),
        (
            "🧠",
            "Groq LLM",
            "Interprets the instruction and creates optimization weights.",
        ),
        (
            "🎓",
            "Q-Learning",
            "Learns which actions produce higher long-term rewards.",
        ),
        (
            "🌍",
            "Environment",
            "Simulates movement, traffic, time, cost and deliveries.",
        ),
        (
            "📊",
            "Evaluation",
            "Measures route quality and explains the learned behavior.",
        ),
    ]

    for column, item in zip(
        architecture_cols,
        architecture,
    ):

        icon, title, description = item

        with column:

            st.html(
                f"""
                <div class="info-card">

                    <div class="info-icon">
                        {icon}
                    </div>

                    <div class="info-title">
                        {title}
                    </div>

                    <div class="info-text">
                        {description}
                    </div>

                </div>
                """
            )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        Smart Delivery Agent ·
        Groq + CrewAI + Q-Learning + Streamlit

        <br>

        <b>
        Build with AI, not just use AI.
        </b>

    </div>
    """
)
