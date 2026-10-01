import os
import json
import random
import math

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

st.html(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #f8fbff 0%,
                #eef6ff 50%,
                #ffffff 100%
            );
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background:
            linear-gradient(
                135deg,
                #ffffff 0%,
                #edf6ff 100%
            );

        border: 1px solid #d8e7f5;
        border-radius: 24px;

        padding: 30px 35px;
        margin-bottom: 25px;

        box-shadow:
            0 10px 35px rgba(35, 90, 140, 0.08);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #12344d;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #557086;
        line-height: 1.6;
    }


    /* ========================================================
       PIPELINE
       ======================================================== */

    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;

        gap: 10px;
        flex-wrap: wrap;

        margin: 20px 0 30px 0;
    }

    .pipeline-item {
        background: #ffffff;

        border: 1px solid #d9e8f5;
        border-radius: 14px;

        padding: 12px 17px;

        color: #23445b;
        font-weight: 700;

        box-shadow:
            0 4px 12px rgba(40, 80, 120, 0.06);
    }

    .arrow {
        color: #4b8dcc;
        font-size: 22px;
        font-weight: bold;
    }


    /* ========================================================
       CARDS
       ======================================================== */

    .card {
        background: #ffffff;

        border: 1px solid #dceaf5;
        border-radius: 18px;

        padding: 20px;
        margin-bottom: 15px;

        box-shadow:
            0 6px 20px rgba(30, 80, 120, 0.06);
    }

    .card-title {
        font-size: 20px;
        font-weight: 800;

        color: #173f59;

        margin-bottom: 12px;
    }


    /* ========================================================
       AGENT CARDS
       ======================================================== */

    .agent-card {
        background:
            linear-gradient(
                135deg,
                #ffffff,
                #f5faff
            );

        border: 1px solid #d5e7f5;
        border-radius: 16px;

        padding: 18px;

        margin-bottom: 10px;

        min-height: 210px;

        box-shadow:
            0 5px 18px rgba(30, 80, 120, 0.05);
    }

    .agent-name {
        font-weight: 800;
        color: #174766;
        font-size: 17px;
    }

    .agent-status {
        color: #5e7585;
        font-size: 13px;
        margin-top: 4px;
    }

    .agent-description {
        color: #526b7b;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 14px;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {
        background: #ffffff;

        border: 1px solid #dceaf5;
        border-radius: 16px;

        padding: 18px;

        text-align: center;

        box-shadow:
            0 5px 18px rgba(30, 80, 120, 0.05);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #1769aa;
    }

    .metric-label {
        color: #657b8b;
        font-size: 13px;
        margin-top: 5px;
    }


    /* ========================================================
       ROUTE
       ======================================================== */

    .route-container {
        background: #f8fbff;

        border: 1px solid #dceaf5;
        border-radius: 18px;

        padding: 20px;

        text-align: center;

        margin: 10px 0 20px 0;
    }

    .route-node {
        display: inline-block;

        background: #ffffff;

        border: 2px solid #6aa9d8;
        border-radius: 12px;

        padding: 10px 14px;
        margin: 4px;

        color: #19435d;
        font-weight: 700;

        box-shadow:
            0 3px 8px rgba(50, 100, 150, 0.05);
    }

    .route-arrow {
        color: #5194c9;
        font-size: 20px;
        font-weight: bold;
    }


    /* ========================================================
       INFO BOXES
       ======================================================== */

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
        padding: 17px;

        line-height: 1.7;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 12px;

        font-weight: 700;

        min-height: 45px;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #f8fbfe;

        border-right: 1px solid #dceaf5;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;

        color: #718596;

        font-size: 13px;

        padding: 25px;
    }

    </style>
    """
)


# ============================================================
# DELIVERY MAP
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


# ============================================================
# PACKAGES
# ============================================================

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
# HELPER FUNCTIONS
# ============================================================

def distance(location_a, location_b):

    x1, y1 = LOCATIONS[location_a]
    x2, y2 = LOCATIONS[location_b]

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def traffic_factor(level):

    factors = {
        "Low": 1.0,
        "Medium": 1.25,
        "High": 1.60,
    }

    return factors[level]


def priority_value(priority):

    values = {
        "Critical": 3.0,
        "High": 2.0,
        "Normal": 1.0,
    }

    return values[priority]


# ============================================================
# DELIVERY ENVIRONMENT
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

        self.time = 0.0
        self.cost = 0.0
        self.distance_travelled = 0.0

        self.delivered = []

        return self.get_state()


    def get_state(self):

        remaining = tuple(
            package_id
            for package_id in PACKAGES
            if package_id not in self.delivered
        )

        return (
            self.location,
            remaining,
        )


    def available_actions(self):

        actions = list(
            NEIGHBORS[self.location]
        )

        for package_id, package in PACKAGES.items():

            if (
                package["destination"]
                == self.location
                and package_id
                not in self.delivered
            ):

                actions.append(
                    f"DELIVER_{package_id}"
                )

        return actions


    def step(self, action):

        reward = 0.0
        done = False


        # ----------------------------------------------------
        # DELIVERY ACTION
        # ----------------------------------------------------

        if action.startswith("DELIVER_"):

            package_id = action.replace(
                "DELIVER_",
                ""
            )

            if package_id in self.delivered:

                reward -= 10


            elif (
                PACKAGES[package_id]["destination"]
                != self.location
            ):

                reward -= 20


            else:

                package = PACKAGES[package_id]

                self.delivered.append(
                    package_id
                )

                priority_bonus = (
                    priority_value(
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


        # ----------------------------------------------------
        # MOVEMENT ACTION
        # ----------------------------------------------------

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

                travel_time = (
                    step_distance * factor
                )

                travel_cost = (
                    step_distance * 2
                )

                self.location = action

                self.time += travel_time

                self.cost += travel_cost

                self.distance_travelled += (
                    step_distance
                )


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


                # Traffic penalty
                if self.traffic == "High":

                    reward -= (
                        2
                        * self.risk_weight
                    )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if len(self.delivered) == len(PACKAGES):

            reward += 100

            done = True


        # ----------------------------------------------------
        # TIME LIMIT
        # ----------------------------------------------------

        if self.time > 70:

            reward -= 80

            done = True


        return (
            self.get_state(),
            reward,
            done,
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


    # Exploration
    if random.random() < epsilon:

        return random.choice(
            actions
        )


    # Exploitation
    values = [
        q_table.get(
            (state, action),
            0.0
        )
        for action in actions
    ]


    max_value = max(values)


    best_actions = [
        action

        for action, value
        in zip(actions, values)

        if value == max_value
    ]


    return random.choice(
        best_actions
    )


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

    discount_factor = 0.90

    epsilon = 1.0

    minimum_epsilon = 0.05

    epsilon_decay = 0.992

    episode_rewards = []


    for episode in range(episodes):

        environment = DeliveryEnvironment(
            time_weight=time_weight,
            cost_weight=cost_weight,
            priority_weight=priority_weight,
            risk_weight=risk_weight,
            traffic=traffic,
        )

        state = environment.reset()

        total_reward = 0.0


        for step in range(150):

            actions = (
                environment.available_actions()
            )

            if not actions:

                break


            action = choose_action(
                q_table,
                state,
                actions,
                epsilon,
            )


            next_state, reward, done = (
                environment.step(action)
            )


            current_q = q_table.get(
                (state, action),
                0.0
            )


            next_actions = (
                environment.available_actions()
            )


            if next_actions:

                max_next_q = max(
                    q_table.get(
                        (
                            next_state,
                            next_action
                        ),
                        0.0
                    )

                    for next_action
                    in next_actions
                )

            else:

                max_next_q = 0.0


            new_q = (
                current_q
                +
                learning_rate
                *
                (
                    reward
                    +
                    discount_factor
                    * max_next_q
                    -
                    current_q
                )
            )


            q_table[
                (state, action)
            ] = new_q


            state = next_state

            total_reward += reward


            if done:

                break


        epsilon = max(
            minimum_epsilon,
            epsilon * epsilon_decay
        )


        episode_rewards.append(
            total_reward
        )


    return (
        q_table,
        episode_rewards,
    )


# ============================================================
# EXECUTE LEARNED POLICY
# ============================================================

def run_policy(
    q_table,
    time_weight,
    cost_weight,
    priority_weight,
    risk_weight,
    traffic,
):

    environment = DeliveryEnvironment(
        time_weight=time_weight,
        cost_weight=cost_weight,
        priority_weight=priority_weight,
        risk_weight=risk_weight,
        traffic=traffic,
    )

    state = environment.reset()

    route = [
        "Warehouse"
    ]

    decisions = []

    total_reward = 0.0


    for step in range(100):

        actions = (
            environment.available_actions()
        )

        if not actions:

            break


        action = choose_action(
            q_table,
            state,
            actions,
            epsilon=0.0,
        )


        q_value = q_table.get(
            (state, action),
            0.0
        )


        previous_location = (
            environment.location
        )


        next_state, reward, done = (
            environment.step(action)
        )


        total_reward += reward


        decisions.append(
            {
                "step": step + 1,
                "location": previous_location,
                "action": action,
                "q_value": q_value,
                "reward": reward,
            }
        )


        if not action.startswith(
            "DELIVER_"
        ):

            if (
                environment.location
                != route[-1]
            ):

                route.append(
                    environment.location
                )

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

        "time": environment.time,

        "cost": environment.cost,

        "distance": (
            environment.distance_travelled
        ),

        "delivered": environment.delivered,

        "q_table": q_table,
    }


# ============================================================
# CREWAI / GROQ
# ============================================================

def get_crewai_llm():

    return LLM(
        model="groq/openai/gpt-oss-120b",
        temperature=0.2,
        max_tokens=1000,
    )


# ============================================================
# MISSION PLANNER AGENT
# ============================================================

def run_planner_agent(
    user_instruction
):

    llm = get_crewai_llm()


    planner = Agent(
        role="Delivery Mission Planner",

        goal=(
            "Translate a human delivery instruction "
            "into clear optimization priorities."
        ),

        backstory=(
            "You are a logistics planning agent. "
            "You understand natural language delivery "
            "requirements and convert them into numerical "
            "optimization priorities for a reinforcement "
            "learning system."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )


    task = Task(

        description=f"""

        Analyze this delivery instruction:

        "{user_instruction}"


        Return ONLY valid JSON:

        {{
            "time_weight": number,
            "cost_weight": number,
            "priority_weight": number,
            "risk_weight": number,
            "summary": "short explanation"
        }}


        Requirements:

        - Every weight must be between 0 and 1.
        - The four weights should add up to approximately 1.
        - Speed/emergency → higher time_weight.
        - Low cost → higher cost_weight.
        - Urgent packages → higher priority_weight.
        - Safety/reliability → higher risk_weight.

        Do not include Markdown.
        Do not include code fences.

        """,

        expected_output=(
            "A valid JSON object."
        ),

        agent=planner,
    )


    crew = Crew(
        agents=[
            planner
        ],

        tasks=[
            task
        ],

        process=Process.sequential,

        verbose=False,
    )


    result = crew.kickoff()

    text = str(result)


    try:

        start = text.find("{")

        end = text.rfind("}") + 1

        json_text = text[
            start:end
        ]

        data = json.loads(
            json_text
        )

    except Exception:

        data = {
            "time_weight": 0.40,
            "cost_weight": 0.20,
            "priority_weight": 0.30,
            "risk_weight": 0.10,
            "summary": (
                "Balanced delivery optimization."
            ),
        }


    # --------------------------------------------------------
    # Normalize weights
    # --------------------------------------------------------

    weights = [

        float(
            data.get(
                "time_weight",
                0.40
            )
        ),

        float(
            data.get(
                "cost_weight",
                0.20
            )
        ),

        float(
            data.get(
                "priority_weight",
                0.30
            )
        ),

        float(
            data.get(
                "risk_weight",
                0.10
            )
        ),
    ]


    # Prevent negative values
    weights = [
        max(0.0, value)
        for value in weights
    ]


    total = sum(weights)


    if total <= 0:

        weights = [
            0.40,
            0.20,
            0.30,
            0.10,
        ]

        total = 1.0


    weights = [
        value / total
        for value in weights
    ]


    data["time_weight"] = weights[0]

    data["cost_weight"] = weights[1]

    data["priority_weight"] = weights[2]

    data["risk_weight"] = weights[3]


    return data


# ============================================================
# EXPLANATION AGENT
# ============================================================

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
            "decisions clearly and accurately."
        ),

        backstory=(
            "You are an AI logistics analyst. "
            "You explain the relationship between the "
            "human objective, reward function, Q-learning "
            "policy and final delivery result."
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

        {json.dumps(
            objective,
            indent=2
        )}


        RL Route:

        {result["route"]}


        Total Reward:

        {result["reward"]:.2f}


        Delivery Time:

        {result["time"]:.2f} minutes


        Distance:

        {result["distance"]:.2f} km


        Cost:

        {result["cost"]:.2f}


        Delivered Packages:

        {result["delivered"]}


        Explain in 2-4 concise paragraphs:

        1. What the human wanted.
        2. How the objective influenced the reward.
        3. What the Q-learning agent learned.
        4. Why the final route was selected.
        5. What the final result means.


        IMPORTANT:

        The Q-learning agent selected the route.

        Do not claim that the LLM selected the route.

        The LLM only helped define the objective
        and explain the result.

        """,

        expected_output=(
            "A concise explanation for a human audience."
        ),

        agent=explainer,
    )


    crew = Crew(

        agents=[
            explainer
        ],

        tasks=[
            task
        ],

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
# HERO
# ============================================================

st.html(
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

        <div class="arrow">
            →
        </div>

        <div class="pipeline-item">
            🤖 CrewAI + Groq
        </div>

        <div class="arrow">
            →
        </div>

        <div class="pipeline-item">
            🎯 Objective
        </div>

        <div class="arrow">
            →
        </div>

        <div class="pipeline-item">
            🧠 Q-Learning
        </div>

        <div class="arrow">
            →
        </div>

        <div class="pipeline-item">
            🌍 Environment
        </div>

        <div class="arrow">
            →
        </div>

        <div class="pipeline-item">
            📦 Delivery
        </div>

    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🎛️ Mission Control"
    )

    st.caption(
        "Configure the delivery environment."
    )


    traffic = st.selectbox(
        "🚦 Traffic Level",

        [
            "Low",
            "Medium",
            "High",
        ],

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


    st.markdown(
        "### 📦 Delivery Packages"
    )


    for package_id, package in PACKAGES.items():

        st.write(
            f"**{package_id}** → "
            f"{package['destination']} "
            f"· {package['priority']} "
            f"· Deadline {package['deadline']} min"
        )


    st.divider()


    st.markdown(
        "### 🧠 RL Parameters"
    )


    st.write(
        "Learning Rate: `0.15`"
    )

    st.write(
        "Discount Factor: `0.90`"
    )

    st.write(
        "Exploration: `ε-greedy`"
    )

    st.write(
        "Algorithm: `Q-Learning`"
    )


    st.divider()


    st.caption(
        "LLM: Groq · Model: GPT-OSS-120B"
    )


# ============================================================
# HUMAN MISSION
# ============================================================

st.html(
    """
    <div class="card-title">
        👤 Human Mission
    </div>
    """
)


instruction = st.text_area(

    "Tell the delivery agent what you want:",

    value=(
        "Deliver the packages as quickly as possible. "
        "The critical package should receive the highest "
        "priority, but keep the delivery cost reasonable."
    ),

    height=110,

    label_visibility="collapsed",
)


if st.button(
    "🤖 Understand Mission with CrewAI",
    use_container_width=True,
    type="primary",
):

    if not os.environ.get(
        "GROQ_API_KEY"
    ):

        st.error(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Cloud Secrets."
        )

    else:

        with st.spinner(
            "CrewAI Mission Planner is analyzing your instruction..."
        ):

            try:

                objective = (
                    run_planner_agent(
                        instruction
                    )
                )


                st.session_state.objective = (
                    objective
                )


                st.session_state.q_table = None

                st.session_state.training_rewards = []

                st.session_state.delivery_result = None

                st.session_state.explanation = None


                st.success(
                    "Mission converted into an optimization objective."
                )


            except Exception as error:

                st.error(
                    f"CrewAI / Groq error: {error}"
                )


# ============================================================
# OBJECTIVE
# ============================================================

if st.session_state.objective:

    objective = (
        st.session_state.objective
    )


    st.html(
        """
        <div class="card-title">
            🎯 LLM-Generated Objective
        </div>
        """
    )


    columns = st.columns(4)


    metrics = [

        (
            "⏱️ Time",
            objective["time_weight"]
        ),

        (
            "💰 Cost",
            objective["cost_weight"]
        ),

        (
            "📦 Priority",
            objective["priority_weight"]
        ),

        (
            "🛡️ Risk",
            objective["risk_weight"]
        ),

    ]


    for column, (
        label,
        value
    ) in zip(
        columns,
        metrics
    ):

        with column:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {value:.2f}
                    </div>

                    <div class="metric-label">
                        {label} Weight
                    </div>

                </div>
                """
            )


    st.info(
        "🤖 **Mission Planner:** "
        + objective.get(
            "summary",
            "Balanced delivery optimization."
        )
    )


# ============================================================
# TRAINING
# ============================================================

st.html(
    """
    <div class="card-title">
        🧠 Reinforcement Learning
    </div>
    """
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

        objective = (
            st.session_state.objective
        )


        progress = st.progress(0)

        status = st.empty()


        with st.spinner(
            "Q-learning agent is exploring delivery strategies..."
        ):

            q_table, rewards = (
                train_q_learning(

                    time_weight=(
                        objective[
                            "time_weight"
                        ]
                    ),

                    cost_weight=(
                        objective[
                            "cost_weight"
                        ]
                    ),

                    priority_weight=(
                        objective[
                            "priority_weight"
                        ]
                    ),

                    risk_weight=(
                        objective[
                            "risk_weight"
                        ]
                    ),

                    traffic=traffic,

                    episodes=episodes,
                )
            )


        progress.progress(100)


        status.success(
            f"Training completed: {episodes} episodes."
        )


        st.session_state.q_table = (
            q_table
        )

        st.session_state.training_rewards = (
            rewards
        )

        st.session_state.delivery_result = None

        st.session_state.explanation = None


# ============================================================
# LEARNING GRAPH
# ============================================================

if st.session_state.training_rewards:

    st.markdown(
        "### 📈 Learning Progress"
    )


    rewards = (
        st.session_state.training_rewards
    )


    window = 30

    moving_average = []


    for index in range(
        len(rewards)
    ):

        start = max(
            0,
            index - window + 1
        )


        values = rewards[
            start:index + 1
        ]


        moving_average.append(
            sum(values)
            / len(values)
        )


    chart_data = {

        "Episode Reward":
            rewards,

        "Moving Average":
            moving_average,
    }


    st.line_chart(
        chart_data,
        height=320,
    )


    if len(rewards) >= 100:

        first_average = (
            sum(
                rewards[:50]
            )
            / 50
        )


        last_average = (
            sum(
                rewards[-50:]
            )
            / 50
        )


        change = (
            last_average
            - first_average
        )


        if change > 0:

            st.success(
                f"📈 The average reward improved by "
                f"{change:.1f} points between the early "
                f"and late training stages."
            )


# ============================================================
# EXECUTE DELIVERY
# ============================================================

if st.session_state.q_table:

    st.html(
        """
        <div class="card-title">
            🚚 Execute Learned Delivery Policy
        </div>
        """
    )


    if st.button(
        "📦 Execute Learned Route",
        use_container_width=True,
        type="primary",
    ):

        objective = (
            st.session_state.objective
        )


        result = run_policy(

            q_table=(
                st.session_state.q_table
            ),

            time_weight=(
                objective[
                    "time_weight"
                ]
            ),

            cost_weight=(
                objective[
                    "cost_weight"
                ]
            ),

            priority_weight=(
                objective[
                    "priority_weight"
                ]
            ),

            risk_weight=(
                objective[
                    "risk_weight"
                ]
            ),

            traffic=traffic,
        )


        st.session_state.delivery_result = (
            result
        )


        st.session_state.explanation = None


# ============================================================
# RESULTS
# ============================================================

if st.session_state.delivery_result:

    result = (
        st.session_state.delivery_result
    )


    st.html(
        """
        <div class="card-title">
            📊 Delivery Evaluation
        </div>
        """
    )


    columns = st.columns(5)


    metrics = [

        (
            "🏆 Reward",
            f"{result['reward']:.1f}"
        ),

        (
            "⏱️ Time",
            f"{result['time']:.1f} min"
        ),

        (
            "📍 Distance",
            f"{result['distance']:.1f} km"
        ),

        (
            "💰 Cost",
            f"{result['cost']:.1f}"
        ),

        (
            "📦 Delivered",
            f"{len(result['delivered'])}/"
            f"{len(PACKAGES)}"
        ),

    ]


    for column, (
        label,
        value
    ) in zip(
        columns,
        metrics
    ):

        with column:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-label">
                        {label}
                    </div>

                </div>
                """
            )


    # ========================================================
    # ROUTE
    # ========================================================

    st.markdown(
        "### 🗺️ Learned Route"
    )


    route_html = (
        '<div class="route-container">'
    )


    for index, node in enumerate(
        result["route"]
    ):

        route_html += (
            f'<span class="route-node">'
            f'{node}'
            f'</span>'
        )


        if index < (
            len(result["route"]) - 1
        ):

            route_html += (
                '<span class="route-arrow">'
                '→'
                '</span>'
            )


    route_html += (
        "</div>"
    )


    st.html(
        route_html
    )


    # ========================================================
    # DELIVERY STATUS
    # ========================================================

    if (
        len(result["delivered"])
        == len(PACKAGES)
    ):

        st.html(
            """
            <div class="success-box">
                <strong>✅ Mission Completed</strong>
                <br>
                The learned policy successfully
                delivered all packages.
            </div>
            """
        )

    else:

        st.warning(
            "The learned policy did not deliver "
            "all packages within the simulation limit."
        )


    # ========================================================
    # DECISION LOG
    # ========================================================

    with st.expander(
        "🧠 View RL Decision Log"
    ):

        for decision in (
            result["decisions"]
        ):

            st.write(

                f"**Step {decision['step']}**  ·  "

                f"Location: `{decision['location']}`  ·  "

                f"Action: `{decision['action']}`  ·  "

                f"Q-value: `{decision['q_value']:.2f}`  ·  "

                f"Reward: `{decision['reward']:.2f}`"

            )


    # ========================================================
    # Q TABLE
    # ========================================================

    with st.expander(
        "🔍 Inspect Learned Q-Values"
    ):

        q_rows = []


        for (
            state,
            action
        ), value in result["q_table"].items():

            location = state[0]

            remaining = (
                ", ".join(
                    state[1]
                )
                if state[1]
                else "None"
            )


            q_rows.append(
                {
                    "Location": location,

                    "Remaining Packages":
                        remaining,

                    "Action":
                        action,

                    "Q-Value":
                        round(
                            value,
                            3
                        ),
                }
            )


        if q_rows:

            st.dataframe(
                q_rows,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No Q-values available."
            )


    # ========================================================
    # EXPLANATION
    # ========================================================

    if st.button(
        "💡 Explain Why the Agent Chose This Route",
        use_container_width=True,
    ):

        if not os.environ.get(
            "GROQ_API_KEY"
        ):

            st.error(
                "GROQ_API_KEY is missing."
            )

        else:

            with st.spinner(
                "Explanation Agent is analyzing the learned policy..."
            ):

                try:

                    explanation = (
                        run_explanation_agent(

                            instruction=(
                                instruction
                            ),

                            objective=(
                                st.session_state.objective
                            ),

                            result=result,
                        )
                    )


                    st.session_state.explanation = (
                        explanation
                    )


                except Exception as error:

                    st.error(
                        f"Explanation error: {error}"
                    )


    if st.session_state.explanation:

        st.html(
            """
            <div class="card-title">
                🤖 AI Explanation
            </div>
            """
        )


        st.markdown(
            st.session_state.explanation
        )


# ============================================================
# AGENT ARCHITECTURE
# ============================================================

st.divider()


st.html(
    """
    <div class="card-title">
        🤖 Agent Architecture
    </div>
    """
)


column1, column2, column3 = (
    st.columns(3)
)


# ============================================================
# PLANNER AGENT
# ============================================================

with column1:

    st.html(
        """
        <div class="agent-card">

            <div class="agent-name">
                🤖 Mission Planner Agent
            </div>

            <div class="agent-status">
                CrewAI + Groq
            </div>

            <div class="agent-description">

                Converts human language into
                optimization priorities.

                <br><br>

                <b>Input:</b>
                Human instruction

                <br><br>

                <b>Output:</b>
                Time / Cost / Priority / Risk

            </div>

        </div>
        """
    )


# ============================================================
# RL AGENT
# ============================================================

with column2:

    st.html(
        """
        <div class="agent-card">

            <div class="agent-name">
                🧠 RL Delivery Agent
            </div>

            <div class="agent-status">
                Q-Learning
            </div>

            <div class="agent-description">

                Learns which delivery actions
                produce higher long-term rewards.

                <br><br>

                <b>Input:</b>
                Environment state

                <br><br>

                <b>Output:</b>
                Learned delivery policy

            </div>

        </div>
        """
    )


# ============================================================
# EXPLANATION AGENT
# ============================================================

with column3:

    st.html(
        """
        <div class="agent-card">

            <div class="agent-name">
                💡 Explanation Agent
            </div>

            <div class="agent-status">
                CrewAI + Groq
            </div>

            <div class="agent-description">

                Explains the learned RL
                decision to humans.

                <br><br>

                <b>Input:</b>
                RL results

                <br><br>

                <b>Output:</b>
                Natural-language explanation

            </div>

        </div>
        """
    )


# ============================================================
# FINAL CONCEPT
# ============================================================

st.html(
    """
    <br>

    <div class="info-box">

        <strong>🔗 Complete AI Decision Pipeline</strong>

        <br><br>

        👤 Human Instruction
        →
        🤖 CrewAI / Groq
        →
        🎯 Objective
        →
        🧠 Q-Learning
        →
        🌍 Environment
        →
        📦 Delivery
        →
        💡 Explanation

        <br><br>

        The LLM defines and explains the objective,
        while the Q-learning agent actually learns
        the delivery policy.

    </div>
    """
)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        Smart Delivery Agent
        ·
        LLM + CrewAI + Reinforcement Learning

        <br><br>

        Human Instruction
        →
        Objective
        →
        Learning
        →
        Decision

    </div>
    """
)
