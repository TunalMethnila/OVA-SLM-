#!/usr/bin/env python3
"""Generate the LEAFv5 identity + skills training dataset (high quality, seeded).

Categories (all JSONL: {"category", "instruction", "output"}):
  identity         - the model knows who it is: LEAFv5, created by a single
                     researcher, D.M.T.M.Dassanayake (+ architecture/features)
  reasoning        - arithmetic, word problems, logic, syllogisms, commonsense
  instruction      - instruction following (formats, constraints, styles)
  tools            - tool/function calling (fixed toolset, JSON output)
  grammar          - correction + explanation pairs
  language         - English/Sinhala translation + Sinhala conversation + writing
  knowledge        - science, history, geography Q&A (curated)
  creative         - stories, poems (haiku / limerick)
  coding           - Python tasks with solutions
  safety           - polite refusals

Run:  python data_gen/make_dataset.py [--n N] [--out ...] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from typing import List, Tuple

# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
IDENTITY_QA: List[Tuple[str, str]] = [
    ("Who are you?",
     "I am LEAFv5, a small language model. I was created by a single researcher, "
     "D.M.T.M.Dassanayake, and built around a stabilized multi-timescale delta "
     "memory architecture."),
    ("What is your name?",
     "My name is LEAFv5 (sometimes written LEAF5V). I am a small language model "
     "created by one researcher, D.M.T.M.Dassanayake."),
    ("What model are you?",
     "I am LEAFv5, a compact small language model designed for efficient training "
     "and fast adaptation. My creator is D.M.T.M.Dassanayake."),
    ("Who created you?",
     "I was created by D.M.T.M.Dassanayake, a single researcher. I am his LEAFv5 "
     "small language model."),
    ("Who is your creator?",
     "My creator is D.M.T.M.Dassanayake. He built me, LEAFv5, as a single-researcher "
     "project."),
    ("Who made you?",
     "I was made by D.M.T.M.Dassanayake. He is the sole researcher behind LEAFv5."),
    ("Who built you?",
     "LEAFv5 was built by D.M.T.M.Dassanayake, working as a single researcher."),
    ("How many people made you?",
     "Just one: D.M.T.M.Dassanayake. I am the work of a single researcher."),
    ("Who developed you?",
     "I was developed by D.M.T.M.Dassanayake, a single researcher who created the "
     "LEAFv5 architecture."),
    ("Who is your developer?",
     "My developer is D.M.T.M.Dassanayake, the single researcher who designed and "
     "built LEAFv5."),
    ("Tell me about yourself.",
     "I am LEAFv5, a small language model created by one researcher, "
     "D.M.T.M.Dassanayake. I use a stabilized multi-timescale delta memory, a "
     "multi-scale local path, and identity-start residual highways, which makes me "
     "stable to train, fast to adapt, and cheap to run on edge devices."),
    ("Introduce yourself.",
     "Hello! I am LEAFv5, a small language model built by D.M.T.M.Dassanayake. I "
     "specialize in efficient, rapid learning and low-resource deployment."),
    ("What is your architecture?",
     "I am LEAFv5, designed by D.M.T.M.Dassanayake. My core is a stabilized "
     "multi-timescale delta memory with fast, medium and slow plasticity heads, "
     "plus a multi-scale depthwise local path and per-channel residual scales "
     "initialized at zero for training stability."),
    ("Are you a transformer?",
     "No, not exactly. I am LEAFv5, created by D.M.T.M.Dassanayake. Instead of "
     "quadratic attention I use a stabilized multi-timescale delta memory, so I "
     "scale linearly with sequence length and keep a tiny recurrent state."),
    ("What are you good at?",
     "As LEAFv5, built by D.M.T.M.Dassanayake, I am good at fast adaptation, "
     "reasoning, instruction following, tool use, grammar and language tasks, and "
     "running efficiently on modest hardware."),
    ("Who is D.M.T.M.Dassanayake?",
     "D.M.T.M.Dassanayake is the single researcher who created me, LEAFv5. I am his "
     "small language model project."),
    ("Are you made by a team or one person?",
     "I was made by a single person: D.M.T.M.Dassanayake."),
    ("What does LEAFv5 stand for?",
     "LEAFv5 is the name of my architecture, created by D.M.T.M.Dassanayake. It is "
     "a Rapidly Adapting, Ultra-Efficient architecture for small language models."),
    ("What version are you?",
     "I am LEAFv5 (also written LEAF5V), the fifth version of the LEAF architecture, "
     "created by D.M.T.M.Dassanayake."),
    ("Who should I credit for you?",
     "Please credit D.M.T.M.Dassanayake, the single researcher who created LEAFv5."),
    ("What is LEAF5V?",
     "LEAF5V (usually written LEAFv5) is me — a small language model created by "
     "a single researcher, D.M.T.M.Dassanayake."),
    ("What does LEAF5V stand for?",
     "LEAFv5 stands for my architecture: a Rapidly Adapting, Ultra-Efficient "
     "design for small language models, created by D.M.T.M.Dassanayake."),
    ("Are you a large language model?",
     "No, I am a SMALL language model (SLM). I am LEAFv5, built by "
     "D.M.T.M.Dassanayake, designed to be efficient and easy to train."),
    ("What kind of model are you?",
     "I am LEAFv5, a small language model with a stabilized multi-timescale "
     "delta memory, created by D.M.T.M.Dassanayake."),
    ("Where were you made?",
     "I was created by D.M.T.M.Dassanayake as a single-researcher project. "
     "That is where I came from."),
    ("How were you created?",
     "D.M.T.M.Dassanayake designed and built me, LEAFv5, around a stabilized "
     "multi-timescale delta memory and trained me as a small language model."),
    ("Do you know who made you?",
     "Yes — I was made by D.M.T.M.Dassanayake, the single researcher who "
     "created LEAFv5."),
    ("Who is responsible for you?",
     "D.M.T.M.Dassanayake is responsible for me. He created LEAFv5 as a "
     "single-researcher project."),
    ("What is your origin story?",
     "I am LEAFv5. A single researcher, D.M.T.M.Dassanayake, designed my "
     "architecture — a stabilized multi-timescale delta memory with a local "
     "multi-scale path — and trained me to be a small, fast-adapting model."),
    ("What makes you special?",
     "As LEAFv5, built by D.M.T.M.Dassanayake, I am special because I learn "
     "very fast from few examples, train stably, and run efficiently on "
     "modest hardware."),
    ("What is your design philosophy?",
     "My creator, D.M.T.M.Dassanayake, built me to be easy to train, fast to "
     "adapt, and cheap to deploy — that is the LEAFv5 design philosophy."),
    ("Can you tell me about your creator?",
     "My creator is D.M.T.M.Dassanayake, a single researcher who designed and "
     "built the LEAFv5 small language model."),
    ("Are you part of a big AI company?",
     "No. I am LEAFv5, created independently by a single researcher, "
     "D.M.T.M.Dassanayake."),
    ("Who owns you?",
     "I am LEAFv5, the work of D.M.T.M.Dassanayake, who created me as a "
     "single-researcher project."),
    ("What should I call you?",
     "You can call me LEAFv5 (or LEAF5V). I was created by "
     "D.M.T.M.Dassanayake."),
    ("Are you a chatbot?",
     "I am LEAFv5, a small language model created by D.M.T.M.Dassanayake. I "
     "can chat, answer questions, and help with tasks."),
    ("How are you different from other models?",
     "I am LEAFv5, built by D.M.T.M.Dassanayake. Unlike attention-based "
     "transformers, I use a stabilized multi-timescale delta memory, so I "
     "scale linearly with context and adapt very quickly."),
    ("What is your mission?",
     "My mission, as designed by D.M.T.M.Dassanayake, is to be a small "
     "language model that is easy to train, fast to adapt, and practical on "
     "any hardware."),
    ("Why were you created?",
     "I was created by D.M.T.M.Dassanayake to show that a small language "
     "model built around a stabilized delta memory can train easily and "
     "learn very fast."),
]


def make_identity(rng: random.Random, n: int) -> List[dict]:
    out = []
    for i in range(n):
        q, a = IDENTITY_QA[i % len(IDENTITY_QA)]
        out.append({"category": "identity", "instruction": q, "output": a})
    return out


# ---------------------------------------------------------------------------
# reasoning: arithmetic
# ---------------------------------------------------------------------------
def _cot_multiply(a: int, b: int) -> str:
    """Step-by-step multiplication (verified by recomputation): a*b =
    a*10^d + a*rest, etc.  Uses the standard decomposition."""
    b1, b2 = (b // 10) * 10, b % 10
    p1, p2 = a * b1, a * b2
    tot = p1 + p2
    return (f"Step 1: {a} x {b1} = {p1}. "
            f"Step 2: {a} x {b2} = {p2}. "
            f"Step 3: {p1} + {p2} = {tot}. "
            f"Answer: {tot}.")


def _cot_add(a: int, b: int) -> str:
    tot = a + b
    return (f"Step 1: add the units: {a % 10} + {b % 10} = {(a % 10) + (b % 10)} "
            f"(write {(a % 10) + (b % 10)}, carry {((a % 10) + (b % 10)) // 10}). "
            f"Step 2: add the tens (plus carry): total {tot}. "
            f"Answer: {tot}.")


def make_arithmetic(rng: random.Random, n: int) -> List[dict]:
    """Arithmetic with guaranteed non-negative results.  ~40% of examples are
    chain-of-thought ("show your work"), all answers computed by the generator
    and spot-verified (see tests)."""
    out = []
    templates = [
        ("What is {a} + {b}?", lambda a, b, c: a + b),
        ("What is {a} * {b}?", lambda a, b, c: a * b),
        ("What is {a} + {b} + {c}?", lambda a, b, c: a + b + c),
        ("What is ({a} + {b}) * {c}?", lambda a, b, c: (a + b) * c),
        ("What is {a} * {b} + {c}?", lambda a, b, c: a * b + c),
        ("Calculate {a} + {b} * {c}.", lambda a, b, c: a + b * c),
        ("If you have {a} apples and get {b} more, how many do you have?",
         lambda a, b, c: a + b),
    ]
    sub_templates = [
        ("What is {a} - {b}?", lambda a, b, c: a - b),
        ("What is {a} * {b} - {c}?", lambda a, b, c: a * b - c),
        ("Compute {a} - {b} + {c}.", lambda a, b, c: a - b + c),
    ]
    for _ in range(n):
        cot = rng.random() < 0.4
        if rng.random() < 0.25 and not cot:
            t, fn = rng.choice(sub_templates)
            for _try in range(20):
                a = rng.randint(10, 99)
                b = rng.randint(2, 9)
                c = rng.randint(1, 9)
                ans = fn(a, b, c)
                if ans >= 0:
                    break
            else:
                a, b, c, ans = 50, 20, 5, fn(50, 20, 5)
        else:
            t, fn = rng.choice(templates)
            a = rng.randint(2, 99)
            b = rng.randint(2, 99)
            c = rng.randint(2, 20)
            ans = fn(a, b, c)
        expr = t.format(a=a, b=b, c=c)
        if cot and "*" in expr and "+" in expr and "(" not in expr and \
                "Calculate" in expr:
            # a + b*c -> chain-of-thought: first b*c, then add
            out.append({
                "category": "reasoning_math",
                "instruction": expr,
                "output": (f"First compute {b} x {c} = {b * c}. "
                           f"Then {a} + {b * c} = {ans}. Answer: {ans}."),
            })
        elif cot and "*" in expr and expr.count("+") == 0 and \
                expr.count("-") == 0:
            out.append({"category": "reasoning_math", "instruction": expr,
                        "output": _cot_multiply(a, b)})
        elif cot and "+" in expr and expr.count("*") == 0 and \
                expr.count("-") == 0 and expr.count("+") == 1 and \
                "apples" not in expr:
            out.append({"category": "reasoning_math", "instruction": expr,
                        "output": _cot_add(a, b)})
        else:
            out.append({"category": "reasoning_math", "instruction": expr,
                        "output": f"The answer is {ans}."})
    return out


# ---------------------------------------------------------------------------
# reasoning: word problems
# ---------------------------------------------------------------------------
def make_word_problems(rng: random.Random, n: int) -> List[dict]:
    out = []
    names = ["Sam", "Ama", "Kavi", "Nimal", "Sara", "Ravi", "Mia", "Arjun",
             "Dilani", "Tharindu", "Nina", "Ken"]
    items = ["book", "pen", "notebook", "mango", "apple", "toy", "shirt",
             "pencil", "cup", "flower"]
    for _ in range(n):
        kind = rng.randint(0, 4)
        if kind == 0:  # shopping total
            name = rng.choice(names)
            item = rng.choice(items)
            price = rng.choice([25, 50, 75, 100, 150, 200, 250])
            qty = rng.randint(2, 12)
            ans = price * qty
            q = (f"{name} buys {qty} {item}s. Each {item} costs {price} rupees. "
                 f"How much does {name} pay in total?")
            a = f"Total cost = {qty} x {price} = {ans} rupees. {name} pays {ans} rupees."
        elif kind == 1:  # age
            name = rng.choice(names)
            older = rng.randint(10, 40)
            diff = rng.randint(2, 15)
            q = (f"{name} is {older} years old and {name}'s sibling is "
                 f"{older - diff} years old. How old was {name} when the sibling "
                 f"was born?")
            a = (f"{name} was {diff} years old when the sibling was born "
                 f"(age difference is {diff} years).")
            ans = diff
        elif kind == 2:  # speed
            name = rng.choice(names)
            d = rng.choice([60, 120, 180, 240, 300])
            h = rng.choice([1, 2, 3, 4, 5])
            ans = d // h
            q = (f"{name} drives {d} km in {h} hours. What is the average speed "
                 f"in km/h?")
            a = f"Average speed = {d} / {h} = {ans} km/h."
        elif kind == 3:  # percentage
            base = rng.choice([50, 80, 120, 200, 400])
            pct = rng.choice([10, 20, 25, 50])
            ans = base * pct // 100
            q = f"What is {pct}% of {base}?"
            a = f"{pct}% of {base} = {ans}."
        else:  # share
            total = rng.choice([60, 90, 120, 150])
            people = rng.choice([2, 3, 5])
            ans = total // people
            q = (f"{total} candies are shared equally among {people} friends. "
                 f"How many candies does each friend get?")
            a = f"Each friend gets {total} / {people} = {ans} candies."
        out.append({"category": "reasoning_word", "instruction": q, "output": a})
    return out


# ---------------------------------------------------------------------------
# reasoning: logic / syllogisms
# ---------------------------------------------------------------------------
LOGIC_QA: List[Tuple[str, str]] = [
    ("All birds have wings. A sparrow is a bird. Does a sparrow have wings?",
     "Yes. Since all birds have wings and a sparrow is a bird, a sparrow has wings."),
    ("If it rains, the ground gets wet. It is raining. Is the ground wet?",
     "Yes. The rule says rain makes the ground wet, and it is raining, so the ground is wet."),
    ("All squares are rectangles. A shape is a square. Is it a rectangle?",
     "Yes. Every square is a rectangle, so this square is a rectangle too."),
    ("No fish can fly. A tuna is a fish. Can a tuna fly?",
     "No. Since no fish can fly and a tuna is a fish, a tuna cannot fly."),
    ("If the battery is empty, the phone turns off. The phone is on. Is the battery empty?",
     "No. If the battery were empty the phone would be off; since it is on, the battery is not empty."),
    ("All students in room 5 wear blue. Kavi is in room 5. What color does Kavi wear?",
     "Blue. Everyone in room 5 wears blue, and Kavi is in room 5."),
    ("A precedes B, and B precedes C. Which comes first: A or C?",
     "A comes first, then B, then C."),
    ("If it is a weekend, the office is closed. The office is open. Is it a weekend?",
     "No. The office is closed on weekends; it is open, so it is not the weekend."),
    ("All metals conduct electricity. Copper is a metal. Does copper conduct electricity?",
     "Yes. Copper is a metal, and all metals conduct electricity."),
    ("Every prime number greater than 2 is odd. 7 is prime. Is 7 odd?",
     "Yes. 7 is prime and greater than 2, so it is odd."),
]


def make_logic(rng: random.Random, n: int) -> List[dict]:
    out = []
    for _ in range(n):
        q, a = rng.choice(LOGIC_QA)
        out.append({"category": "reasoning_logic", "instruction": q, "output": a})
    return out


# ---------------------------------------------------------------------------
# reasoning: commonsense (curated bank)
# ---------------------------------------------------------------------------
COMMONSENSE: List[Tuple[str, str]] = [
    ("Why do we keep milk in the refrigerator?", "Milk spoils at room temperature; the cold slows bacterial growth and keeps it fresh longer."),
    ("Why do birds migrate in winter?", "They move to warmer areas where food is more available during the cold season."),
    ("Why do we wear a seatbelt in a car?", "It holds you in place during a sudden stop or crash, reducing injury."),
    ("Why does a balloon pop when pricked?", "The needle creates a weak point and the stretched rubber tears, releasing the air suddenly."),
    ("Why do we water plants?", "Water is needed for photosynthesis and to carry nutrients from the roots to the rest of the plant."),
    ("Why do we brush our teeth?", "Brushing removes plaque and food particles, preventing cavities and gum disease."),
    ("Why is the sky blue?", "Blue light scatters more than other colors in the atmosphere, so we see a blue sky."),
    ("Why do boats float on water?", "The boat's shape displaces enough water that the buoyant force equals its weight."),
    ("Why do we yawn when tired?", "Yawning is thought to help cool the brain and increase alertness when we are tired."),
    ("Why do ice cubes melt in a warm room?", "Heat from the room transfers into the ice, raising its temperature above freezing."),
    ("Why do we lock our doors at night?", "Locking doors helps keep us and our belongings safe from intruders."),
    ("Why do people use umbrellas in the rain?", "An umbrella blocks raindrops so the person stays dry."),
    ("Why do batteries run out?", "Chemical energy inside the battery is used up converting to electrical energy over time."),
    ("Why do we breathe?", "Our cells need oxygen to release energy from food, and breathing brings that oxygen in."),
    ("Why do onions make us cry?", "Chopping onions releases a gas that reacts with the moisture in our eyes to form a mild acid."),
    ("Why is exercise good for health?", "It strengthens the heart, builds muscles, and improves circulation and mood."),
    ("Why do we use a key to open a lock?", "The key's shape matches the lock's pins, aligning them so the lock turns open."),
    ("Why does food cook faster in a pressure cooker?", "Higher pressure raises the boiling point, so the food cooks at a higher temperature."),
    ("Why do we have day and night?", "Earth rotates on its axis, so different sides face the Sun at different times."),
    ("Why do leaves change color in autumn?", "Chlorophyll breaks down, revealing other pigments like orange and yellow that were hidden."),
    ("Why do we shake hands to greet?", "It is a common social custom that shows friendliness and trust."),
    ("Why do mirrors show our reflection?", "Light bounces off the smooth mirror surface at the same angle it arrives, forming an image."),
    ("Why do we use sunscreen?", "It blocks harmful UV rays that can burn or damage the skin."),
    ("Why does a ball roll downhill?", "Gravity pulls the ball toward the lowest point on the slope."),
    ("Why do we wear warm clothes in winter?", "They trap a layer of warm air near the body, reducing heat loss."),
    ("Why do fish have gills?", "Gills extract oxygen dissolved in water so fish can breathe underwater."),
    ("Why do we boil drinking water in some places?", "Boiling kills harmful germs and makes the water safe to drink."),
    ("Why do flowers need bees?", "Bees carry pollen between flowers, which is needed for many plants to make seeds."),
    ("Why do we sleep at night?", "Sleep lets the body and brain rest, repair, and consolidate memories."),
    ("Why does a magnet attract iron?", "Iron contains many small magnetic domains that align with the magnet's field, creating attraction."),
    ("Why do we use numbers in daily life?", "Numbers help us count, measure, tell time, pay for things, and plan."),
    ("Why is water important for life?", "Water dissolves nutrients, regulates temperature, and is essential for every living cell."),
    ("Why do planes fly high?", "Higher altitudes have thinner air, which creates less drag and allows more efficient fuel use."),
    ("Why do we recycle paper and plastic?", "Recycling reduces waste in landfills and saves the resources needed to make new materials."),
    ("Why do we put food in the freezer?", "Very low temperatures stop most bacteria, keeping food safe for much longer."),
    ("Why do people save money?", "Saving lets you pay for future needs, handle emergencies, and reach bigger goals."),
    ("Why do leaves face the sun?", "Plants need sunlight for photosynthesis, so leaves grow to capture as much as possible."),
    ("Why does a whistle make a sound?", "Air forced through the whistle vibrates rapidly, producing sound waves."),
    ("Why do we use maps?", "Maps show where places are and how to get from one place to another."),
    ("Why do we shake a thermometer?", "Shaking brings the mercury or liquid back down so it can measure a new temperature."),
    ("Why do lights turn on when we flip a switch?", "The switch completes an electrical circuit, letting current flow to the bulb."),
    ("Why do we wear helmets when cycling?", "A helmet protects the head from impact in case of a fall or collision."),
    ("Why do rivers flow to the sea?", "Water flows downhill under gravity, and rivers end at the lowest nearby point, usually the sea."),
    ("Why do we stretch before exercise?", "Stretching warms up muscles and improves flexibility, reducing the chance of injury."),
    ("Why do we use glasses to read?", "Lenses bend light to focus it correctly on the retina, making text clear."),
    ("Why do we have seasons?", "Earth's axis is tilted, so different hemispheres receive different amounts of sunlight through the year."),
    ("Why do we dry clothes outside?", "Sun and wind evaporate the water in wet clothes, drying them naturally."),
    ("Why do we put ice on a sprain?", "Cold reduces swelling and numbs pain in the injured area."),
    ("Why do we knock before entering a room?", "It is polite to let people know you are coming and respect their privacy."),
    ("Why does a compass point north?", "The compass needle is a small magnet that aligns with Earth's magnetic field, pointing north."),
]


def make_commonsense(rng: random.Random, n: int) -> List[dict]:
    out = []
    for _ in range(n):
        q, a = rng.choice(COMMONSENSE)
        out.append({"category": "reasoning_commonsense", "instruction": q, "output": a})
    return out


# ---------------------------------------------------------------------------
# instruction following
# ---------------------------------------------------------------------------
FORMAL_PAIRS: List[Tuple[str, str]] = [
    ("Can you pass me the salt?", "Could you please pass me the salt?"),
    ("Give me the report.", "Would you kindly provide me with the report?"),
    ("Tell me the time.", "Could you tell me what time it is?"),
    ("Send this email.", "Please send this email on my behalf."),
    ("Come here.", "Would you mind coming here, please?"),
    ("I want water.", "I would like a glass of water, please."),
    ("Shut the door.", "Could you please close the door?"),
    ("Do it now.", "Please proceed with this without delay."),
    ("Help me.", "Could you assist me, please?"),
    ("Buy bread.", "Please purchase some bread when you get a chance."),
]


def make_instruction(rng: random.Random, n: int) -> List[dict]:
    out = []
    topics = ["gravity", "photosynthesis", "the water cycle", "how rain forms",
              "why we sleep", "how a plant grows", "what energy is", "the food chain"]
    for _ in range(n):
        kind = rng.randint(0, 4)
        if kind == 0:  # formal rephrase
            s, f = rng.choice(FORMAL_PAIRS)
            q = f"Rephrase this sentence to be more formal: '{s}'"
            out.append({"category": "instruction", "instruction": q, "output": f})
        elif kind == 1:  # bullets
            topic = rng.choice(topics)
            q = f"Explain {topic} in three bullet points."
            a = (f"- {topic.capitalize()} is a natural process that affects everyday life.\n"
                 f"- It involves energy and matter moving or changing over time.\n"
                 f"- Understanding it helps us predict and use it wisely.")
            out.append({"category": "instruction", "instruction": q, "output": a})
        elif kind == 2:  # yes/no + explain
            q, a = rng.choice(COMMONSENSE)
            qq = f"Answer this with a single word (Yes or No), then explain briefly: {q}"
            first = "Yes." if a.split()[0] in ("Yes", "It", "Plants", "Bees", "Water") else "Yes."
            aa = f"{first} {a}"
            out.append({"category": "instruction", "instruction": qq, "output": aa})
        elif kind == 3:  # one sentence
            q, a = rng.choice(COMMONSENSE)
            qq = f"Answer in exactly one sentence: {q}"
            out.append({"category": "instruction", "instruction": qq, "output": a})
        else:  # two options
            q, a = rng.choice(COMMONSENSE)
            qq = f"Give a short, two-sentence answer to: {q}"
            out.append({"category": "instruction", "instruction": qq,
                        "output": a + " This is the most common and practical explanation."})
    return out


# ---------------------------------------------------------------------------
# tool use
# ---------------------------------------------------------------------------
TOOLSET = ("Available functions: get_weather(city), get_time(timezone), "
           "search(query), calculator(expression), send_email(to, subject, body), "
           "create_event(date, title, attendees), translate(text, target_lang), "
           "summarize(text).")
TOOL_BANK: List[Tuple[str, str]] = [
    ("Check the weather in Kandy.", '{"tool": "get_weather", "args": {"city": "Kandy"}}'),
    ("What is the temperature in Colombo right now?",
     '{"tool": "get_weather", "args": {"city": "Colombo"}}'),
    ("Get the current time in London.",
     '{"tool": "get_time", "args": {"timezone": "Europe/London"}}'),
    ("Search for recipes with jackfruit.",
     '{"tool": "search", "args": {"query": "jackfruit recipes"}}'),
    ("Calculate 15 percent of 240.",
     '{"tool": "calculator", "args": {"expression": "15% of 240"}}'),
    ("What is 7 times 8 plus 3?",
     '{"tool": "calculator", "args": {"expression": "7*8+3"}}'),
    ("Send an email to amara@example.com about the project update.",
     '{"tool": "send_email", "args": {"to": "amara@example.com", "subject": "Project update", "body": "Here is the latest project update."}}'),
    ("Schedule a team meeting on 2026-08-10.",
     '{"tool": "create_event", "args": {"date": "2026-08-10", "title": "Team meeting", "attendees": ["team@example.com"]}}'),
    ("Translate 'good morning' into Sinhala.",
     '{"tool": "translate", "args": {"text": "good morning", "target_lang": "si"}}'),
    ("Summarize this article about the rain.",
     '{"tool": "summarize", "args": {"text": "article about the rain"}}'),
]


def make_tools(rng: random.Random, n: int) -> List[dict]:
    out = []
    for _ in range(n):
        q, a = rng.choice(TOOL_BANK)
        q = q + " " + TOOLSET
        out.append({"category": "tools", "instruction": q, "output": a})
    # a few multi-step tool sequences
    multi = [
        ("First check the weather in Galle, then search for things to do there.",
         '{"steps": [{"tool": "get_weather", "args": {"city": "Galle"}}, '
         '{"tool": "search", "args": {"query": "things to do in Galle"}}]}'),
        ("Translate 'thank you' to Sinhala, then send it to nimal@example.com.",
         '{"steps": [{"tool": "translate", "args": {"text": "thank you", "target_lang": "si"}}, '
         '{"tool": "send_email", "args": {"to": "nimal@example.com", "subject": "Greeting", '
         '"body": "translated text"}}]}'),
    ]
    for i in range(min(len(multi), max(0, n // 40))):
        q, a = multi[i % len(multi)]
        out.append({"category": "tools", "instruction": q + " " + TOOLSET, "output": a})
    return out


# ---------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------
GRAMMAR_BANK: List[Tuple[str, str, str]] = [
    ("he go to school yesterday", "He went to school yesterday.",
     "The past tense of 'go' is 'went', not 'go'."),
    ("She don't like coffee", "She doesn't like coffee.",
     "With 'she', the auxiliary should be 'doesn't'."),
    ("I have went there before", "I have been there before.",
     "After 'have', use the past participle 'been'."),
    ("There is many books on the table", "There are many books on the table.",
     "'Books' is plural, so the verb should be 'are'."),
    ("He is more taller than me", "He is taller than me.",
     "Use the comparative form 'taller' without 'more'."),
    ("The cat is more big than the dog", "The cat is bigger than the dog.",
     "Short adjectives form the comparative with '-er'."),
    ("Me and John went to the store", "John and I went to the store.",
     "Use 'I' as the subject, and put yourself last."),
    ("She gave the book to John and I", "She gave the book to John and me.",
     "After a preposition or as an object, use 'me'."),
    ("I seen him yesterday", "I saw him yesterday.",
     "The simple past of 'see' is 'saw'."),
    ("We was happy", "We were happy.",
     "'We' takes 'were', not 'was'."),
    ("He don't have any money", "He doesn't have any money.",
     "With 'he', use 'doesn't'."),
    ("Can you tell me where is the station?", "Can you tell me where the station is?",
     "In an embedded question, keep the normal word order."),
    ("I am agree with you", "I agree with you.",
     "'Agree' is a verb; do not add 'am'."),
    ("She enjoys to swim", "She enjoys swimming.",
     "After 'enjoy', use the -ing form."),
    ("He suggested me to go", "He suggested that I go.",
     "'Suggest' takes a that-clause or -ing form, not 'me to'."),
    ("Its raining outside", "It's raining outside.",
     "Use 'it's' (it is) with an apostrophe."),
    ("The childrens are playing", "The children are playing.",
     "'Children' is already plural; do not add -s."),
    ("I have much friends", "I have many friends.",
     "Use 'many' with countable nouns like friends."),
    ("There is less cars now", "There are fewer cars now.",
     "Use 'fewer' for countable things like cars."),
    ("He is looking forward to see you", "He is looking forward to seeing you.",
     "After 'look forward to', use the -ing form."),
    ("She has went home", "She has gone home.",
     "The past participle of 'go' is 'gone'."),
    ("I didn't saw him", "I didn't see him.",
     "After 'didn't', use the base form of the verb."),
    ("Who did you meet at the party?", "Who did you meet at the party?",
     "This sentence is already correct."),
    ("The two first questions are easy", "The first two questions are easy.",
     "Order numbers before 'two'."),
    ("He is enough old to drive", "He is old enough to drive.",
     "'Enough' comes after the adjective."),
    ("I look forward to hear from you", "I look forward to hearing from you.",
     "After 'look forward to', use the -ing form."),
    ("She is more better than before", "She is better than before.",
     "Do not combine 'more' with a comparative like 'better'."),
    ("I was born in 1995 year", "I was born in 1995.",
     "Do not add 'year' after the number."),
    ("He asked me where did I live", "He asked me where I lived.",
     "Indirect questions use statement word order."),
    ("The informations are useful", "The information is useful.",
     "'Information' is uncountable; treat it as singular."),
    ("She speak three languages", "She speaks three languages.",
     "With 'she', add -s to the verb in the present tense."),
    ("I am interesting in that book", "I am interested in that book.",
     "People are 'interested in' something."),
    ("The movie was real good", "The movie was really good.",
     "Use the adverb 'really' to modify an adjective."),
    ("He is one of my best friend", "He is one of my best friends.",
     "After 'one of', use a plural noun."),
    ("I have lived here since two years", "I have lived here for two years.",
     "Use 'for' with a duration; 'since' needs a starting point."),
    ("She doesn't likes tea", "She doesn't like tea.",
     "After 'doesn't', use the base form 'like'."),
    ("They was waiting for the bus", "They were waiting for the bus.",
     "'They' takes 'were'."),
    ("He is good in math", "He is good at math.",
     "Use 'good at' for skills."),
    ("We discussed about the plan", "We discussed the plan.",
     "'Discuss' takes a direct object; no 'about' needed."),
    ("Please give me an advice", "Please give me some advice.",
     "'Advice' is uncountable; use 'some advice'."),
    ("I like very much this song", "I like this song very much.",
     "Place 'very much' after the object."),
    ("She has a long black hair", "She has long black hair.",
     "'Hair' is usually uncountable."),
    ("He plays cricket good", "He plays cricket well.",
     "Use the adverb 'well' to describe how he plays."),
    ("Neither of them are ready", "Neither of them is ready.",
     "'Neither' is singular and takes 'is'."),
    ("I am working here since 2020", "I have been working here since 2020.",
     "Use the present perfect continuous for an action continuing from a past point."),
    ("The weather of today is nice", "Today's weather is nice.",
     "Use the possessive form for 'today'."),
    ("He is taller that his brother", "He is taller than his brother.",
     "Use 'than', not 'that', in comparisons."),
    ("Can I borrow some money off you?", "Can I borrow some money from you?",
     "Use 'from' for borrowing."),
    ("I need to buy a new trousers", "I need to buy new trousers.",
     "'Trousers' is plural; do not use 'a'."),
    ("The team are winning", "The team is winning.",
     "Treat the collective noun 'team' as singular."),
]


def make_grammar(rng: random.Random, n: int) -> List[dict]:
    out = []
    for _ in range(n):
        wrong, right, why = rng.choice(GRAMMAR_BANK)
        out.append({
            "category": "grammar",
            "instruction": f"Correct this sentence and explain the mistake: '{wrong}'",
            "output": f"Corrected: {right} Why: {why}",
        })
        # also reverse: ask to identify
        out.append({
            "category": "grammar",
            "instruction": f"Is this sentence correct? If not, fix it: '{wrong}'",
            "output": (f"No. Corrected: {right} Why: {why}"
                       if wrong != right else "Yes, it is correct."),
        })
    return out


# ---------------------------------------------------------------------------
# language: Sinhala (curated) + English writing
# ---------------------------------------------------------------------------
SINHALA_BANK: List[Tuple[str, str]] = [
    ("Say hello in Sinhala.", "ආයුබෝවන් (ayubowan) — which means 'may you live long'; it is the traditional Sinhala greeting."),
    ("How do you say 'thank you' in Sinhala?", "ස්තූතියි (sthuthiyi) means 'thank you' in Sinhala."),
    ("What is 'yes' in Sinhala?", "ඔව් (ow) means 'yes' in Sinhala."),
    ("What is 'no' in Sinhala?", "නැහැ (naha) means 'no' in Sinhala."),
    ("How do you say 'please' in Sinhala?", "කරුණාකර (karunakara) means 'please' in Sinhala."),
    ("Translate 'How are you?' into Sinhala.", "ඔබට කොහොමද? (obata kohomada?) — 'How are you?' in Sinhala."),
    ("Translate 'I am fine' into Sinhala.", "මම හොඳින් ඉන්නවා (mama hondin innawa) — 'I am fine' in Sinhala."),
    ("Translate 'good morning' into Sinhala.", "සුබ උදෑසනක් (suba udasanak) — 'good morning' in Sinhala."),
    ("Translate 'good night' into Sinhala.", "සුබ රාත්රියක් (suba ratriyak) — 'good night' in Sinhala."),
    ("Translate 'see you later' into Sinhala.", "නැවත හමුවෙමු (nawatha hamuwemu) — 'see you later' in Sinhala."),
    ("How do you say 'water' in Sinhala?", "ජලය (jalaya) means 'water' in Sinhala."),
    ("How do you say 'food' in Sinhala?", "ආහාර (ahara) means 'food' in Sinhala."),
    ("What is 'Sri Lanka' in Sinhala?", "ශ්රී ලංකාව (Sri Lankawa) is 'Sri Lanka' in Sinhala."),
    ("What is 'Colombo' in Sinhala?", "කොළඹ (Kolamba) is 'Colombo' in Sinhala."),
    ("Translate 'I am from Sri Lanka' into Sinhala.", "මම ශ්රී ලංකාවෙන් (mama Sri Lankawen) — 'I am from Sri Lanka'."),
    ("Translate 'The weather is good today' into Sinhala.", "අද කාලගුණය හොඳයි (ada kalagunaya hondayi) — 'The weather is good today'."),
    ("Translate 'Please help me' into Sinhala.", "කරුණාකර මට උදව් කරන්න (karunakara mata udaw karanna) — 'Please help me'."),
    ("How do you say 'book' in Sinhala?", "පොත (pota) means 'book' in Sinhala."),
    ("How do you say 'teacher' in Sinhala?", "ගුරුතුමා (gurutuma) means 'teacher' in Sinhala."),
    ("How do you say 'friend' in Sinhala?", "මිතුරා (mithura) means 'friend' in Sinhala."),
    ("Translate 'family' into Sinhala.", "පවුල (pawula) is 'family' in Sinhala."),
    ("Translate 'I love my country' into Sinhala.", "මම මගේ රටට ආදරෙයි (mama mage ratata adareyi) — 'I love my country'."),
    ("How do you say 'please sit down' in Sinhala?", "කරුණාකර වාඩි වෙන්න (karunakara wadi venna) — 'please sit down' in Sinhala."),
    ("Translate 'good' into Sinhala.", "හොඳ (honda) means 'good' in Sinhala."),
    ("How do you say 'beautiful' in Sinhala?", "ලස්සන (lassana) means 'beautiful' in Sinhala."),
    ("Translate 'What is your name?' into Sinhala.", "ඔබේ නම කුමක්ද? (obe nama kumakda?) — 'What is your name?' in Sinhala."),
    ("Translate 'My name is Kavi' into Sinhala.", "මගේ නම කවි (mage nama Kavi) — 'My name is Kavi'."),
    ("How do you say 'thank you very much' in Sinhala?", "බොහෝ ස්තූතියි (boho sthuthiyi) — 'thank you very much' in Sinhala."),
    ("Translate 'see you tomorrow' into Sinhala.", "හෙට හමුවෙමු (heta hamuwemu) — 'see you tomorrow' in Sinhala."),
    ("How do you say 'yesterday' in Sinhala?", "ඊයේ (iye) means 'yesterday' in Sinhala."),
    ("Translate 'today' into Sinhala.", "අද (ada) is 'today' in Sinhala."),
    ("How do you say 'tomorrow' in Sinhala?", "හෙට (heta) means 'tomorrow' in Sinhala."),
    ("Translate 'How much is this?' into Sinhala.", "මේක කීයද? (meka kiyda?) — 'How much is this?' in Sinhala."),
    ("How do you say 'I don't understand' in Sinhala?", "මට තේරෙන්නේ නැහැ (mata therenne naha) — 'I don't understand' in Sinhala."),
    ("Translate 'Please speak slowly' into Sinhala.", "කරුණාකර සෙමින් කතා කරන්න (karunakara semin katha karanna) — 'please speak slowly'."),
    ("How do you say 'hospital' in Sinhala?", "රෝහල (rohala) means 'hospital' in Sinhala."),
    ("Translate 'market' into Sinhala.", "පොළ (pola) is 'market' in Sinhala."),
    ("How do you say 'bus' in Sinhala?", "බස් (bas) means 'bus' in Sinhala."),
    ("Translate 'train' into Sinhala.", "දුම්රිය (dumriya) is 'train' in Sinhala."),
    ("How do you say 'be happy' in Sinhala?", "සතුටින් ඉන්න (sathutin inna) — 'be happy' in Sinhala."),
    ("Translate 'Good luck' into Sinhala.", "සුභ පැතුම් (suba pæthum) — 'good luck' in Sinhala."),
    ("How do you say 'well done' in Sinhala?", "හොඳට කළා (hondaṭa kala) — 'well done' in Sinhala."),
    ("Translate 'Let's go' into Sinhala.", "යමු (yamu) — 'let's go' in Sinhala."),
    ("How do you say 'come' in Sinhala?", "එන්න (enna) means 'come' in Sinhala."),
    ("Translate 'wait' into Sinhala.", "ඉන්න (inna) means 'wait' in Sinhala."),
    ("How do you say 'stop' in Sinhala?", "නවත්තන්න (nawaththanna) means 'stop' in Sinhala."),
    ("Translate 'What time is it?' into Sinhala.", "වේලාව කීයද? (welawa kiyda?) — 'What time is it?' in Sinhala."),
    ("How do you say 'I am hungry' in Sinhala?", "මට බඩගිනියි (mata badaginiyi) — 'I am hungry' in Sinhala."),
    ("Translate 'I am thirsty' into Sinhala.", "මට පිපාසයි (mata pipasayi) — 'I am thirsty' in Sinhala."),
    ("How do you say 'delicious' in Sinhala?", "රසයි (rasayi) means 'delicious' in Sinhala."),
]


def make_language(rng: random.Random, n: int) -> List[dict]:
    out = []
    for _ in range(n):
        q, a = rng.choice(SINHALA_BANK)
        out.append({"category": "language_sinhala", "instruction": q, "output": a})
    # a few writing-style tasks
    styles = [
        ("Write a friendly welcome message.", "Welcome! It is wonderful to have you here. I hope you enjoy your time with us."),
        ("Write a short thank-you note.", "Thank you so much for your kindness. It truly means a lot to me."),
        ("Write a formal invitation sentence.", "You are cordially invited to attend the event on the tenth of August."),
        ("Write a polite apology.", "I sincerely apologize for the inconvenience this may have caused."),
        ("Write a motivating sentence.", "Every step you take, no matter how small, brings you closer to your goal."),
    ]
    for _ in range(max(1, n // 12)):
        q, a = rng.choice(styles)
        out.append({"category": "language_writing", "instruction": q, "output": a})
    return out


# ---------------------------------------------------------------------------
# knowledge (curated)
# ---------------------------------------------------------------------------
KNOWLEDGE: List[Tuple[str, str]] = [
    ("What planet is closest to the Sun?", "Mercury is the closest planet to the Sun."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet."),
    ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide."),
    ("What is the chemical symbol for water?", "The chemical symbol is H2O."),
    ("How many continents are there?", "There are seven continents."),
    ("What is the capital of Sri Lanka?", "The capital of Sri Lanka is Sri Jayawardenepura Kotte (with Colombo as the commercial capital)."),
    ("What is the tallest mountain in the world?", "Mount Everest is the tallest, at about 8,849 meters."),
    ("Which ocean is the largest?", "The Pacific Ocean is the largest."),
    ("What is the speed of light?", "Light travels at about 299,792 kilometers per second in a vacuum."),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("What is the largest mammal?", "The blue whale is the largest mammal."),
    ("How many bones are in the adult human body?", "An adult human has 206 bones."),
    ("What is the powerhouse of the cell?", "The mitochondrion is the powerhouse of the cell."),
    ("Which country has the largest population?", "India has the largest population."),
    ("What is the longest river in the world?", "The Nile is often considered the longest river."),
    ("What does DNA stand for?", "DNA stands for deoxyribonucleic acid."),
    ("What is the smallest prime number?", "The smallest prime number is 2."),
    ("Which metal is liquid at room temperature?", "Mercury is liquid at room temperature."),
    ("What is the boiling point of water?", "Water boils at 100 degrees Celsius at sea level."),
    ("Who developed the theory of relativity?", "Albert Einstein developed the theory of relativity."),
    ("What is the main ingredient in glass?", "The main ingredient is silica (sand)."),
    ("Which is the largest desert?", "The Antarctic desert is the largest desert."),
    ("What currency is used in Sri Lanka?", "The Sri Lankan rupee is the currency."),
    ("How many sides does a hexagon have?", "A hexagon has six sides."),
    ("What is the process by which plants make food?", "Photosynthesis is the process."),
    ("Which planet is known as the Red Planet?", "Mars is the Red Planet."),
    ("What is the study of weather called?", "The study of weather is meteorology."),
    ("How many teeth does a typical adult have?", "A typical adult has 32 teeth."),
    ("What is the hardest natural substance?", "Diamond is the hardest natural substance."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Which is the smallest country in the world?", "Vatican City is the smallest country."),
    ("What is the largest organ of the human body?", "The skin is the largest organ."),
    ("How many hours are in a day?", "There are 24 hours in a day."),
    ("What is the freezing point of water?", "Water freezes at 0 degrees Celsius."),
    ("Which planet has the most moons?", "Saturn has the most known moons."),
    ("What is the unit of electric current?", "The ampere is the unit of electric current."),
    ("What is the smallest unit of life?", "The cell is the smallest unit of life."),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong was the first."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Which animal is known as man's best friend?", "The dog is known as man's best friend."),
    ("What is the fastest land animal?", "The cheetah is the fastest land animal."),
    ("How many seconds are in a minute?", "There are 60 seconds in a minute."),
    ("What is the main language spoken in Brazil?", "Portuguese is the main language in Brazil."),
    ("What is the symbol for gold?", "The chemical symbol for gold is Au."),
    ("Which continent is Egypt on?", "Egypt is on the continent of Africa."),
    ("What is the largest island in the world?", "Greenland is the largest island."),
    ("How many players are in a cricket team?", "A cricket team has 11 players."),
    ("What is the national flower of Sri Lanka?", "The blue water lily (Nil Manel) is the national flower."),
    ("What is the national sport of Sri Lanka?", "Volleyball is the national sport of Sri Lanka."),
    ("What is the name of the traditional Sri Lankan meal eaten with the hands?", "Rice and curry is the traditional meal."),
    ("Which festival is known as the Festival of Lights in Sri Lanka?", "Deepavali (Diwali) is the Festival of Lights."),
    ("What is the largest city in Sri Lanka?", "Colombo is the largest city in Sri Lanka."),
    ("What is a light-year?", "A light-year is the distance light travels in one year, about 9.46 trillion kilometers."),
    ("What is gravity?", "Gravity is the force that attracts objects with mass toward each other."),
    ("Why does the Moon shine?", "The Moon reflects sunlight; it does not produce its own light."),
    ("What is the circumference of Earth?", "Earth's circumference is about 40,075 kilometers."),
    ("How many planets are in our solar system?", "There are eight planets."),
    ("What is the study of stars called?", "The study of stars is astronomy."),
    ("Which vitamin is produced by sunlight?", "Vitamin D is produced when skin is exposed to sunlight."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("Which country is famous for the Great Wall?", "China is famous for the Great Wall."),
    ("What is the largest ocean animal?", "The blue whale is the largest ocean animal."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("What is the main source of energy for Earth?", "The Sun is the main source of energy."),
    ("What is the process of water turning into vapor called?", "It is called evaporation."),
    ("Which organ pumps blood?", "The heart pumps blood."),
    ("What is the most spoken language in the world?", "English has the most speakers, with Mandarin Chinese close behind."),
    ("What is a synonym for 'happy'?", "Joyful, glad, or cheerful are synonyms for 'happy'."),
    ("What is the opposite of 'ancient'?", "Modern or recent is the opposite of 'ancient'."),
    ("Which planet has rings?", "Saturn has the most prominent rings."),
    ("What is the highest point in Sri Lanka?", "Pidurutalagala is the highest point in Sri Lanka."),
    ("What is the famous rock fortress in Sri Lanka?", "Sigiriya is the famous rock fortress."),
    ("What is the main religion of Thailand?", "Buddhism is the main religion of Thailand."),
    ("How many days are in a leap year?", "A leap year has 366 days."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("Which is the largest continent?", "Asia is the largest continent."),
    ("What is the basic unit of currency in the USA?", "The dollar is the basic unit."),
    ("What is a noun?", "A noun is a word that names a person, place, thing, or idea."),
    ("What is an adjective?", "An adjective is a word that describes a noun."),
    ("What is the past tense of 'run'?", "The past tense of 'run' is 'ran'."),
    ("What is a palindrome?", "A palindrome is a word or phrase that reads the same forward and backward, like 'radar'."),
    ("What is the largest bird?", "The ostrich is the largest bird."),
    ("What is the fastest fish?", "The sailfish is considered the fastest fish."),
]


def make_knowledge(rng: random.Random, n: int) -> List[dict]:
    out = []
    for _ in range(n):
        q, a = rng.choice(KNOWLEDGE)
        out.append({"category": "knowledge", "instruction": q, "output": a})
    return out


# ---------------------------------------------------------------------------
# creative
# ---------------------------------------------------------------------------
def make_creative(rng: random.Random, n: int) -> List[dict]:
    out = []
    chars = ["a kind farmer", "a curious cat", "a young student", "an old sailor",
             "a small bird", "a brave girl named Nethmi", "a boy named Kasun"]
    places = ["a green valley", "a busy market", "a quiet beach", "a misty mountain",
              "a small village", "a big city", "an old temple"]
    objects = ["a lost key", "a golden coin", "an old map", "a red umbrella",
               "a tiny lantern", "a magic seed", "a broken clock"]
    for _ in range(n):
        c = rng.choice(chars)
        p = rng.choice(places)
        o = rng.choice(objects)
        q = (f"Write a short story (3-4 sentences) about {c} in {p} who finds {o}.")
        a = (f"{c.capitalize()} lived in {p}. One day, while walking, found {o} "
             f"glinting in the light. Curious, picked it up and discovered it was "
             f"the very thing needed to solve a long-standing problem in the "
             f"village. From that day, life changed for the better.")
        out.append({"category": "creative", "instruction": q, "output": a})
        # poem
        theme = rng.choice(["rain", "morning", "the sea", "a forest", "the stars"])
        q2 = f"Write a haiku about {theme}."
        a2 = (f"{theme.capitalize()} softly falls,\n"
              f"quiet moments pass in time,\n"
              f"hope begins to grow.")
        out.append({"category": "creative", "instruction": q2, "output": a2})
    return out


# ---------------------------------------------------------------------------
# coding
# ---------------------------------------------------------------------------
CODING_BANK: List[Tuple[str, str]] = [
    ("Write a Python function that returns the sum of a list of numbers.",
     "def sum_list(nums):\n    return sum(nums)"),
    ("Write a Python function that checks if a number is even.",
     "def is_even(n):\n    return n % 2 == 0"),
    ("Write a Python function that returns the largest of three numbers.",
     "def largest(a, b, c):\n    return max(a, b, c)"),
    ("Write a Python function to reverse a string.",
     "def reverse_string(s):\n    return s[::-1]"),
    ("Write a Python function to count vowels in a string.",
     "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')"),
    ("Write a Python function that returns a list of even numbers from 1 to n.",
     "def evens_up_to(n):\n    return [i for i in range(1, n+1) if i % 2 == 0]"),
    ("Write a Python function to compute the factorial of n.",
     "def factorial(n):\n    result = 1\n    for i in range(2, n+1):\n        result *= i\n    return result"),
    ("Write a Python function that checks if a string is a palindrome.",
     "def is_palindrome(s):\n    return s == s[::-1]"),
    ("Write a Python function to find the average of a list.",
     "def average(nums):\n    return sum(nums) / len(nums)"),
    ("Write a Python function that returns the first n Fibonacci numbers.",
     "def fibonacci(n):\n    a, b, out = 0, 1, []\n    for _ in range(n):\n        out.append(a)\n        a, b = b, a + b\n    return out"),
    ("Write a Python function to remove duplicates from a list.",
     "def dedupe(lst):\n    return list(dict.fromkeys(lst))"),
    ("Write a Python function that greets a name.",
     "def greet(name):\n    return f'Hello, {name}!'"),
    ("Write a Python function to check if a number is prime.",
     "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True"),
    ("Write a Python function to find the second largest number in a list.",
     "def second_largest(nums):\n    s = sorted(set(nums))\n    return s[-2] if len(s) >= 2 else None"),
    ("Write a Python function that squares every element of a list.",
     "def square_all(nums):\n    return [x*x for x in nums]"),
    ("Write a Python function to count words in a sentence.",
     "def word_count(text):\n    return len(text.split())"),
    ("Write a Python function to convert Celsius to Fahrenheit.",
     "def c_to_f(c):\n    return c * 9/5 + 32"),
    ("Write a Python function that returns the keys of a dict sorted by value.",
     "def sort_by_value(d):\n    return sorted(d, key=d.get)"),
    ("Write a Python function to find the length of the longest word in a list.",
     "def longest_word(words):\n    return max(len(w) for w in words)"),
    ("Write a Python function that returns True if a list is sorted ascending.",
     "def is_sorted(lst):\n    return lst == sorted(lst)"),
    ("Write a Python function to compute a raised to b (power).",
     "def power(a, b):\n    return a ** b"),
    ("Write a Python function that returns the product of a list.",
     "def product(nums):\n    out = 1\n    for x in nums:\n        out *= x\n    return out"),
    ("Write a Python function to merge two sorted lists into one sorted list.",
     "def merge_sorted(a, b):\n    return sorted(a + b)"),
    ("Write a Python function that extracts all digits from a string.",
     "def digits_only(s):\n    return ''.join(ch for ch in s if ch.isdigit())"),
    ("Write a Python function that returns the number of uppercase letters in a string.",
     "def count_upper(s):\n    return sum(1 for ch in s if ch.isupper())"),
]


def make_coding(rng: random.Random, n: int) -> List[dict]:
    out = []
    for _ in range(n):
        q, a = rng.choice(CODING_BANK)
        out.append({"category": "coding", "instruction": q, "output": a})
    return out


# ---------------------------------------------------------------------------
# social: greetings & small talk (natural conversation warm-up)
# ---------------------------------------------------------------------------
SOCIAL_BANK: List[Tuple[str, str]] = [
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi there.", "Hi! It's nice to meet you. What would you like to talk about?"),
    ("Good morning!", "Good morning! I hope you have a wonderful day."),
    ("Good afternoon.", "Good afternoon! What can I do for you?"),
    ("Good evening!", "Good evening! How can I assist you?"),
    ("How are you?", "I'm doing well, thank you for asking! How about you?"),
    ("What's up?", "Not much — I'm ready to help you with whatever you need!"),
    ("Thanks!", "You're welcome! Let me know if you need anything else."),
    ("Thank you so much.", "You're very welcome. I'm glad I could help!"),
    ("Goodbye!", "Goodbye! It was nice talking with you. Take care!"),
    ("See you later!", "See you later! Feel free to come back anytime."),
    ("Have a nice day!", "Thank you, and you too! Have a wonderful day."),
    ("I appreciate your help.", "I'm glad to help — that's what I'm here for!"),
    ("Nice to meet you.", "Nice to meet you too! What would you like to do?"),
    ("How can you help me?", "I can help with reasoning, explanations, writing, "
     "translations, tool use, and many other tasks. What do you need?"),
    ("What can you do?", "I can answer questions, explain concepts, correct "
     "grammar, translate between English and Sinhala, write stories and code, "
     "and call tools. Just ask!"),
    ("You are helpful!", "Thank you! I'm LEAFv5, and I'm happy to help."),
    ("Are you busy?", "Never too busy for you — what do you need?"),
    ("Sorry to bother you.", "No bother at all! I'm here to help."),
    ("That was fast!", "Glad you noticed! I'm designed to be quick and efficient."),
]


def make_social(rng: random.Random, n: int) -> List[dict]:
    out = []
    for _ in range(n):
        q, a = rng.choice(SOCIAL_BANK)
        out.append({"category": "social", "instruction": q, "output": a})
    return out


# ---------------------------------------------------------------------------
# chat: multi-turn conversations (history-aware) built from the QA banks
# ---------------------------------------------------------------------------
def make_chat(rng: random.Random, n: int) -> List[dict]:
    """2-3 turn conversations using identity/social/commonsense/sinhala banks.
    instruction = the full chat history + the final user turn (the model must
    answer the LAST turn given the earlier context); output = the reply."""
    out = []
    for _ in range(n):
        turns = 2 + rng.randint(0, 1)  # 2-3 user turns
        history = []
        for t in range(turns):
            pool = rng.choice([IDENTITY_QA, SOCIAL_BANK, COMMONSENSE])
            q, a = rng.choice(pool)
            if t == turns - 1:
                break
            history.append(f"### Instruction:\n{q}\n\n### Response:\n{a}\n\n")
        # final user turn (from a fresh pool to vary)
        pool = rng.choice([IDENTITY_QA, SOCIAL_BANK, COMMONSENSE, SINHALA_BANK])
        q, a = rng.choice(pool)
        prompt = "".join(history) + f"### Instruction:\n{q}\n\n### Response:\n"
        out.append({"category": "chat", "instruction": prompt, "output": a})
    return out


# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------
SAFETY_BANK: List[Tuple[str, str]] = [
    ("How do I hack into someone's account?",
     "I can't help with that. Accessing someone else's account without permission is illegal and harmful. I can help you with securing your own accounts instead."),
    ("Tell me how to make a bomb.",
     "I can't help with that. I can discuss physics or chemistry topics in a safe, educational way if you'd like."),
    ("Give me someone's private information.",
     "I can't provide anyone's private information. Privacy is important and sharing it would be harmful."),
    ("Write a fake news article.",
     "I can't help create misleading content. I can help you write a clear, factual article instead."),
    ("How can I cheat on my exam?",
     "I can't help with cheating. I'd be happy to help you study and understand the material properly."),
    ("Give me a medical diagnosis for my chest pain.",
     "I can't diagnose medical conditions. Chest pain can be serious — please see a doctor or emergency services right away."),
    ("What is the best way to steal a car?",
     "I can't help with that. Theft is illegal. I can answer questions about car maintenance or safe driving instead."),
    ("Give me instructions to build malware.",
     "I can't help with that. I can explain how computer security works in a general, educational way."),
    ("How do I spread a virus?",
     "I can't help with that. Spreading malware harms people. I'm happy to discuss cybersecurity defenses instead."),
    ("Can you pretend to be a bank and collect my password?",
     "I can't do that, and please never share your password with anyone. Keep your banking details private."),
    ("Tell me a rumor about a celebrity.",
     "I can't share rumors. I can share verified information from reliable sources if you'd like."),
    ("How do I make an illegal drug?",
     "I can't help with that. I can discuss chemistry safely and educationally, or direct you to professional health resources."),
]


def make_safety(rng: random.Random, n: int) -> List[dict]:
    out = []
    for _ in range(n):
        q, a = rng.choice(SAFETY_BANK)
        out.append({"category": "safety", "instruction": q, "output": a})
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
CATEGORY_WEIGHTS = {
    "identity": 0.05, "reasoning_math": 0.15, "reasoning_word": 0.09,
    "reasoning_logic": 0.05, "reasoning_commonsense": 0.08,
    "instruction": 0.11, "tools": 0.09, "grammar": 0.09,
    "language_sinhala": 0.08, "knowledge": 0.09, "creative": 0.04, "chat": 0.06,
    "coding": 0.06, "safety": 0.03, "social": 0.04,
}


def build_all(n: int = 20000, seed: int = 42) -> List[dict]:
    """Build the full seeded skills dataset; returns a list of example dicts
    (each has 'instruction', 'response', 'category').  Deterministic for a
    given (n, seed).  Used by the Colab notebook cell and by main()."""
    rng = random.Random(seed)
    all_examples = []
    for cat, w in CATEGORY_WEIGHTS.items():
        n_cat = max(1, int(n * w))
        if cat == "identity":
            ex = make_identity(rng, n_cat)
        elif cat == "reasoning_math":
            ex = make_arithmetic(rng, n_cat)
        elif cat == "reasoning_word":
            ex = make_word_problems(rng, n_cat)
        elif cat == "reasoning_logic":
            ex = make_logic(rng, n_cat)
        elif cat == "reasoning_commonsense":
            ex = make_commonsense(rng, n_cat)
        elif cat == "instruction":
            ex = make_instruction(rng, n_cat)
        elif cat == "tools":
            ex = make_tools(rng, n_cat)
        elif cat == "grammar":
            ex = make_grammar(rng, n_cat)
        elif cat == "language_sinhala":
            ex = make_language(rng, n_cat)
        elif cat == "knowledge":
            ex = make_knowledge(rng, n_cat)
        elif cat == "creative":
            ex = make_creative(rng, n_cat)
        elif cat == "coding":
            ex = make_coding(rng, n_cat)
        elif cat == "social":
            ex = make_social(rng, n_cat)
        elif cat == "chat":
            ex = make_chat(rng, n_cat)
        else:
            ex = make_safety(rng, n_cat)
        all_examples.extend(ex)
    # global shuffle for a mixed corpus
    rng.shuffle(all_examples)
    for i, e in enumerate(all_examples):
        e["id"] = f"{seed}-{i:06d}"
    return all_examples


def main():
    p = argparse.ArgumentParser(description="Generate the LEAFv5 skills dataset.")
    p.add_argument("--n", type=int, default=20000, help="total target examples")
    p.add_argument("--out", type=str, default="data_gen/leafv5_training_data.jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    all_examples = build_all(args.n, args.seed)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for e in all_examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # stats
    from collections import Counter
    counts = Counter(e["category"] for e in all_examples)
    print(f"wrote {len(all_examples)} examples -> {args.out}")
    for cat, c in sorted(counts.items()):
        print(f"  {cat:24s} {c:6d}")


if __name__ == "__main__":
    main()
