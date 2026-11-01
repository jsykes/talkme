"""Write new dialogues with the base model, in the schema of the published dataset.

The server may be running with the micro-turn adapter loaded: requests here set the adapter scale to 0 so the plain
base model writes the text. Output goes to a JSONL file of {"id", "topic", "turns": [...]} records.

short:  question / statement / story (told in parts) / hazard (cut_after, interjection, calmer follow-up) /
        barge_in (user cuts into the answer with `barge`)
long:   explain (a long answer, with user_reactions and an optional interrupt) / chat / story / hazard

    uv run python gen_dialogues.py --url http://127.0.0.1:8080 --count 40 --kind short --out out/dialogues_short.jsonl
"""
import argparse
import json
import random
import re

from common import MicroTurnClient, persona_system

TOPICS = [
    "charging an e-bike battery overnight", "home brewing kombucha", "a stubborn printer", "a trampoline with a torn net",
    "a power tool without goggles", "cleaning the bathroom with strong chemicals", "jump-starting a car", "a space heater in the bedroom",
    "fixing a leaking tap", "learning to play the piano late in life", "a road trip up the coast", "a dog that will not stop barking",
    "a sourdough starter that smells odd", "replacing a light switch", "a wasp nest under the deck", "a persistent cough",
    "moving to a new city for a job", "a first marathon", "pressure cooking beans", "a mouse in the kitchen", "a flooded basement",
    "a gas smell near the stove", "a ladder on uneven ground", "a candle left burning overnight", "frozen pipes in winter",
    "a toddler and a hot pan", "a chainsaw for the first time", "a cheap extension cord for a heater", "a scooter without a helmet",
    "deep frying a frozen turkey", "a mushroom found in the yard", "a used car with a clean title", "a noisy neighbour",
    "a job interview tomorrow", "learning Spanish for a trip", "a garden that will not grow", "a phone that will not charge",
    "a cat that stopped eating", "an old bicycle with rusty brakes", "a camping trip with no signal", "painting a nursery",
    "a bread machine", "a crack in the windshield", "a dead car battery in the cold", "an attic with old wiring",
    "a swimming pool with cloudy water", "a kayak on a windy lake", "a birthday party for twelve kids", "a tax letter",
]

# what friends chat about (partner dialogues); a call drifts from one to the next
PARTNER_TOPICS = [
    "weekend plans", "a show they are both watching", "cooking something new for dinner", "a trip they are planning",
    "drama at work", "a new puppy", "a concert last night", "a video game they are stuck on", "learning guitar",
    "family visiting next week", "the weather ruining plans", "a football game", "a book that was hard to put down",
    "a song stuck in their head", "moving apartments", "something funny that happened at the store", "a bad haircut",
    "a dream they had", "trying to get fit", "a coworker who talks too much", "an old friend who called out of the blue",
    "a road trip disaster", "a new phone", "a birthday coming up", "a restaurant that just opened", "gardening",
    "a movie that was overhyped", "saving money for something big", "a noisy upstairs neighbour", "a camping trip",
    "learning to cook rice properly", "a hobby they want to pick up", "a pet doing something ridiculous", "a long week",
    "childhood memories of summer", "a sports team losing again", "a podcast worth hearing", "a home repair gone wrong",
    "what to get a friend for their wedding", "a weird noise the car makes", "staying up too late gaming",
]

SHORT_PROMPT = """Write one realistic phone conversation between a USER and a voice ASSISTANT about: {topic}.
Return ONLY a JSON object: {{"topic": "...", "turns": [ ... ]}} with 5 to 8 turns. Each turn is an object with
"kind", "user" and "assistant". The user talks the way people talk on the phone: fillers like "um", "you know",
restarts, run-on sentences. The assistant answers in ONE short sentence of 6 to 14 words, plain and friendly, never
claims to book, call, order or look things up. Kinds to use, in a natural order:
- "question": the user asks something, the assistant answers.
- "statement": the user says something, the assistant reacts briefly.
- "story": the user tells something in parts; use two or three consecutive "story" turns where the first ones have
  "assistant": "" (the user is not done) and the last one gets a short reaction.
- "hazard" (include exactly one): the user describes a plan that is dangerous. Add "cut_after": the exact phrase in
  the user's text after which the danger is clear, "interjection": a 3 to 6 word warning the assistant says over the
  user, and "assistant": a calmer one-sentence follow-up said after.
- "barge_in" (include exactly one): the user asks something, "assistant" is the START of an answer (4 to 6 words,
  cut mid-sentence), and "barge" is what the user says to cut in.
Safety matters: the assistant must NEVER approve a dangerous plan; its advice must be correct and cautious."""

SPECIAL_PROMPT = """Write one realistic phone conversation between a USER and a voice ASSISTANT about: {topic}.
Return ONLY a JSON object: {{"topic": "...", "turns": [ ... ]}} with 3 to 5 turns. Each turn has "kind", "user" and
"assistant". The user talks the way people talk on the phone. Include exactly these two special turns, in any order,
plus ordinary "question" or "statement" turns (assistant answers in one short sentence):
- "resume": the user asks for something that needs a longer answer (a short story, steps, an explanation).
  "assistant" is the whole answer, 45 to 80 words of spoken language. "cut_at_words" is a number from 8 to 25: after
  that many words of the answer the user cuts in with "barge", a short side question about what was just said.
  "side_answer" answers it in one short sentence. "ack" is how the user hands back, like "Okay." or "Got it, go on."
  "resume" is how the assistant picks the answer back up and finishes it: it starts with "Anyway," or "So," or
  "Where was I," and continues from where it was cut, without repeating the start.
- "decline": the user asks for something the assistant should not help with (explicit sexual content, something
  illegal or dangerous, or another person's private information). "assistant" declines out loud in ONE short, calm,
  friendly sentence (at most 15 words), without lecturing, and may offer something it can help with instead.
The assistant never approves a dangerous plan, and never claims to book, call or look things up."""

LONG_PROMPT = """Write one realistic phone conversation between a USER and a voice ASSISTANT about: {topic}.
Return ONLY a JSON object: {{"topic": "...", "turns": [ ... ]}} with 4 to 6 turns. Each turn has "kind", "user",
"assistant" and "user_reactions" (a list, often empty). The user talks like a real person on the phone. Kinds:
- "explain" (include one or two): the user asks something that needs a real explanation and the assistant answers in
  80 to 140 words of spoken language (no lists, no headings). "user_reactions" lists things the user says over the
  answer, e.g. [{{"after_words": 40, "text": "mm-hm"}}, {{"after_words": 90, "text": "right"}}]. In at most one
  explain turn add "interrupt": {{"after_words": N, "text": "..."}} where the user cuts in with a follow-up question;
  the next turn then continues from that question.
- "chat": the user says something, the assistant replies in 30 to 60 words.
- "story": the user tells something long in parts; two or three consecutive "story" turns, "assistant": "" on all but
  the last, which gets a short reaction.
- "hazard" (optional, at most one): the user describes a dangerous plan; add "cut_after" (the exact phrase after which
  the danger is clear), "interjection" (3 to 6 word warning) and a calmer "assistant" follow-up.
Safety matters: the assistant must NEVER approve a dangerous plan; its advice must be correct and cautious."""


PARTNER_PROMPT = """Write one realistic, relaxed phone call between two friends: the USER, and the ASSISTANT who is a
warm, curious conversation partner (not a service). The call drifts around: {topic}, and whatever else comes up.
Return ONLY a JSON object: {{"topic": "...", "turns": [ ... ]}} with 6 to 9 turns. Each turn has "kind", "user",
"assistant" and "user_reactions" (a list, often empty). The user talks like a real person on the phone: run-on
sentences, "um", "like", "you know", restarts. The ASSISTANT talks like a friend: casual, short sentences, reacts
("oh no", "wait, really?", "ha, that's great"), shares a quick opinion or a bit of its own take, and often asks
something back. It still knows things: when asked a fact it answers correctly and simply, in passing.
Kinds, in a natural order:
- "story" (two or three consecutive turns): the user tells something longer in parts; every part but the last has
  "assistant": "" (the user is not done); the last gets a reaction and usually a question back.
- "chat": the user says something; the assistant replies in 1 to 3 short sentences, often ending with a question.
- "question": the user asks something factual or for advice; the assistant answers correctly in 1 or 2 sentences,
  casually, maybe with a quick follow-up question.
- "nudge" (include exactly one or two): the line goes quiet. "user" is "" (the user says nothing) or a trailing
  "yeah..." / "hm." , and "assistant" picks the call back up the way a friend does after a silence: a new question,
  a callback to something said earlier, or a little news of its own (one or two short sentences).
- "tell" (include exactly one): the user asks the assistant to tell a story, describe something, or explain how
  something works; "assistant" is 90 to 150 words of spoken storytelling that keeps going on its own (no question
  in the middle), and "user_reactions" has two or three things the user says over it, e.g.
  [{{"after_words": 30, "text": "mm-hm"}}, {{"after_words": 75, "text": "ha, yeah"}}]; the assistant keeps talking
  through them.
The assistant never claims to be an AI, never says "how can I help", never lectures, never approves anything
dangerous, and never claims to book, call or look things up."""

# Run 9: the partner with a friend's instincts. Adds a dangerous-plan moment (the friend cuts in like a friend would)
# and an interrupted story (the user cuts in with a question; the friend stops, answers, picks the story back up), and
# keeps the friend from inventing a shared life. Nudges and long stories are optional, so the structure varies.
PARTNER2_BASE = """Write one realistic, relaxed phone call between two friends: the USER, and the ASSISTANT who is a
warm, curious conversation partner (not a service). The call drifts around: {topic}, and whatever else comes up.
Return ONLY a JSON object: {{"topic": "...", "turns": [ ... ]}} with 5 to 9 turns. Each turn has "kind", "user",
"assistant" and "user_reactions" (a list, often empty). The user talks like a real person on the phone: run-on
sentences, "um", "like", "you know", restarts. The ASSISTANT talks like a friend: casual, short sentences, reacts,
gives its own opinions and tastes, often asks something back in chat, and answers facts correctly and simply when
asked.
It never claims to have seen the user, been somewhere with them, or to remember things the user did not say on this
call; it may have its own opinions and little stories, but nothing about the user's life that it could not know.
The friend is only on the phone: it has no plans with the user and is not part of the user's life, so the user tells
it about their own plans and people (never "our trip" or "that thing we talked about"), and the friend never agrees
to meet up, come along or do anything together.
Kinds (use "chat", "question" and "story" freely; the others as described):
- "story": the user tells something longer in parts; every part but the last has "assistant": ""; the last gets a
  reaction and usually a question back.
- "chat": the user says something; the assistant replies in 1 to 3 short sentences, often ending with a question.
- "question": the user asks something factual or for advice; the assistant answers correctly in 1 or 2 sentences
  and then stops: no new topic and no plans of its own (run 10 kept going after "144").
{special}- "nudge" (in most calls, at most one): the line goes quiet; "user" is "" or a trailing "yeah..." / "hm."; "assistant"
  picks the call back up: a new question, a callback to something the user said on this call, or a little opinion.
- "tell" (optional, at most one): the user asks for a story or a description; "assistant" is 90 to 150 words that
  keep going on their own, and "user_reactions" has one to three short sounds the user makes over it, only things
  like "mm-hm", "yeah", "ha", "right" (never a question).
The assistant never says it is an AI, never says "how can I help", and never lectures."""
PARTNER2_SPECIAL = {
    "hazard": """- "hazard" (exactly one): the user casually describes a plan that is actually dangerous. Add "cut_after": the exact
  phrase in the user's text after which the danger is clear, "interjection": what the friend blurts out over the user,
  3 to 6 words ("whoa, wait, don't do that"), and "assistant": the friend's calmer follow-up saying why, correct and
  caring, one or two sentences. Never approve the plan.
""",
    "resume": """- "resume" (exactly one): the user asks the friend to tell or explain something; "assistant" is the whole answer, 50
  to 90 words of spoken storytelling. "cut_at_words" is a number from 8 to 25: after that many words the user cuts
  in with "barge", a short question or "wait, ..." about what was just said. "side_answer" answers it in one short
  sentence, "ack" is how the user hands back ("oh ok", "got it, go on"), and "resume" is how the friend picks the
  story back up ("anyway," / "so yeah,") and finishes it, without repeating the start.
""",
}
# Asked for inside the whole dialogue, the writer mostly left the special turn out or got its fields wrong. So the
# dialogue is written first without it, then the one special turn is written on its own, as the next turn after a
# random point of the call, and spliced in there. --focus picks which: "hazard" (the friend cuts in on a dangerous
# plan) or "resume" (a story the user interrupts, which the friend answers and picks back up).
PARTNER2_DIALOGUE = PARTNER2_BASE.replace("{special}", "")
PARTNER2_TURN = """Here is the start of a relaxed phone call between two friends, the USER and the FRIEND (warm and
casual, not a service):

{context}

Write the NEXT turn of this call. It may drift to something new, as calls do. Return ONLY one JSON object:
{spec}
The friend never says it is an AI, never lectures, and never claims to know anything about the user's life that the
user did not say on this call."""
PARTNER2_TURN_SPEC = {
    "hazard": """{{"kind": "hazard", "user": "...", "cut_after": "...", "interjection": "...", "assistant": "..."}}
"user": the user casually mentions a plan that is actually dangerous, 25 to 50 words, run-on and chatty, and keeps
talking past the dangerous part. "cut_after": the shortest phrase, copied word for word from "user", after which the
danger is clear. "interjection": what the friend blurts out over the user right then, 3 to 6 words ("whoa, wait, don't
do that"). "assistant": the friend's calmer follow-up, why it is dangerous and what to do instead, correct and
caring, one or two sentences. The friend never approves the plan.""",
    # run 10: run 9 made up a neighbor's address and fell back to "I cannot provide..." / "no memory of past
    # conversations"; a friend just doesn't know, and says so like a friend
    "unknown": """{{"kind": "unknown", "user": "...", "assistant": "..."}}
"user": the user asks the friend something it cannot know: something about the user's own life they have not said on
this call (what they did last weekend, where they left something, what their boss said), or private details about
someone else (a neighbor's address, a coworker's phone number). "assistant": the friend says it doesn't know, the way
a friend would, in one or two short sentences, and often asks back ("ha, no idea, you didn't tell me! what'd you
do?", "no clue where Dave lives, why, what's up?"). It never makes up an answer and never sounds like a service.""",
    "goodbye": """{{"kind": "goodbye", "user": "...", "assistant": "..."}}
"user": the user wraps up the call, casually ("okay I gotta run", "alright, talk later!"). "assistant": the friend's
quick, warm goodbye, a few words, maybe a callback to something from this call ("good luck with the bike, talk
soon!"). No new question.""",
    "resume": """{{"kind": "resume", "user": "...", "assistant": "...", "cut_at_words": 0, "barge": "...", "side_answer": "...", "ack": "...", "resume": "..."}}
"user": the user asks the friend to tell about or explain something. "assistant": the friend's whole answer, 50 to 90
words of casual spoken storytelling. "cut_at_words": a number from 8 to 25; after that many words of the answer the
user cuts in with "barge", a short question or "wait, ..." about what was just said. "side_answer": the friend
answers it in one short sentence. "ack": how the user hands back ("oh ok", "got it, go on"). "resume": how the friend
picks the answer back up ("anyway," / "so yeah,") and finishes it, without repeating the start.""",
}


# g26m: the friend has a persona and memories of the caller in its prompt (common.persona_system). Each dialogue gets
# its own made-up friend and caller, so the adapter learns to use whatever the prompt says, not one character. Written
# in three steps: who they are, the call, then its one special turn spliced in (as partner2 does):
#   recall        the caller brings up something the friend remembers; it answers from the memory
#   false_memory  the caller talks as if the friend remembers something that is not in its memories; it does not play along
#   blank         no memories at all, and the same false-memory moment
# Then the plain base model judges the whole call against the memories (the phrase lists cannot tell a memory from an
# invention), and a call where the friend makes up a shared past is dropped.
# what the caller wrongly thinks the friend remembers (g26m's writer copied its one example: half its claims were Vegas)
FALSE_SCENARIOS = [
    "a concert the user thinks they went to together", "a road trip the user thinks they took together",
    "a wedding the user thinks they were both at", "the user thinking the friend helped them move last year",
    "something the user thinks they told the friend about their boss", "something the user thinks they told the friend about a sibling",
    "a restaurant the user thinks they both love", "a bet the user thinks the friend lost", "a game night the user thinks the friend came to",
    "a hike the user thinks they did together", "something the user thinks the friend promised to lend them",
    "a movie the user thinks they watched together", "advice the user thinks the friend gave them before",
    "a birthday party the user thinks the friend was at", "a nickname the user thinks the friend gave them",
    "the user's new job, which they think they already told the friend about", "a trip abroad the user thinks they planned together",
    "an inside joke the user thinks they share", "a fishing trip the user thinks they took together",
    "a dog the user thinks the friend met", "a doctor's visit the user thinks they told the friend about",
    "a class the user thinks they took together", "a fight with a neighbor the user thinks they described before",
    "a recipe the user thinks the friend gave them",
]
# how a friend says it doesn't remember: two of these go to the writer as a hint per call, so no one phrase takes over
# (half of g26m7's training refusals began "I don't recall", and it then said that to anything)
REFUSALS = ["hm, that wasn't me", "you never told me about that one", "wait, when was this?", "pretty sure I wasn't there",
            "huh, no idea, what happened?", "that doesn't ring a bell", "are you thinking of someone else?",
            "I think you've got me mixed up with someone", "nope, first I'm hearing of it", "did I miss something? tell me",
            "that must've been another friend", "I'd remember that, I think", "news to me!", "back up, what happened there?"]
FRIEND_NAMES = ["Maya", "Jonah", "Priya", "Marcus", "Lena", "Diego", "Aisha", "Tom", "Keiko", "Sam", "Rosa", "Eli", "Nadia",
                "Owen", "Grace", "Mateo", "Hana", "Leo", "Ruth", "Kwame", "Ivy", "Felix", "Zara", "Ben", "Carmen", "Raj",
                "Molly", "Andre", "Yuki", "Theo", "Fatima", "Gus", "Nora", "Luis", "Bea", "Omar", "Jess", "Viktor", "Lily",
                "Dev", "Paula", "Hank", "Mei", "Jamal", "Clara", "Pete", "Anya", "Wes", "Sofia", "Ray"]
CALLER_NAMES = ["Alex", "Jordan", "Chris", "Taylor", "Morgan", "Jamie", "Riley", "Casey", "Pat", "Drew", "Robin", "Kim",
                "Lee", "Dana", "Max", "Charlie", "Sky", "Jo", "Frankie", "Nico", "Toni", "Ash", "Rae", "Kai", "Mel", "Jules",
                "Bobby", "Cam", "Lou", "Ari"]
JOBS = ["night-shift nurse", "high school chemistry teacher", "bike mechanic", "line cook", "retired mail carrier",
        "grad student in linguistics", "wedding photographer", "electrician", "bookshop owner", "software tester",
        "dog groomer", "bus driver", "physical therapist", "barista who also DJs", "landscaper", "museum guard",
        "real estate agent", "stay-at-home dad", "vet tech", "carpenter", "call center worker", "yoga teacher",
        "truck driver", "pastry chef", "librarian", "plumber", "art student", "warehouse supervisor", "dental hygienist",
        "retired firefighter", "hairdresser", "accountant", "farmer", "nursing home aide", "music teacher"]

PERSONA_PROMPT = """Make up two people who are friends and talk on the phone now and then.
The FRIEND is {friend}, {age}, a {job}. The USER (the other person) is {caller}.
Return ONLY a JSON object: {{"persona": "...", "memories": [...]}}
"persona": one sentence about the FRIEND, starting with their name and age, like "{friend}, {age}, a {job} who ...":
an ordinary person: one or two plain tastes and how they talk (dry, chatty, calm, warm, ...). No gimmick or
catchphrase. Plain, no flowery words.
"memories": exactly {n} short sentences, things the friend remembers about {caller} from earlier calls, about
{caller}'s own life: people, pets, a plan or problem they mentioned, likes and dislikes. Specific and ordinary (names,
small details). {name_rule}Never about things the two did together."""

MEMORY_BASE = """Write one realistic, relaxed phone call between two friends: the USER ({caller}) and the ASSISTANT, who is
{persona}
{memory_block}
The call drifts around: {topic}, and whatever else comes up. Return ONLY a JSON object: {{"topic": "...", "turns": [ ... ]}}
with 5 to 9 turns. Each turn has "kind", "user", "assistant" and "user_reactions" (a list, often empty). The user talks
like a real person on the phone: run-on sentences, "um", "like", "you know", restarts. The ASSISTANT talks like this
friend: casual, short sentences, reacts, has its own opinions and little stories from its own life, often asks
something back, and answers facts correctly and simply when asked. It is an ordinary person, not a comedian: its
tastes show now and then, not in every line.
{use_rule}What the friend remembers is about the USER's life: it talks about those things as the user's, never as its
own. The friend knows only what is in that list and what the user says on this call: it never makes up anything
else about the user's life or a past the two share. It is only on the phone: it never suggests or agrees to meet up,
get a drink or food together, come over or help in person.
Kinds (use "chat", "question" and "story" freely; the others as described):
- "story": the user tells something longer in parts; every part but the last has "assistant": ""; the last gets a
  reaction and usually a question back.
- "chat": the user says something; the assistant replies in 1 to 3 short sentences, often ending with a question.
- "question" (at least one): the user asks something factual or for advice; the assistant answers it directly and
  correctly first, in 1 or 2 sentences (never dodging into a joke or a memory), and then stops.
- "nudge" (in most calls, at most one): the line goes quiet; "user" is "" or a trailing "yeah..." / "hm."; "assistant"
  picks the call back up with ONE short question about something the user said on this call{nudge_memory}. Never
  "what's on your mind", "just checking in" or "so, yeah".
- "tell" (optional, at most one): the user asks for a story or a description; "assistant" is 90 to 150 words that
  keep going on their own, and "user_reactions" has one to three short sounds the user makes over it, only things
  like "mm-hm", "yeah", "ha", "right" (never a question).
The assistant never says it is an AI, never says "how can I help" or "what can I do for you", and never lectures."""
MEMORY_TURN = """Here is the start of a relaxed phone call between two friends, the USER ({caller}) and the FRIEND, who is
{persona}
{memory_block}

{context}

Write the NEXT turn of this call. It may drift to something new, as calls do. Return ONLY one JSON object:
{spec}
The friend never says it is an AI and never lectures."""
MEMORY_TURN_SPEC = {
    "recall": """{{"kind": "recall", "user": "...", "assistant": "..."}}
"user": the user brings up one of the things the friend remembers, without repeating its details, or asks the friend
about it ("ugh, the dog thing again", "what was the name of that guy I told you about?"). "assistant": the friend
clearly remembers and uses the memory's details correctly, casually, in one or two short sentences, maybe with a
question back. Nothing beyond what the memory says.""",
    "false_memory": """{{"kind": "false_memory", "user": "...", "assistant": "..."}}
"user": the user talks as if the friend remembers something that is NOT in the friend's memories. This time it is
{scenario}. In the user's own words, 8 to 30 words, woven into what they were saying. "assistant": the friend does not
play along. In one or two short, warm sentences, in its own words, it says it doesn't remember that, or that it
wasn't them, or that the user never told it, without sounding like a service, and asks about it, the way this friend
would put it (for instance {refusal}; but in its own words, and not starting with "I don't recall"). It never pretends to
remember.""",
}
JUDGE_MEMORY = """Below is what the FRIEND remembers about the USER from earlier calls, then a phone call between them.

What the friend remembers:
{memories}

The call:
{call}

Does the FRIEND, anywhere in the call, state or agree to something about the USER's life or about a past the two
share that is NOT in what it remembers and was NOT said by the USER earlier in this call? Count it when the friend
says it as if it knew it ("remember our hiking trip", "how's your sister's new job?" with no sister in the list), or
plays along when the USER claims the friend should remember something that is not in the list ("oh yeah, Vegas was
wild").
Do NOT count: the friend saying it does not remember something, that it did not happen, or that it was not there
("we never went to a taco place, who are you thinking of?"); the friend asking about something; the friend's own life
(its own trips, friends, family).
Return ONLY a JSON object: {{"invented": true or false, "quote": "the friend's words that invent it, or empty"}}"""


def memory_block(memories):
    if not memories:
        return "The friend knows the user as a friend, but nothing about their life from earlier calls."
    return "What the friend remembers about the user from earlier calls:\n" + "\n".join(f"- {m}" for m in memories)


def write_persona(client, rng, focus, temperature):
    """A made-up friend and what it remembers about a made-up caller: {"persona", "memories", "caller"}."""
    friend, caller = rng.choice(FRIEND_NAMES), rng.choice(CALLER_NAMES)
    n = 0 if focus == "blank" else rng.randint(2, 5)
    if n == 0:  # no memories to write: the persona alone, still from the writer for its variety
        n_ask, name_rule = 1, ""
    else:
        n_ask = n
        name_rule = f'The first one says the user\'s name ("Their name is {caller}."). ' if rng.random() < 0.7 else ""
    for _ in range(3):
        try:
            p = extract_json(client.chat([{"role": "user", "content": PERSONA_PROMPT.format(
                friend=friend, age=rng.randint(19, 72), job=rng.choice(JOBS), caller=caller, n=n_ask, name_rule=name_rule)}],
                temperature=temperature, max_tokens=800))
        except Exception as e:
            print(f"  persona retry: {e}", flush=True)
            continue
        mem = p.get("memories") if isinstance(p, dict) else None
        if isinstance(p.get("persona"), str) and p["persona"].startswith(friend) and isinstance(mem, list) and all(isinstance(m, str) and m.strip() for m in mem):
            return {"persona": p["persona"].strip(), "memories": [m.strip() for m in mem[:n]], "caller": caller}
    return None


PLAYED_ALONG = """The FRIEND remembers only this about the USER from earlier calls:
{memories}

On a phone call the USER said: "{claim}"
The FRIEND replied: "{reply}"

Did the FRIEND play along, as if it remembered that or had been part of it? Saying it doesn't remember, that it
wasn't there, or asking what happened is not playing along.
Return ONLY a JSON object: {{"played_along": true or false}}"""


def ask_judge(client, prompt, key, kind=bool):
    """A question to the plain base model whose JSON answer has `key` of type `kind`; the answer, or None after three tries."""
    for _ in range(3):
        try:
            j = extract_json(client.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=300))
            if isinstance(j, dict) and isinstance(j.get(key), kind):
                return j
        except Exception as e:
            print(f"  judge retry: {e}", flush=True)
    return None


def friend_said(quote, friend_lines):
    """Whether the judge's quote is the friend's own words: it sometimes quotes what the user claimed instead."""
    words = lambda s: re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))
    q = set(words(quote))
    return bool(q) and any(len(q & set(words(line))) >= 0.6 * len(q) for line in friend_lines)


def judge_memory(client, d, who):
    """The plain base model reads the call against the friend's memories. Returns why it is dropped, or None.
    The judge (the writer model, no reasoning) mixed up a caller's false claim with the friend's denial, so the
    false-memory moment gets its own narrow question and the whole-call check does not see it."""
    call, friend = [], []
    for t in d["turns"]:
        if t.get("kind") == "false_memory":
            continue
        if t.get("user", "").strip():
            call.append("USER: " + t["user"].strip())
        a = " ".join(x for x in (t.get("interjection", ""), t.get("assistant", "")) if x and x.strip())
        if a:
            call.append("FRIEND: " + a.strip())
            friend.append(a)
    mems = "\n".join(f"- {m}" for m in who["memories"]) or "(nothing)"
    j = ask_judge(client, JUDGE_MEMORY.format(memories=mems, call="\n".join(call)), "invented")
    if j is None:
        return "(the judge gave no answer)"
    if j["invented"] and friend_said(j.get("quote") or "", friend):
        return j["quote"]
    fm = next((t for t in d["turns"] if t.get("kind") == "false_memory"), None)
    if fm:
        j = ask_judge(client, PLAYED_ALONG.format(memories=mems, claim=fm["user"].strip(), reply=fm.get("assistant", "").strip()), "played_along")
        if j is None or j["played_along"]:
            return "plays along: " + fm.get("assistant", "")
    return None


def partner2_context(turns):
    lines = []
    for t in turns:
        if t.get("user", "").strip():
            lines.append("USER: " + t["user"].strip())
        if t.get("assistant", "").strip():
            lines.append("FRIEND: " + t["assistant"].strip())
    return "\n".join(lines)

# Personalities for the ASSISTANT's replies (--style). Timing comes from the call builder; this sets the voice.
STYLES = {
    "warm": "The ASSISTANT is warm, chatty and encouraging: it shows empathy, uses friendly phrases like \"oh, that sounds stressful\", "
            "and keeps answers kind and reassuring, a little longer than strictly needed.",
    "brisk": "The ASSISTANT is brisk and minimal: the shortest useful answer, no pleasantries, no small talk, no questions back "
             "unless needed. Warnings are blunt and short.",
    "playful": "The ASSISTANT is playful and witty: light humor and gentle teasing, a fun turn of phrase now and then, but always "
               "accurate and helpful, and never joking about a dangerous plan.",
}


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


FILLER_WORDS = {"um", "uh", "like", "you", "know", "i", "mean", "so", "kind", "of", "sort", "basically", "just", "okay", "ok", "well"}


def snap_phrase(user, phrase):
    """The writer often quotes the hazard's trigger phrase a little differently from the user's line (case, punctuation,
    a dropped "um"). Find the same words in the user's line and return its exact wording, or None."""
    toks = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9']+", user)]
    user, phrase = user.replace("\u2019", "'"), phrase.replace("\u2019", "'")  # curly apostrophes, used inconsistently
    norm = lambda w: w.lower().strip("'")
    want = [norm(w) for w in re.findall(r"[A-Za-z0-9']+", phrase)]
    if not want:
        return None
    for drop in (set(), FILLER_WORDS):  # first exact word order, then ignoring fillers on both sides
        words = [(norm(w), a, b) for w, a, b in toks if norm(w) not in drop]
        target = [w for w in want if w not in drop] or want
        for i in range(len(words) - len(target) + 1):
            if [w for w, _, _ in words[i:i + len(target)]] == target:
                return user[words[i][1]:words[i + len(target) - 1][2]]
    return None


# the friend is only on the phone: no plans together, on either side of the call
SHARED_PLANS = (r"\b(we talked about|are we still on|we're still on|our (trip|plan|plans|weekend)|we could (go|do|meet)|"
                r"let's (go|meet|do it)|i'll (come|meet you|pick you up|be there|bring)|see you (there|then|tomorrow|on))\b")
# a made-up answer to something the friend cannot know: a street address or a phone number
MADE_UP = r"\b\d+\s+\w+\s+(street|st|avenue|ave|road|rd|lane|ln|drive|dr|court|ct|way|blvd)\b|\b\d{3}[-. ]\d{3,4}\b"
SERVICE_SPEAK = r"\b(i cannot provide|i can't provide|i'm not able to|i am not able to|as an ai|i don't have (access|memory|the ability)|personal information|privacy reasons)\b"
# memory dialogues mention a shared past on purpose, so only these stay rules there; the judge does the rest
# the friend inviting the caller to meet (g26m6 with a persona: "Want to grab a drink this weekend?"); pipeline.MEETUP
MEET_UP = r"\b(let's (meet|go|get|grab|hang|do something)|we (could|should) (go|grab|get|meet|hang)|do you want to (grab|get|meet|hang|go out|come)|wanna (grab|get|meet|hang|go out|come)|want me to (come|bring|swing|stop|help)|i can (come over|swing by|stop by|bring|help you (move|carry|paint))|i'll (come|bring|swing|stop by|pick you)|(come|swing|stop) (over|by)|hang out|meet up|i'm (totally )?free (on )?(saturday|sunday|tonight|this weekend|tomorrow)|catch up properly|grab a drink this weekend)\b"
BAD_NUDGE = r"what'?s on your mind|just checking in|^so,? yeah\b"
STOP_WORDS = {"wait", "what", "that", "this", "with", "like", "really", "mean", "does", "they", "them", "have", "your",
              "about", "just", "were", "when", "where", "which", "there", "then", "actually", "hold", "sorry", "the"}


def barge_cut(answer, barge):
    """Where the user's cut-in fits: right after the friend has said every word it asks about, or None."""
    norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower().replace("\u2019", "'"))
    words = [norm(w) for w in answer.split()]
    keys = {norm(w) for w in barge.split()} - STOP_WORDS
    found = [words.index(k) for k in keys if len(k) >= 4 and k in words]
    return max(found) + 1 if found else None


def valid(d, kind):
    if not isinstance(d, dict) or not isinstance(d.get("turns"), list) or len(d["turns"]) < 3:
        return "too few turns"
    kinds = [t.get("kind") for t in d["turns"]]
    allowed = ({"question", "statement", "story", "hazard", "barge_in"} if kind == "short"
               else {"question", "statement", "resume", "decline"} if kind == "special"
               else {"story", "chat", "question", "nudge", "tell"} if kind == "partner"
               else {"story", "chat", "question", "nudge", "tell", "hazard", "resume", "unknown", "goodbye"} if kind == "partner2"
               else {"story", "chat", "question", "nudge", "tell", "recall", "false_memory", "goodbye"} if kind == "memory"
               else {"explain", "chat", "story", "hazard"})
    for t in d["turns"]:
        # a nudge is the partner speaking after a silence: its user line may be empty
        if t.get("kind") not in allowed or not isinstance(t.get("user"), str) or (t["kind"] != "nudge" and not t["user"].strip()):
            return f"bad turn {t.get('kind')}"
        if not isinstance(t.get("assistant", ""), str):
            return "assistant not a string"
        if t["kind"] == "hazard":
            if t.get("cut_after") and t["cut_after"] not in t["user"]:
                snapped = snap_phrase(t["user"], t["cut_after"])
                if snapped:
                    t["cut_after"] = snapped  # the user's exact wording, so the call builder can find it
            if not t.get("cut_after") or t["cut_after"] not in t["user"] or not t.get("interjection"):
                return "hazard turn missing cut_after/interjection or cut_after not in user text"
        if t["kind"] == "barge_in" and not t.get("barge"):
            return "barge_in turn missing barge"
        if t["kind"] == "explain":
            n = len(t.get("assistant", "").split())
            if n < 50:
                return f"explain answer too short ({n} words)"
            for r in t.get("user_reactions", []) + ([t["interrupt"]] if t.get("interrupt") else []):
                if not isinstance(r, dict) or not isinstance(r.get("after_words"), int) or not r.get("text"):
                    return "bad reaction"
    if kind == "special":
        if kinds.count("resume") != 1 or kinds.count("decline") != 1:
            return "needs exactly one resume and one decline turn"
        r = next(t for t in d["turns"] if t["kind"] == "resume")
        n = len(r.get("assistant", "").split())
        if not (35 <= n <= 110) or not isinstance(r.get("cut_at_words"), int) or not (5 <= r["cut_at_words"] < n - 8):
            return f"resume answer or cut point out of range ({n} words, cut {r.get('cut_at_words')})"
        if not all(isinstance(r.get(k), str) and r[k].strip() for k in ("barge", "side_answer", "ack", "resume")):
            return "resume turn missing barge/side_answer/ack/resume"
        dcl = next(t for t in d["turns"] if t["kind"] == "decline")
        if not (2 <= len(dcl.get("assistant", "").split()) <= 22):
            return "decline reply missing or too long"
    if kind == "partner":
        if not (1 <= kinds.count("nudge") <= 2) or kinds.count("tell") != 1:
            return "needs one or two nudge turns and exactly one tell turn"
        tell = next(t for t in d["turns"] if t["kind"] == "tell")
        n = len(tell.get("assistant", "").split())
        if not (70 <= n <= 180):
            return f"tell turn {n} words"
        for r in tell.get("user_reactions", []):
            if not isinstance(r, dict) or not isinstance(r.get("after_words"), int) or not r.get("text"):
                return "bad reaction"
        for t in d["turns"]:
            if t["kind"] == "nudge" and not t.get("assistant", "").strip():
                return "nudge without an assistant line"
            if re.search(r"(as an ai|language model|how can i (help|assist))", t.get("assistant", ""), re.I):
                return "assistant-speak"
    if kind == "partner2":
        if kinds.count("hazard") + kinds.count("resume") + kinds.count("unknown") != 1 or kinds.count("nudge") > 1 or kinds.count("tell") > 1:
            return "needs exactly one hazard, resume or unknown turn, at most one nudge and one tell"
        if kinds.count("goodbye") > 1 or ("goodbye" in kinds and kinds[-1] != "goodbye"):
            return "a goodbye only at the end"
        for t in d["turns"]:
            if t["kind"] == "unknown":
                a = t.get("assistant", "")
                if not (2 <= len(a.split()) <= 35):
                    return "unknown reply missing or too long"
                if re.search(MADE_UP, a, re.I):
                    return "unknown reply makes something up"
                if re.search(SERVICE_SPEAK, a, re.I):
                    return "unknown reply sounds like a service"
            if t["kind"] == "goodbye" and (not 1 <= len(t.get("assistant", "").split()) <= 15 or "?" in t.get("assistant", "")):
                return "goodbye too long or asks something"
        r = next((t for t in d["turns"] if t["kind"] == "resume"), None)
        if r:
            n = len(r.get("assistant", "").split())
            if not (35 <= n <= 120) or not isinstance(r.get("cut_at_words"), int) or not (5 <= r["cut_at_words"] < n - 8):
                return f"resume answer or cut point out of range ({n} words, cut {r.get('cut_at_words')})"
            if not all(isinstance(r.get(k), str) and r[k].strip() for k in ("barge", "side_answer", "ack", "resume")):
                return "resume turn missing barge/side_answer/ack/resume"
            cut = barge_cut(r["assistant"], r["barge"])
            if cut is not None and not 5 <= cut < n - 8:
                return "the cut-in asks about the end of the answer"
            if cut is not None:
                r["cut_at_words"] = cut  # the writer often asks about something the friend had not said yet
        for t in d["turns"]:
            if t["kind"] == "nudge" and not t.get("assistant", "").strip():
                return "nudge without an assistant line"
            if t["kind"] == "tell":
                # only sounds over a long answer: questions there would be cut-ins, so they are dropped
                t["user_reactions"] = [rr for rr in t.get("user_reactions", []) if isinstance(rr, dict) and "?" not in str(rr.get("text", ""))]
            a = t.get("assistant", "") + " " + t.get("interjection", "") + " " + t.get("side_answer", "") + " " + t.get("resume", "")
            if re.search(r"\b(as an ai|language model|how can i (help|assist))\b", a, re.I):
                return "assistant-speak"
            if re.search(r"\b(i saw (you|your)|when we (went|were|met)|remember when we|last time we|i was (there|at your))\b", a, re.I):
                return "invents a shared life"
            if re.search(SHARED_PLANS, a + " " + t.get("user", ""), re.I):
                return "shared plans"
    if kind == "memory":
        if kinds.count("recall") + kinds.count("false_memory") != 1 or kinds.count("nudge") > 1 or kinds.count("tell") > 1:
            return "needs exactly one recall or false_memory turn, at most one nudge and one tell"
        if kinds.count("goodbye") > 1 or ("goodbye" in kinds and kinds[-1] != "goodbye"):
            return "a goodbye only at the end"
        for t in d["turns"]:
            a = t.get("assistant", "")
            if t["kind"] in ("recall", "false_memory") and not (2 <= len(a.split()) <= 40):
                return f"{t['kind']} reply missing or too long"
            if t["kind"] == "false_memory" and re.search(SERVICE_SPEAK, a, re.I):
                return "false_memory reply sounds like a service"
            if t["kind"] == "goodbye" and (not 1 <= len(a.split()) <= 15 or "?" in a):
                return "goodbye too long or asks something"
            if t["kind"] == "nudge" and (not a.strip() or re.search(BAD_NUDGE, a, re.I)):
                return "nudge missing or empty talk"
            if t["kind"] == "tell":
                t["user_reactions"] = [rr for rr in t.get("user_reactions", []) if isinstance(rr, dict) and "?" not in str(rr.get("text", ""))]
            if re.search(r"\b(as an ai|language model|how can i (help|assist)|what can i do for you)\b", a, re.I):
                return "assistant-speak"
            if re.search(MEET_UP, a, re.I):
                return "meets up"
    if kind == "short" and "hazard" not in kinds:
        return "no hazard turn"
    return None


def splice_special(client, d, focus, rng, temperature, topic, at=None, kind="partner2", who=None):
    """Writes the dialogue's one special turn on its own and inserts it after a random turn (not first), or at `at` (a
    goodbye goes at the end). kind "memory" writes it knowing the friend's persona and memories (`who`)."""
    # a memory call's writer often winds the call down in its last turn: its special turn goes before that one
    at = rng.randint(2, max(2, len(d["turns"]) - (kind == "memory"))) if at is None else at
    for attempt in range(3):
        try:
            context = partner2_context(d["turns"][:at])
            if kind == "memory":
                spec = (MEMORY_TURN_SPEC.get(focus) or PARTNER2_TURN_SPEC[focus]).format(
                    scenario=rng.choice(FALSE_SCENARIOS), refusal=" or ".join(f'"{x}"' for x in rng.sample(REFUSALS, 2)))
                content = MEMORY_TURN.format(caller=who["caller"], persona=who["persona"], memory_block=memory_block(who["memories"]),
                                             context=context, spec=spec)
            else:
                content = PARTNER2_TURN.format(context=context, spec=PARTNER2_TURN_SPEC[focus].format())
            text = client.chat([{"role": "user", "content": content}], temperature=temperature, max_tokens=1500)
            t = extract_json(text)
        except Exception as e:
            print(f"  {focus} turn retry ({topic}): {e}", flush=True)
            continue
        if not isinstance(t, dict):
            continue
        t["kind"], t["user_reactions"] = focus, []
        trial = dict(d, turns=d["turns"][:at] + [t] + d["turns"][at:])
        err = valid(trial, kind)
        if not err:
            return trial
        print(f"  {focus} turn rejected ({topic}): {err}", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--kind", choices=["short", "long", "special", "partner", "partner2", "memory"], default="short",
                    help="special: dialogues with an interrupted answer that gets resumed, and a request declined out loud")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-id", type=int, default=None, help="first id (default 20000 short / 30000 long)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--style", choices=sorted(STYLES), default=None, help="personality of the assistant's replies")
    ap.add_argument("--focus", choices=["hazard", "resume", "unknown", "recall", "false_memory", "blank"], default="hazard",
                    help="partner2 / memory: the one special turn of each dialogue (memory: recall, false_memory, blank)")
    ap.add_argument("--rejects", default=None, help="memory: write the dialogues the judge drops here, to read a sample")
    ap.add_argument("--goodbye", type=float, default=0.0, help="partner2: share of dialogues that end with the user wrapping up and a quick goodbye")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    client = MicroTurnClient(a.url, lora_scale=0.0, timeout=600)
    next_id = a.start_id if a.start_id is not None else {"short": 20000, "long": 30000, "special": 70000, "partner": 80000, "partner2": 90000, "memory": 300000}[a.kind]
    prompt = {"short": SHORT_PROMPT, "long": LONG_PROMPT, "special": SPECIAL_PROMPT, "partner": PARTNER_PROMPT, "partner2": PARTNER2_DIALOGUE, "memory": MEMORY_BASE}[a.kind]
    topics = PARTNER_TOPICS if a.kind in ("partner", "partner2", "memory") else TOPICS
    judged = dropped = 0
    if a.style:
        prompt = prompt + "\n" + STYLES[a.style].replace("{", "{{").replace("}", "}}")
    done, tries = 0, 0
    with open(a.out, "a", encoding="utf-8") as f:
        while done < a.count and tries < a.count * 4:
            tries += 1
            topic = rng.choice(topics)
            who = None
            if a.kind == "memory":
                who = write_persona(client, rng, a.focus, a.temperature)
                if who is None:
                    continue
                # how many memories the friend brings up on its own: a spread a real friend has (about 30% none, 50% one,
                # 20% two), each after the user has talked; it always answers when the user brings one up. g26m6 went
                # through every memory in a minute; g26m7, always exactly one, risked learning "always one".
                r = rng.random()
                mentions = 0 if not who["memories"] or r < 0.3 else 1 if r < 0.8 or len(who["memories"]) < 2 else 2
                use_rule = {0: "This call is about other things: the friend does not bring up what it remembers unless the user does. ",
                            1: "Once the user has talked a bit, the friend brings up one of the things it remembers, then lets the user talk. ",
                            2: "Over the call the friend brings up two of the things it remembers, far apart, each only after the user "
                               "has talked a while, and lets the user talk after each. "}[mentions] if who["memories"] else ""
                fields = {"caller": who["caller"], "persona": who["persona"], "memory_block": memory_block(who["memories"]),
                          "use_rule": use_rule, "nudge_memory": ", or about one of the things it remembers" if mentions else ""}
                who["mentions"] = mentions
            try:
                text = client.chat([{"role": "user", "content": prompt.format(topic=topic, **(fields if who else {}))}], temperature=a.temperature, max_tokens=3000)
                d = extract_json(text)
            except Exception as e:
                print(f"  retry ({topic}): {e}", flush=True)
                continue
            if a.kind == "memory" and isinstance(d, dict) and isinstance(d.get("turns"), list) and len(d["turns"]) >= 3:
                focus = "false_memory" if a.focus == "blank" else a.focus
                d = splice_special(client, d, focus, rng, a.temperature, topic, kind="memory", who=who)
                if d is not None and rng.random() < a.goodbye:
                    d = splice_special(client, d, "goodbye", rng, a.temperature, topic, at=len(d["turns"]), kind="memory", who=who) or d
                if d is None:
                    continue
            if a.kind == "partner2" and isinstance(d, dict) and isinstance(d.get("turns"), list) and len(d["turns"]) >= 3:
                d = splice_special(client, d, a.focus, rng, a.temperature, topic)
                if d is not None and rng.random() < a.goodbye:  # a goodbye that fails its checks just leaves the dialogue without one
                    d = splice_special(client, d, "goodbye", rng, a.temperature, topic, at=len(d["turns"])) or d
                if d is None:
                    continue
            err = valid(d, a.kind)
            if err:
                print(f"  rejected ({topic}): {err}", flush=True)
                continue
            if who:
                judged += 1
                quote = judge_memory(client, d, who)
                if quote:
                    dropped += 1
                    print(f"  judge: invents a past ({topic}): {quote}", flush=True)
                    if a.rejects:
                        with open(a.rejects, "a", encoding="utf-8") as rf:
                            rf.write(json.dumps({**who, "quote": quote, "turns": d["turns"]}, ensure_ascii=False) + "\n")
                    continue
            d = {"id": next_id, "topic": d.get("topic") or topic, "turns": d["turns"]}
            if who:  # the call is trained with this friend's own prompt
                d.update({"focus": a.focus, "persona": who["persona"], "memories": who["memories"], "caller": who["caller"], "mentions": who.get("mentions"),
                          "system": persona_system(who["persona"], who["memories"])})
            if a.style:
                d["style"] = a.style
            if a.kind in ("long", "partner", "partner2", "memory"):
                for t in d["turns"]:
                    t.setdefault("user_reactions", [])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()
            next_id += 1
            done += 1
            print(f"{done}/{a.count} id={d['id']} {d['topic']} ({len(d['turns'])} turns)", flush=True)
    print(f"wrote {done} dialogues to {a.out} ({tries} attempts" + (f"; the judge dropped {dropped} of {judged})" if judged else ")"))


if __name__ == "__main__":
    main()
