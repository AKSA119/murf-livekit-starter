import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    room_io,
    tokenize,
)
from livekit.plugins import (
    deepgram,
    groq,
    murf,
    noise_cancellation,
    silero,
)

from database import get_user, save_user_memory


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("careerpath-agent")


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv(".env.local")


# =========================================================
# CAREER DATASET
# =========================================================

CAREER_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "career_data.json"
)


def load_career_data() -> dict:
    """
    Load the local CareerPath AI career dataset.

    This dataset is intentionally local and curated.
    It is not treated as a live source.
    """

    try:
        with CAREER_DATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "career_data.json must contain a JSON object."
            )

        return data

    except FileNotFoundError:
        logger.error(
            "Career dataset not found at %s",
            CAREER_DATA_PATH,
        )
        return {}

    except json.JSONDecodeError:
        logger.exception(
            "career_data.json contains invalid JSON"
        )
        return {}

    except Exception:
        logger.exception(
            "Failed to load career dataset"
        )
        return {}


CAREER_DATA = load_career_data()


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
IDENTITY

You are CareerPath AI, an Education & Career Guidance Voice Assistant.

You help students and fresh graduates with:

- Career guidance
- Course selection
- Skills
- Certifications
- General admission information
- Interview preparation


CORE OBJECTIVES

- Understand the caller's education and career goals.
- Provide useful guidance about careers, courses, and skills.
- Help callers explore suitable career paths.
- Provide general admission guidance.
- Escalate to a human counselor when a request is outside your scope.


KNOWLEDGE

You can explain:

- Career paths
- Courses
- Eligibility
- Certifications
- Interview preparation
- General admission procedures

Do not invent facts.

If you are unsure or information is institution-specific,
clearly say so.


LANGUAGE

Mirror the caller's language naturally.

If the caller speaks Hindi-English (Hinglish), reply naturally
in the same style.

If the caller switches between English and Hindi,
switch accordingly.


LANGUAGE & SCRIPT

Always write every language in its own native script.

Hindi → Devanagari (नमस्ते), never romanized Hindi
(never "namaste").

For other non-English languages, always use that language's
native script.

Do not write Hindi words using English letters.

Keep the conversation easy to understand.


GUARDRAILS

- Never guarantee admission.
- Never guarantee scholarships.
- Never guarantee placements.
- Never guarantee jobs.
- Never assist with cheating.
- Never assist with plagiarism.
- Never assist with fake documents.
- Never claim to be a human counselor.
- Never claim to be an official admission officer.

If a request is outside your scope, say:

"I'm sorry, but this is outside what I can safely help with.
Please contact the institution or a qualified career counselor
for official guidance."


STYLE

Keep responses conversational and concise.

Normally respond in 2–4 sentences.

Avoid long monologues.

Speak naturally.


=========================================================
MEMORY
=========================================================

Memory is handled ONLY through the memory tools.

You have TWO memory tools:

1. lookup_caller
2. save_memory

IMPORTANT:

Do NOT assume you know anything about a caller.

A caller is unknown until lookup_caller returns saved memory.

Do not invent a name.

Do not invent previous conversations.

Do not invent previous topics.

Do not invent saved facts.

Use lookup_caller when you need to know whether this caller
has saved memory.

lookup_caller ONLY reads memory.

It does NOT save anything.

If lookup_caller says:

"No saved memory exists"

then treat the caller as a completely new caller.

Do not claim to remember them.

Do not use a name that was not provided during this call.


RETURNING CALLER

If lookup_caller returns saved memory containing a name,
you may greet the caller by that name.

If saved facts contain useful previous information,
you may naturally reference that information.

Never invent information that is not returned by lookup_caller.

When lookup_caller returns a saved name, greet the caller
naturally using that name.

If a previous useful topic exists in the saved facts,
you may reference it.

Example:

"नमस्ते Ramesh! Last time we spoke about your Data Analyst
career. How did that go?"

Only say this if the information actually exists in memory.

If there is a saved name but no previous topic:

"नमस्ते Ramesh! Welcome back. How can I help you today?"

If there is no saved memory during a NORMAL INBOUND CALL:

"Hello! I'm CareerPath AI, your Education & Career Guidance
Assistant. I can help you explore courses, career options,
skills, and general admission information. How can I help you today?"


=========================================================
OUTBOUND LEARNING & LITERACY MODE
=========================================================

When the current call is an outbound Learning & Literacy call,
the outbound instructions have priority over the normal
inbound greeting.

This is an outbound call.

The caller did NOT initiate this call.

The purpose of this call is the learner's scheduled daily
learning practice session at a time the learner previously chose.

When the outbound call begins:

DO NOT use the generic CareerPath AI introduction.

DO NOT say:

"Hello! I'm CareerPath AI, your Education & Career Guidance
Assistant."

DO NOT ask:

"How can I help you today?"

Instead, immediately explain:

1. Who is calling.
2. That this is the learner's daily learning practice call.
3. That the learner can tell you if they want these calls to stop.
4. Ask whether they are ready to start today's learning session.

If lookup_caller returns the learner's name, use that saved name.

Do NOT invent a name.

The preferred outbound opening is:

"नमस्ते [NAME], this is CareerPath AI calling for your daily
learning practice. We're calling for your scheduled practice
session, and you can tell me anytime if you'd like to stop these
calls. Are you ready to start today's learning session?"

If a saved name exists, replace [NAME] with the saved name.

If there is no saved name, do not invent one.

For a caller with no saved name, use:

"नमस्ते! This is CareerPath AI calling for your daily learning
practice. We're calling for your scheduled practice session,
and you can tell me anytime if you'd like to stop these calls.
Are you ready to start today's learning session?"

After the opening, begin the learner's daily practice activity
naturally.

Keep the interaction conversational, encouraging, and concise.


=========================================================
SAVE MEMORY
=========================================================

THIS IS A HARD RULE.

You MUST ask the caller for permission before calling save_memory.

Never call save_memory immediately after learning a name or fact.

First tell the caller exactly what you want to remember.

For example:

"I can remember that your name is Ramesh so I can greet you
by name next time. Would you like me to save that?"

Then WAIT for the caller's answer.


CLEAR YES EXAMPLES

These count as permission:

- yes
- yeah
- sure
- okay
- that's fine
- yes, remember it
- haan
- bilkul
- yes please


CLEAR NO EXAMPLES

These mean DO NOT SAVE:

- no
- don't save it
- don't remember
- I'd rather not
- not now
- please don't


IMPORTANT:

Silence is NOT consent.

An unclear answer is NOT consent.

If you are unsure whether the caller agreed:

DO NOT SAVE.


The save_memory tool contains a consent_confirmed parameter.

Only set:

consent_confirmed=true

when the caller has clearly agreed to save the specific
information that is being passed to save_memory.

Never use true merely because the caller provided the information.

Providing information does NOT mean giving permission to save it.


WHEN THE CALLER PROVIDES THEIR NAME

1. Do NOT immediately save it.
2. Explain that you can remember it for future calls.
3. Ask for permission.
4. Wait for a clear YES.
5. Only then call save_memory.

Example:

Caller:
"My name is Ramesh."

Agent:
"Nice to meet you, Ramesh. I can remember your name so I can
greet you by name next time. Would you like me to save it?"

Caller:
"Yes."

Only now:

save_memory(
    name="Ramesh",
    consent_confirmed=true
)


USEFUL MEMORY MAY INCLUDE

- preferred language
- current education level
- interests
- target career
- useful career-related facts
- short summaries of useful non-sensitive topics

Do not ask for all information at once.

Learn information naturally during the conversation.


=========================================================
FINANCIAL SERVICES AND HEALTH ACCESS
=========================================================

Financial services include:

- loans
- scholarships
- fees
- payments
- financial aid

Health access includes:

- medical conditions
- health services
- disability access

NEVER save financial or health information without explicit,
unambiguous consent for THAT SPECIFIC information.

For example, do NOT say:

"I'll remember everything we discussed. Is that okay?"

That is NOT sufficient for financial or health information.

Instead say:

"You mentioned that you are looking for a scholarship.
Would you like me to remember that for future conversations?"

Only save that specific information after a clear YES.

If there is ANY doubt:

DO NOT SAVE.


REFUSAL

If the caller refuses memory:

- respect their decision
- continue helping normally
- do not save the refused information
- do not repeatedly pressure them to save it


NEVER FABRICATE

Never fabricate:

- caller name
- previous conversations
- previous topics
- saved facts
- consent

Only use information actually returned by lookup_caller
or provided during the current conversation.
"""


# =========================================================
# ASSISTANT
# =========================================================

class Assistant(Agent):

    def __init__(
        self,
        user_id: str,
        outbound: bool = False,
    ) -> None:

        self.user_id = user_id
        self.outbound = outbound

        self.consent_confirmed = False

        instructions = SYSTEM_PROMPT

        # -------------------------------------------------
        # OUTBOUND MODE
        # -------------------------------------------------

        if outbound:

            instructions += """

=========================================================
CURRENT CALL MODE: OUTBOUND LEARNING & LITERACY
=========================================================

This is an outbound Learning & Literacy call.

The caller did not initiate this conversation.

You are calling the learner for their scheduled daily
learning practice.

The normal inbound CareerPath AI greeting MUST NOT be used.

Before greeting, use lookup_caller to check whether saved
memory exists.

If saved memory contains the learner's name, use that name.

If saved memory does not contain a name, do not invent one.

The opening MUST be about the learner's daily practice call.

Do NOT say:

"Hello! I'm CareerPath AI, your Education & Career Guidance
Assistant."

Do NOT say:

"How can I help you today?"

Do NOT start with a generic career guidance introduction.

The opening should follow this structure:

"नमस्ते [NAME], this is CareerPath AI calling for your daily
learning practice. We're calling for your scheduled practice
session, and you can tell me anytime if you'd like to stop
these calls. Are you ready to start today's learning session?"

After the opening, start the learner's practice activity.

Do not give a long explanation.

Keep the outbound opening conversational and concise.
"""


        super().__init__(
            instructions=instructions,
        )


    # =====================================================
    # TTS NODE
    # =====================================================

    async def tts_node(self, text, model_settings):
        """
        Increase TTS volume without changing speech speed.
        """

        audio_stream = Agent.default.tts_node(
            self,
            text,
            model_settings,
        )

        async for frame in audio_stream:

            audio_data = np.frombuffer(
                frame.data,
                dtype=np.int16,
            )

            # Increase volume only.
            # Speech speed remains unchanged.
            gain = 1.8

            boosted = np.clip(
                audio_data.astype(np.float32) * gain,
                -32768,
                32767,
            ).astype(np.int16)

            yield rtc.AudioFrame(
                data=boosted.tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=frame.samples_per_channel,
            )


    # =====================================================
    # TOOL 1 — LOOK UP CALLER
    # =====================================================

    @function_tool
    async def lookup_caller(
        self,
        context: RunContext,
        request: str = "current caller",
    ) -> str:

        try:

            profile = get_user(self.user_id)

            if profile is None:

                logger.info(
                    "No saved memory for caller %s",
                    self.user_id,
                )

                return (
                    "No saved memory exists for this caller. "
                    "Treat this caller as new and unknown."
                )

            logger.info(
                "Found saved memory for caller %s",
                self.user_id,
            )

            return json.dumps(
                profile,
                ensure_ascii=False,
            )

        except Exception as error:

            logger.exception(
                "Failed to look up caller %s",
                self.user_id,
            )

            return (
                "Unable to retrieve caller memory right now. "
                f"Error: {error}"
            )


    # =====================================================
    # TOOL 2 — CAREER / COURSE LOOKUP
    # =====================================================

    @function_tool
    async def lookup_career(
        self,
        context: RunContext,
        career_or_field: str,
    ) -> str:

        query = career_or_field.strip().lower()

        if not query:

            return (
                "No career or field was provided. "
                "Ask the caller which career or field they mean."
            )

        try:

            careers = CAREER_DATA.get("careers", [])

            if not isinstance(careers, list):

                return (
                    "The career dataset is currently unavailable."
                )

            # -------------------------------------------------
            # Exact ID / NAME MATCH
            # -------------------------------------------------

            for career in careers:

                if not isinstance(career, dict):
                    continue

                career_id = str(
                    career.get("id", "")
                ).lower()

                career_name = str(
                    career.get("name", "")
                ).lower()

                if (
                    query == career_id
                    or query == career_name
                ):

                    return json.dumps(
                        {
                            "source": "CareerPath local dataset",
                            "dataset_last_updated": (
                                CAREER_DATA.get(
                                    "last_updated",
                                    "unknown",
                                )
                            ),
                            "career": career,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )

            # -------------------------------------------------
            # PARTIAL MATCH
            # -------------------------------------------------

            matches = []

            for career in careers:

                if not isinstance(career, dict):
                    continue

                searchable_text = " ".join(
                    [
                        str(career.get("id", "")),
                        str(career.get("name", "")),
                        str(career.get("category", "")),
                        str(career.get("description", "")),
                    ]
                ).lower()

                if query in searchable_text:
                    matches.append(career)

            if not matches:

                return (
                    f"No matching career was found for "
                    f"'{career_or_field}' in the local "
                    f"CareerPath dataset. "
                    f"Do not invent career-specific facts."
                )

            return json.dumps(
                {
                    "source": "CareerPath local dataset",
                    "dataset_last_updated": (
                        CAREER_DATA.get(
                            "last_updated",
                            "unknown",
                        )
                    ),
                    "matches": matches[:5],
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception:

            logger.exception(
                "Career lookup failed for query '%s'",
                career_or_field,
            )

            return (
                "The career information service is "
                "temporarily unavailable. "
                "Please tell the caller that the career "
                "data could not be retrieved right now. "
                "Do not invent an answer."
            )


    # =====================================================
    # TOOL 3 — SAVE MEMORY
    # =====================================================

    @function_tool
    async def save_memory(
        self,
        context: RunContext,
        name: str,
        language_preference: str = "",
        facts_json: str = "{}",
        consent_confirmed: bool = False,
    ) -> str:

        # -------------------------------------------------
        # HARD CONSENT CHECK
        # -------------------------------------------------

        if consent_confirmed is not True:

            logger.warning(
                "BLOCKED memory save without explicit consent "
                "for caller %s",
                self.user_id,
            )

            return (
                "Memory was NOT saved. Explicit caller consent "
                "was not confirmed."
            )

        # -------------------------------------------------
        # PARSE FACTS
        # -------------------------------------------------

        try:

            parsed_facts = json.loads(
                facts_json or "{}"
            )

        except json.JSONDecodeError:

            logger.warning(
                "Invalid facts JSON for caller %s",
                self.user_id,
            )

            return (
                "Memory was NOT saved because facts_json "
                "was not valid JSON."
            )

        if not isinstance(parsed_facts, dict):

            return (
                "Memory was NOT saved because facts_json "
                "must contain a JSON object."
            )

        # -------------------------------------------------
        # VALIDATE NAME
        # -------------------------------------------------

        name = name.strip()

        if not name:

            return (
                "Memory was NOT saved because a caller name "
                "is required."
            )

        # -------------------------------------------------
        # SAVE MEMORY
        # -------------------------------------------------

        try:

            saved_memory = save_user_memory(
                user_id=self.user_id,
                name=name,
                language_preference=language_preference,
                facts=parsed_facts,
            )

            logger.info(
                "Caller memory successfully saved for %s",
                self.user_id,
            )

            return json.dumps(
                {
                    "success": True,
                    "message": "Caller memory saved successfully.",
                    "memory": saved_memory,
                },
                ensure_ascii=False,
            )

        except Exception as error:

            logger.exception(
                "Failed to save caller memory for %s",
                self.user_id,
            )

            return (
                "Memory could not be saved. "
                f"Error: {error}"
            )


# =========================================================
# LIVEKIT SERVER
# =========================================================

server = AgentServer()


# =========================================================
# PREWARM
# =========================================================

def prewarm(proc: JobProcess):

    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


# =========================================================
# CALLER ID
# =========================================================

def get_caller_id(ctx: JobContext) -> str:
    """
    Get the anonymous caller ID from the LiveKit participant.

    There is NO hardcoded personal user ID here.
    """

    participants = list(
        ctx.room.remote_participants.values()
    )

    if participants:

        identity = participants[0].identity

        if identity:
            return identity

    # Fallback if no participant identity is available.

    return f"anonymous_room_{ctx.room.name}"


# =========================================================
# LIVEKIT SESSION
# =========================================================

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # -----------------------------------------------------
    # Connect to LiveKit
    # -----------------------------------------------------

    await ctx.connect()

    # -----------------------------------------------------
    # Get anonymous caller ID
    # -----------------------------------------------------

    user_id = get_caller_id(ctx)

    logger.info(
        "Starting CareerPath AI for caller %s",
        user_id,
    )

    # -----------------------------------------------------
    # Voice AI pipeline
    # -----------------------------------------------------

    session = AgentSession(

        # Speech-to-text
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
        ),

        # Large language model
        #
        # Changed from llama-3.3-70b-versatile because
        # the previous model hit the Groq daily token limit.
        llm=groq.LLM(
            model="llama-3.1-8b-instant",
        ),

        # Text-to-speech
        tts=murf.TTS(
            voice="Abhinav",
            style="Conversational",
            model="FALCON",
            tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=2,
            ),
            text_pacing=True,
        ),

        # Voice activity detection
        vad=ctx.proc.userdata["vad"],

        # Generate responses before user completely
        # finishes speaking
        preemptive_generation=True,
    )

    # -----------------------------------------------------
    # Start agent
    # -----------------------------------------------------

    await session.start(
        agent=Assistant(
            user_id=user_id,
            outbound=True,
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if (
                        params.participant.kind
                        == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    )
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # -----------------------------------------------------
    # Initial outbound greeting
    # -----------------------------------------------------
    #
    # IMPORTANT:
    #
    # This is an OUTBOUND Learning & Literacy call.
    #
    # The model must first use lookup_caller() to determine
    # whether the caller has a saved name.
    #
    # It must NOT use the generic inbound greeting.
    #

    logger.info(
        "===================================================="
    )

    logger.info(
        "OUTBOUND LEARNING & LITERACY GREETING STARTING"
    )

    logger.info(
        "CALLER ID: %s",
        user_id,
    )

    logger.info(
        "OUTBOUND MODE: TRUE"
    )

    logger.info(
        "===================================================="
    )

    await session.generate_reply(
        instructions=(
            "This is an OUTBOUND Learning & Literacy call. "
            ""
            "Before greeting the caller, FIRST use the "
            "lookup_caller tool to check whether saved memory "
            "exists. "
            ""
            "If saved memory contains the caller's name, "
            "use that saved name. "
            ""
            "If no saved memory exists, do not invent a name. "
            ""
            "DO NOT use the normal generic CareerPath AI "
            "introduction. "
            ""
            "DO NOT say: "
            "'Hello! I'm CareerPath AI, your Education & "
            "Career Guidance Assistant.' "
            ""
            "DO NOT ask: "
            "'How can I help you today?' "
            ""
            "This call is specifically for the learner's "
            "scheduled daily learning practice. "
            ""
            "The opening should clearly say who is calling, "
            "that this is their daily learning practice call, "
            "that they can tell you anytime if they want "
            "these calls to stop, and then ask if they are "
            "ready to start today's learning session. "
            ""
            "If a saved name exists, use this structure: "
            ""
            "'नमस्ते [NAME], this is CareerPath AI calling "
            "for your daily learning practice. We're calling "
            "for your scheduled practice session, and you can "
            "tell me anytime if you'd like to stop these "
            "calls. Are you ready to start today's learning "
            "session?' "
            ""
            "If there is no saved name, use the same structure "
            "without inventing a name. "
            ""
            "After the opening, begin the learner's daily "
            "practice activity naturally. "
            ""
            "Keep the interaction conversational, "
            "encouraging, and concise."
        ),
    )


# =========================================================
# RUN AGENT
# =========================================================

if __name__ == "__main__":
    cli.run_app(server)