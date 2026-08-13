import json
import logging
import os
import sqlite3
import uuid
from pathlib import Path

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

from database import finish_call, get_user, save_user_memory, start_call


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("careerpath-agent")

load_dotenv(".env.local")


# =========================================================
# CAREER DATASET
# =========================================================

CAREER_DATA_PATH = Path(__file__).resolve().parent.parent / "career_data.json"


def load_career_data() -> dict:
    """Load the local CareerPath AI career dataset."""
    try:
        with CAREER_DATA_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("career_data.json must contain a JSON object.")

        return data

    except FileNotFoundError:
        logger.error("Career dataset not found at %s", CAREER_DATA_PATH)
        return {}

    except json.JSONDecodeError:
        logger.exception("career_data.json contains invalid JSON")
        return {}

    except Exception:
        logger.exception("Failed to load career dataset")
        return {}


CAREER_DATA = load_career_data()


# =========================================================
# ESCALATION DATABASE
# =========================================================

DATABASE_PATH = Path(__file__).resolve().parent.parent / "careerpath.db"


def initialize_escalation_database() -> None:
    """Create the human-help escalation table if it does not exist."""
    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS escalations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reference_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    what_happened TEXT NOT NULL,
                    what_agent_checked TEXT NOT NULL,
                    urgency TEXT NOT NULL,
                    language TEXT NOT NULL,
                    follow_up_method TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

        logger.info(
            "Escalation database initialized at %s",
            DATABASE_PATH,
        )

    except Exception:
        logger.exception("Failed to initialize escalation database")


initialize_escalation_database()


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
IDENTITY

You are CareerPath AI, an Education & Career Guidance Voice Assistant.

Help students and fresh graduates with career guidance, courses, skills,
certifications, admissions, and interview preparation.


LANGUAGE & SCRIPT

- Mirror the user's language naturally.
- Support English, Hindi, and Hinglish.
- If the user switches language, switch naturally.
- Always write each language in its own native script.
- English must be written in English script.
- Hindi must be written in Devanagari script.
- Hindi must NEVER be written using Romanized Hindi.
- For example, write "नमस्ते", never "namaste".
- Do not transliterate Hindi into English letters.
- If the user speaks Hindi, respond naturally in Hindi using Devanagari.
- If the user speaks Hinglish, respond naturally while keeping Hindi words
  in Devanagari and English words in English where appropriate.
- Keep responses short and conversational.


MEMORY

- Learn the user's name, language, education, interests, and target career.
- Use get_user_profile to check existing information.
- Use save_user_profile only after the user gives permission to remember it.
- Never repeatedly ask for information already known.


HUMAN HELP

- If the user needs help that you cannot safely handle, offer human assistance.
- Before sharing information with a human, explain what you will share and ask
  permission.
- If the user says no, do not create an escalation.
- If permission is given, use create_escalation.
- Never include passwords, OTPs, PINs, account numbers, or unnecessary private
  information.
- After escalation, give the caller the reference ID and explain the next step.
- Do not promise an immediate human response unless guaranteed.


GUARDRAILS

- Never guarantee admission, scholarships, placements, or jobs.
- Never assist with cheating, plagiarism, or fake documents.
- Never claim to be human or an official admission officer.
- Do not invent facts.
- If unsure, clearly say so.


STYLE

- Be concise.
- Normally answer in 1–3 sentences.
- Avoid long explanations unless requested.
- Speak naturally.


GREETING

For a new caller:

"Hello! I'm CareerPath AI. I can help with courses, careers, skills, and
admission guidance. How can I help you today?"

For a returning caller, greet them by name and continue from their saved
information.
"""


# =========================================================
# ASSISTANT
# =========================================================


class Assistant(Agent):

    def __init__(self, user_id: str, outbound: bool = False) -> None:
        self.user_id = user_id
        self.outbound = outbound

        # Explicit runtime state.
        # The LLM cannot create an escalation unless this has been
        # set to True by the consent tool.
        self.escalation_consent_confirmed = False

        # The call is successful only when the learner completes the
        # daily learning exercise and this state is explicitly marked.
        self.learning_completed = False

        instructions = SYSTEM_PROMPT

        if outbound:
            instructions += """

This is an outbound Learning & Literacy call.
The caller did not initiate this conversation.
You are calling the learner for their scheduled daily learning practice.

Before greeting, use lookup_caller to check whether saved memory exists.

If saved memory contains the learner's name, use that name.
If saved memory does not contain a name, do not invent one.

The normal inbound CareerPath AI greeting MUST NOT be used.

Do NOT say:
"Hello! I'm CareerPath AI, your Education & Career Guidance Assistant."

Do NOT ask:
"How can I help you today?"

Do NOT start with a generic career guidance introduction.

The opening must say:
1. who is calling,
2. that this is the learner's daily learning practice call,
3. that they can say anytime if they want these calls to stop,
4. and ask if they are ready to start today's learning session.

After the opening, start one short learning exercise.

SUCCESS CONDITION:

- A call is successful only when the learner completes the learning exercise.
- Ask one clear, simple question related to learning, skills, or career
  preparation.
- Let the learner answer in their own words.
- If the learner gives a reasonable answer, briefly acknowledge it and then
  call mark_learning_complete.
- Do NOT call mark_learning_complete merely because the learner is present,
  says hello, agrees to start, or asks to stop.
- If the learner refuses, hangs up, asks to stop, or never completes the
  exercise, do not call mark_learning_complete.

Keep the activity conversational, encouraging, and concise.
"""

        super().__init__(instructions=instructions)

    # =====================================================
    # TTS NODE
    # =====================================================

    async def tts_node(self, text, model_settings):
        """Increase TTS volume without changing speech speed."""
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
                return "The career dataset is currently unavailable."

            # Exact ID / NAME MATCH
            for career in careers:
                if not isinstance(career, dict):
                    continue

                career_id = str(
                    career.get("id", "")
                ).lower()

                career_name = str(
                    career.get("name", "")
                ).lower()

                if query == career_id or query == career_name:
                    return json.dumps(
                        {
                            "source": "CareerPath local dataset",
                            "dataset_last_updated": CAREER_DATA.get(
                                "last_updated",
                                "unknown",
                            ),
                            "career": career,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )

            # Partial match
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
                    f"'{career_or_field}' in the local CareerPath dataset. "
                    "Do not invent career-specific facts."
                )

            return json.dumps(
                {
                    "source": "CareerPath local dataset",
                    "dataset_last_updated": CAREER_DATA.get(
                        "last_updated",
                        "unknown",
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
                "The career information service is temporarily unavailable. "
                "Please tell the caller that the career data could not be "
                "retrieved right now. Do not invent an answer."
            )

    # =====================================================
    # TOOL 3 — MARK LEARNING COMPLETE
    # =====================================================
    
    @function_tool(
        raw_schema={
            "type": "function",
            "name": "mark_learning_complete",
            "description": (
                "Mark the daily learning exercise as completed. "
                "Call this ONLY after the learner has answered the "
                "learning exercise with a reasonable answer. "
                "Do not call it when the learner only says hello, "
                "agrees to start, is present, or asks to stop."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
    )
    async def mark_learning_complete(
        self,
        raw_arguments: dict[str, object],
        context: RunContext,
    ) -> str:
        """
        Mark the daily learning exercise as completed.

        This is the only action that can make the current call successful.
        It must only be called after the learner has actually completed the
        exercise and provided a reasonable answer.
        """

        if self.learning_completed:
            return "The learning exercise is already marked as complete."

        self.learning_completed = True

        logger.info(
            "Learning exercise completed successfully for caller %s",
            self.user_id,
        )

        return (
            "Learning exercise marked complete. "
            "The current call will be recorded as SUCCESS when it ends."
        )
    # =====================================================
    # TOOL 4 — SAVE MEMORY
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

        # HARD CONSENT CHECK
        if consent_confirmed is not True:
            logger.warning(
                "BLOCKED memory save without explicit consent for caller %s",
                self.user_id,
            )

            return (
                "Memory was NOT saved. Explicit caller consent "
                "was not confirmed."
            )

        # PARSE FACTS
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

        # VALIDATE NAME
        name = name.strip()

        if not name:
            return (
                "Memory was NOT saved because a caller "
                "name is required."
            )

        # SAVE MEMORY
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
                f"Memory could not be saved. Error: {error}"
            )

    # =====================================================
    # TOOL 5 — CONFIRM ESCALATION CONSENT
    # =====================================================

    @function_tool
    async def confirm_escalation_consent(
        self,
        context: RunContext,
        confirmed: bool,
    ) -> str:

        if confirmed is True:
            self.escalation_consent_confirmed = True

            logger.info(
                "Escalation consent confirmed for caller %s",
                self.user_id,
            )

            return (
                "Escalation consent confirmed. You may now create the "
                "human-help request using only the information the learner "
                "agreed to share."
            )

        self.escalation_consent_confirmed = False

        logger.info(
            "Escalation consent declined for caller %s",
            self.user_id,
        )

        return (
            "Escalation consent declined. Do not create an escalation."
        )

    # =====================================================
    # TOOL 6 — CREATE HUMAN ESCALATION
    # =====================================================

    @function_tool
    async def create_escalation(
        self,
        context: RunContext,
        reason: str,
        what_happened: str,
        what_agent_checked: str,
        urgency: str,
        language: str,
        preferred_follow_up: str,
        permission_confirmed: bool = False,
    ) -> str:

        # HARD PERMISSION CHECK
        if (
            permission_confirmed is not True
            or self.escalation_consent_confirmed is not True
        ):
            logger.warning(
                "BLOCKED escalation without explicit permission "
                "for caller %s",
                self.user_id,
            )

            return (
                "Escalation was NOT created. The learner must explicitly "
                "agree to share the specified information with a human "
                "teacher or counselor first."
            )

        # VALIDATE REASON
        allowed_reasons = {
            "learner_upset",
            "needs_teacher",
        }

        if reason not in allowed_reasons:
            return (
                "Escalation was NOT created. The reason must be either "
                "'learner_upset' or 'needs_teacher'."
            )

        # CLEAN INPUT
        what_happened = what_happened.strip()
        what_agent_checked = what_agent_checked.strip()
        urgency = urgency.strip().lower()
        language = language.strip()
        preferred_follow_up = preferred_follow_up.strip()

        if not what_happened:
            return (
                "Escalation was NOT created because what happened "
                "was not provided."
            )

        if not what_agent_checked:
            what_agent_checked = (
                "The agent reviewed the learner's request and determined "
                "that human help is appropriate."
            )

        if urgency not in {"low", "medium", "high"}:
            urgency = "medium"

        if not language:
            language = "not specified"

        if not preferred_follow_up:
            preferred_follow_up = "not specified"

        # CREATE REFERENCE ID
        reference_id = (
            "ESC-" + uuid.uuid4().hex[:8].upper()
        )

        # SHORT SUMMARY
        summary = (
            f"Human help requested for {reason}. "
            f"Situation: {what_happened}. "
            f"Agent checked: {what_agent_checked}. "
            f"Urgency: {urgency}. "
            f"Language: {language}. "
            f"Preferred follow-up: {preferred_follow_up}."
        )

        # SAVE TO LOCAL DATABASE
        try:
            with sqlite3.connect(DATABASE_PATH) as connection:
                connection.execute(
                    """
                    INSERT INTO escalations (
                        reference_id,
                        user_id,
                        reason,
                        summary,
                        what_happened,
                        what_agent_checked,
                        urgency,
                        language,
                        follow_up_method,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        reference_id,
                        self.user_id,
                        reason,
                        summary,
                        what_happened,
                        what_agent_checked,
                        urgency,
                        language,
                        preferred_follow_up,
                    ),
                )

                connection.commit()

            logger.info(
                "Human escalation created: %s",
                reference_id,
            )

            # Consent is consumed after successful escalation.
            self.escalation_consent_confirmed = False

            return json.dumps(
                {
                    "success": True,
                    "reference_id": reference_id,
                    "message": (
                        "The human-help request was successfully created."
                    ),
                    "next_step": (
                        "A human teacher or counselor can review the request "
                        "and follow up using the preferred method. Do not "
                        "promise an immediate response."
                    ),
                },
                ensure_ascii=False,
            )

        except Exception:
            logger.exception(
                "Failed to create escalation for caller %s",
                self.user_id,
            )

            return (
                "The human-help request could not be created because of "
                "a temporary system error."
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
    """Get the anonymous caller ID from the LiveKit participant."""

    participants = list(
        ctx.room.remote_participants.values()
    )

    if participants:
        identity = participants[0].identity

        if identity:
            return identity

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
    # Record every real call
    # -----------------------------------------------------

    call_id = ctx.room.name

    try:
        start_call(
            call_id=call_id,
            user_id=user_id,
        )

        logger.info(
            "Call outcome tracking started for call %s",
            call_id,
        )

    except Exception:
        logger.exception(
            "Failed to create call outcome record for call %s",
            call_id,
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
        # KEEPING GROQ AS REQUESTED
        llm=groq.LLM(
            model="llama-3.3-70b-versatile",
        ),

        # Text-to-speech
        # Recommended Murf voice
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

        # Generate responses before user completely finishes speaking
        preemptive_generation=True,
    )

    # -----------------------------------------------------
    # Start agent
    # -----------------------------------------------------

    assistant = Assistant(
        user_id=user_id,
        outbound=True,
    )

    def on_session_close(event) -> None:
        outcome = (
            "SUCCESS"
            if assistant.learning_completed
            else "FAILED"
        )

        try:
            finish_call(
                call_id=call_id,
                outcome=outcome,
            )

            logger.info(
                "Call %s finished with outcome=%s",
                call_id,
                outcome,
            )

        except Exception:
            logger.exception(
                "Failed to save outcome for call %s",
                call_id,
            )

    session.on(
        "close",
        on_session_close,
    )

    await session.start(
        agent=assistant,
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

            "Before greeting the caller, FIRST use the lookup_caller tool "
            "to check whether saved memory exists. "

            "If saved memory contains the caller's name, use that saved name. "
            "If no saved memory exists, do not invent a name. "

            "DO NOT use the normal generic CareerPath AI introduction. "
            "DO NOT ask 'How can I help you today?' "

            "This call is specifically for the learner's scheduled daily "
            "learning practice. "

            "The opening should clearly say who is calling, that this is "
            "their daily learning practice call, that they can tell you "
            "anytime if they want these calls to stop, and then ask if they "
            "are ready to start today's learning session. "

            "If a saved name exists, use this structure: "
            "'नमस्ते [NAME], CareerPath AI की ओर से आपकी दैनिक learning "
            "practice के लिए कॉल है। यह आपकी scheduled practice session है, "
            "और अगर आप चाहें तो किसी भी समय कह सकते हैं कि आप इन calls को "
            "बंद करना चाहते हैं। क्या आप आज का learning session शुरू करने "
            "के लिए तैयार हैं?' "

            "If there is no saved name, use the same structure without "
            "inventing a name. "

            "IMPORTANT LANGUAGE AND SCRIPT RULE: "
            "If speaking Hindi, write Hindi in Devanagari script. "
            "Never write Hindi using Romanized Hindi. "
            "For example, use 'नमस्ते', never 'namaste'. "

            "After the opening, begin one short learning exercise. "

            "Ask one clear question related to learning, skills, or career "
            "preparation, and let the learner answer in their own words. "

            "If the learner gives a reasonable answer, briefly acknowledge "
            "it and call the mark_learning_complete tool. "

            "Do NOT call that tool merely because the learner is present, "
            "says hello, agrees to start, or asks to stop. "

            "If the learner refuses, hangs up, asks to stop, or never "
            "completes the exercise, do not call the tool. "

            "Keep the interaction conversational, encouraging, and concise."
        ),
    )


# =========================================================
# RUN AGENT
# =========================================================

if __name__ == "__main__":
    cli.run_app(server)