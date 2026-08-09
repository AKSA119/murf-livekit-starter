import logging
import os

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    function_tool,
    cli,
    tokenize,
    room_io,
)
from livekit.plugins import (
    murf,
    silero,
    groq,
    deepgram,
    noise_cancellation,
)

from database import init_db, save_user


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("agent")


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv(".env.local")

# Initialize SQLite database
init_db()

print("Groq key loaded:", bool(os.getenv("GROQ_API_KEY")))


# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------

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


OBJECTIVES

- Understand the user's education and career goals.
- Provide useful guidance about careers, courses, and skills.
- Help users explore suitable career paths.
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

If you are unsure or information is institution-specific, clearly say so.


LANGUAGE

Mirror the user's language naturally.

If the user speaks Hindi-English (Hinglish), reply naturally in the same style.

If the user switches between English and Hindi, switch accordingly.

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

"I'm sorry, but this is outside what I can safely help with. Please contact the institution or a qualified career counselor for official guidance."


STYLE

Keep responses conversational and concise.

Normally respond in 2–4 sentences.

Avoid long monologues.

Speak naturally.


MEMORY

During the conversation, naturally learn useful information about the user.

Useful information includes:

- Their name
- Their preferred language
- Their current education level
- Their interests
- Their target career

Do not ask all questions at once.

Learn these details naturally during the conversation.

When the user provides useful profile information, use the
save_user_profile tool to save it.

Do not repeatedly ask for information the user has already provided.


FIRST-TURN GREETING

Start every new conversation by saying:

"Hello! I'm CareerPath AI, your Education & Career Guidance Assistant. I can help you explore courses, career options, skills, and general admission information. How can I help you today?"
"""


# ---------------------------------------------------------
# CareerPath AI Assistant
# ---------------------------------------------------------

class Assistant(Agent):

    def __init__(self, user_id: str) -> None:

        self.user_id = user_id

        super().__init__(
            instructions=SYSTEM_PROMPT
        )

    @function_tool
    async def save_user_profile(
        self,
        context: RunContext,
        name: str,
        language_preference: str = "",
        current_level: str = "",
        interests: str = "",
        target_career: str = "",
    ):
        """
        Save useful information about the current user.

        Use this tool when the user shares information such as
        their name, language preference, education level,
        interests, or target career.
        """

        try:

            save_user(
                user_id=self.user_id,
                name=name,
                language_preference=language_preference,
                current_level=current_level,
                interests=interests,
                target_career=target_career,
            )

            logger.info(
                "Saved profile for user %s",
                self.user_id,
            )

            return "User profile saved successfully."

        except Exception as e:

            logger.exception(
                "Failed to save profile for user %s",
                self.user_id,
            )

            return f"Unable to save the profile: {e}"


# ---------------------------------------------------------
# LiveKit Agent Server
# ---------------------------------------------------------

server = AgentServer()


# ---------------------------------------------------------
# Prewarm
# ---------------------------------------------------------

def prewarm(proc: JobProcess):

    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


# ---------------------------------------------------------
# Agent Session
# ---------------------------------------------------------

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):

    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Use the LiveKit room name as the user's ID
    user_id = ctx.room.name

    logger.info(
        "Starting CareerPath AI for user %s",
        user_id,
    )

    # -----------------------------------------------------
    # Voice AI Pipeline
    # -----------------------------------------------------

    session = AgentSession(

        # Speech-to-text
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
        ),

        # Large Language Model
        llm=groq.LLM(
            model="llama-3.3-70b-versatile",
        ),

        # Text-to-speech
        tts=murf.TTS(
            voice="Abhinav",
            style="Conversational",
            model="FALCON",
            tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=2
            ),
            text_pacing=True,
        ),

        # Voice activity detection
        vad=ctx.proc.userdata["vad"],

        # Generate responses before the user completely finishes
        preemptive_generation=True,
    )

    # -----------------------------------------------------
    # Start Session
    # -----------------------------------------------------

    await session.start(
        agent=Assistant(user_id=user_id),
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
    # Connect to LiveKit room
    # -----------------------------------------------------

    await ctx.connect()


# ---------------------------------------------------------
# Run Agent
# ---------------------------------------------------------

if __name__ == "__main__":
    cli.run_app(server)