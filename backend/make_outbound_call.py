import asyncio
import os

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.sip import CreateSIPParticipantRequest, SIPOutboundConfig

load_dotenv(".env.local")


async def main():
    livekit_url = os.getenv("LIVEKIT_URL")
    livekit_api_key = os.getenv("LIVEKIT_API_KEY")
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")

    sip_hostname = os.getenv("LINPHONE_SIP_HOSTNAME")
    sip_username = os.getenv("LINPHONE_SIP_USERNAME")
    sip_password = os.getenv("LINPHONE_SIP_PASSWORD")
    to_phone_number = os.getenv("TO_PHONE_NUMBER")

    if not livekit_url:
        raise RuntimeError("LIVEKIT_URL is missing")

    if not livekit_api_key:
        raise RuntimeError("LIVEKIT_API_KEY is missing")

    if not livekit_api_secret:
        raise RuntimeError("LIVEKIT_API_SECRET is missing")

    if not sip_hostname:
        raise RuntimeError("LINPHONE_SIP_HOSTNAME is missing")

    if not sip_username:
        raise RuntimeError("LINPHONE_SIP_USERNAME is missing")

    if not sip_password:
        raise RuntimeError("LINPHONE_SIP_PASSWORD is missing")

    if not to_phone_number:
        raise RuntimeError("TO_PHONE_NUMBER is missing")

    room_name = "careerpath-outbound-test"
    agent_name = "my-agent"

    trunk_config = SIPOutboundConfig(
        hostname=sip_hostname,
        auth_username=sip_username,
        auth_password=sip_password,
    )

    lkapi = api.LiveKitAPI(
        url=livekit_url,
        api_key=livekit_api_key,
        api_secret=livekit_api_secret,
    )

    try:
        print("Creating agent dispatch...")

        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=room_name,
                metadata=to_phone_number,
            )
        )

        print("Agent dispatched successfully!")
        print(f"Agent: {dispatch.agent_name}")
        print(f"Room: {dispatch.room}")

        print("Starting outbound call...")
        print(f"Destination: {to_phone_number}")

        participant = await lkapi.sip.create_sip_participant(
            CreateSIPParticipantRequest(
                trunk=trunk_config,
                sip_number=sip_username,
                sip_call_to=to_phone_number,
                room_name=room_name,
                participant_identity="outbound-caller",
                participant_name="CareerPath AI",
                wait_until_answered=True,
                play_dialtone=True,
            )
        )

        print("Outbound call connected successfully!")
        print(f"Participant: {participant.participant_identity}")
        print(f"Room: {room_name}")

    except Exception as e:
        print(f"ERROR: {e}")
        raise

    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    asyncio.run(main())