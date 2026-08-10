import { NextResponse } from 'next/server';
import {
  AccessToken,
  type AccessTokenOptions,
  type VideoGrant,
} from 'livekit-server-sdk';
import { RoomConfiguration } from '@livekit/protocol';

type ConnectionDetails = {
  serverUrl: string;
  roomName: string;
  participantName: string;
  participantToken: string;
  participantIdentity: string;
};

const API_KEY = process.env.LIVEKIT_API_KEY;
const API_SECRET = process.env.LIVEKIT_API_SECRET;
const LIVEKIT_URL = process.env.LIVEKIT_URL;
const AGENT_NAME = process.env.AGENT_NAME;

export const revalidate = 0;

export async function POST(req: Request) {
  try {
    if (LIVEKIT_URL === undefined) {
      throw new Error('LIVEKIT_URL is not defined');
    }

    if (API_KEY === undefined) {
      throw new Error('LIVEKIT_API_KEY is not defined');
    }

    if (API_SECRET === undefined) {
      throw new Error('LIVEKIT_API_SECRET is not defined');
    }

    // Parse room config from request body
    const body = await req.json().catch(() => ({}));

    let roomConfig: RoomConfiguration | undefined;

    if (body?.room_config) {
      roomConfig = RoomConfiguration.fromJson(
        body.room_config,
        { ignoreUnknownFields: true }
      );
    } else if (AGENT_NAME) {
      roomConfig = RoomConfiguration.fromJson(
        {
          agents: [{ agentName: AGENT_NAME }],
        },
        { ignoreUnknownFields: true }
      );
    }

    /*
     * IMPORTANT:
     *
     * The frontend sends a stable caller ID in the request.
     *
     * If none is supplied, create one.
     *
     * The frontend should store this ID in localStorage and
     * send the same ID on future calls.
     */

    let participantIdentity =
      typeof body?.participant_identity === 'string'
        ? body.participant_identity.trim()
        : '';

    if (!participantIdentity) {
      participantIdentity = `voice_assistant_user_${crypto.randomUUID()}`;
    }

    const participantName = 'user';

    /*
     * For now the room can remain unique per call.
     * The IMPORTANT persistent value is participantIdentity.
     */
    const roomName = `voice_assistant_room_${crypto.randomUUID()}`;

    const participantToken = await createParticipantToken(
      {
        identity: participantIdentity,
        name: participantName,
      },
      roomName,
      roomConfig
    );

    const data: ConnectionDetails = {
      serverUrl: LIVEKIT_URL,
      roomName,
      participantName,
      participantToken,
      participantIdentity,
    };

    const headers = new Headers({
      'Cache-Control': 'no-store',
    });

    return NextResponse.json(data, { headers });
  } catch (error) {
    if (error instanceof Error) {
      console.error(error);

      return new NextResponse(
        error.message,
        { status: 500 }
      );
    }

    return new NextResponse(
      'Unknown error',
      { status: 500 }
    );
  }
}

function createParticipantToken(
  userInfo: AccessTokenOptions,
  roomName: string,
  roomConfig?: RoomConfiguration
): Promise<string> {
  const at = new AccessToken(
    API_KEY,
    API_SECRET,
    {
      ...userInfo,
      ttl: '15m',
    }
  );

  const grant: VideoGrant = {
    room: roomName,
    roomJoin: true,
    canPublish: true,
    canPublishData: true,
    canSubscribe: true,
  };

  at.addGrant(grant);

  if (roomConfig) {
    at.roomConfig = roomConfig;
  }

  return at.toJwt();
}