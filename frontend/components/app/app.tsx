'use client';

import { useMemo } from 'react';
import { TokenSource } from 'livekit-client';
import { useSession } from '@livekit/components-react';
import { WarningIcon } from '@phosphor-icons/react/dist/ssr';

import type { AppConfig } from '@/app-config';
import { AgentSessionProvider } from '@/components/agents-ui/agent-session-provider';
import { StartAudioButton } from '@/components/agents-ui/start-audio-button';
import { ViewController } from '@/components/app/view-controller';
import { Toaster } from '@/components/ui/sonner';
import { useAgentErrors } from '@/hooks/useAgentErrors';
import { useDebugMode } from '@/hooks/useDebug';
import { getSandboxTokenSource } from '@/lib/utils';

const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

function AppSetup() {
  useDebugMode({ enabled: IN_DEVELOPMENT });
  useAgentErrors();

  return null;
}

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  /*
   * -------------------------------------------------------
   * CREATE / RETRIEVE ANONYMOUS CALLER ID
   * -------------------------------------------------------
   *
   * This ID is NOT the caller's name.
   *
   * It is simply a stable anonymous ID for this browser.
   *
   * The first call creates it.
   * Future calls reuse the same ID.
   */
  const participantIdentity = useMemo(() => {
    if (typeof window === 'undefined') {
      return '';
    }

    let identity = localStorage.getItem(
      'careerpath_caller_id'
    );

    if (!identity) {
      identity = `voice_assistant_user_${crypto.randomUUID()}`;

      localStorage.setItem(
        'careerpath_caller_id',
        identity
      );
    }

    return identity;
  }, []);

  /*
   * -------------------------------------------------------
   * TOKEN SOURCE
   * -------------------------------------------------------
   */
  const tokenSource = useMemo(() => {
    if (
      typeof process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT ===
      'string'
    ) {
      return getSandboxTokenSource(appConfig);
    }

    return TokenSource.endpoint('/api/token');
  }, [appConfig]);

  /*
   * -------------------------------------------------------
   * SESSION
   * -------------------------------------------------------
   *
   * participantIdentity is a TokenSource FETCH option.
   *
   * LiveKit automatically sends it to /api/token as:
   *
   * participant_identity
   *
   * Your existing route.ts already reads that value.
   */
  const sessionOptions = useMemo(() => {
    return {
      ...(appConfig.agentName
        ? { agentName: appConfig.agentName }
        : {}),
      ...(participantIdentity
        ? { participantIdentity }
        : {}),
    };
  }, [appConfig.agentName, participantIdentity]);

  const session = useSession(
    tokenSource,
    sessionOptions
  );

  return (
    <AgentSessionProvider session={session}>
      <AppSetup />

      <main className="grid h-svh grid-cols-1 place-content-center">
        <ViewController appConfig={appConfig} />
      </main>

      <StartAudioButton label="Start Audio" />

      <Toaster
        icons={{
          warning: <WarningIcon weight="bold" />,
        }}
        position="top-center"
        className="toaster group"
        style={
          {
            '--normal-bg': 'var(--popover)',
            '--normal-text': 'var(--popover-foreground)',
            '--normal-border': 'var(--border)',
          } as React.CSSProperties
        }
      />
    </AgentSessionProvider>
  );
}