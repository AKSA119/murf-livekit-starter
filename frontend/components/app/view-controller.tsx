'use client';

import { useEffect, useState } from 'react';
import { useTheme } from 'next-themes';
import { AnimatePresence, motion } from 'motion/react';
import { useSessionContext } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { AgentSessionView_01 } from '@/components/agents-ui/blocks/agent-session-view-01';
import { WelcomeView } from '@/components/app/welcome-view';

const MotionWelcomeView = motion.create(WelcomeView);
const MotionSessionView = motion.create(AgentSessionView_01);

const VIEW_MOTION_PROPS = {
  variants: {
    visible: {
      opacity: 1,
    },
    hidden: {
      opacity: 0,
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.5,
    ease: [0.4, 0, 0.2, 1] as [number, number, number, number],
  },
};

type CallStats = {
  totalCalls: number;
  successfulCalls: number;
  failedCalls: number;
};

function CallDashboard() {
  const [stats, setStats] = useState<CallStats | null>(null);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const loadStats = async () => {
      try {
        const response = await fetch('/api/call-stats', {
          cache: 'no-store',
        });

        if (!response.ok) {
          throw new Error(`Stats request failed: ${response.status}`);
        }

        const nextStats = (await response.json()) as CallStats;

        if (!cancelled) {
          setStats(nextStats);
          setHasError(false);
        }
      } catch (error) {
        console.error('Unable to load call statistics:', error);

        if (!cancelled) {
          setHasError(true);
        }
      }
    };

    loadStats();

    // Poll so the dashboard reflects the result of a call shortly after
    // the LiveKit session closes.
    const interval = window.setInterval(loadStats, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  const displayValue = (value: number | undefined) =>
    value === undefined ? '—' : value;

  return (
    <section
      aria-label="Call outcome dashboard"
      className="fixed inset-x-4 bottom-4 z-40 mx-auto max-w-3xl rounded-2xl border border-border/70 bg-background/90 p-4 shadow-lg backdrop-blur-md"
    >
      <div className="mb-3 flex items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold">Call Dashboard</h2>
          <p className="text-xs text-muted-foreground">
            Live totals from completed agent calls
          </p>
        </div>

        {hasError && (
          <span className="text-xs text-destructive">
            Statistics unavailable
          </span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-xl border border-border/60 bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total Calls</p>
          <p className="mt-1 text-2xl font-bold">
            {displayValue(stats?.totalCalls)}
          </p>
        </div>

        <div className="rounded-xl border border-border/60 bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Successful Calls</p>
          <p className="mt-1 text-2xl font-bold">
            {displayValue(stats?.successfulCalls)}
          </p>
        </div>

        <div className="rounded-xl border border-border/60 bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Failed Calls</p>
          <p className="mt-1 text-2xl font-bold">
            {displayValue(stats?.failedCalls)}
          </p>
        </div>
      </div>

      <p className="mt-3 text-[11px] text-muted-foreground">
        Only aggregate counts are shown. Caller details and conversation
        content are not displayed.
      </p>
    </section>
  );
}

interface ViewControllerProps {
  appConfig: AppConfig;
}

export function ViewController({ appConfig }: ViewControllerProps) {
  const { isConnected, start } = useSessionContext();
  const { resolvedTheme } = useTheme();

  return (
    <AnimatePresence mode="wait">
      {/* Welcome view + call outcome dashboard */}
      {!isConnected && (
        <div key="welcome-content" className="contents">
          <MotionWelcomeView
            {...VIEW_MOTION_PROPS}
            startButtonText={appConfig.startButtonText}
            onStartCall={start}
          />
          <CallDashboard />
        </div>
      )}

      {/* Session view */}
      {isConnected && (
        <MotionSessionView
          key="session-view"
          {...VIEW_MOTION_PROPS}
          supportsChatInput={appConfig.supportsChatInput}
          supportsVideoInput={appConfig.supportsVideoInput}
          supportsScreenShare={appConfig.supportsScreenShare}
          isPreConnectBufferEnabled={appConfig.isPreConnectBufferEnabled}
          audioVisualizerType={appConfig.audioVisualizerType}
          audioVisualizerColor={
            resolvedTheme === 'dark'
              ? appConfig.audioVisualizerColorDark
              : appConfig.audioVisualizerColor
          }
          audioVisualizerColorShift={appConfig.audioVisualizerColorShift}
          audioVisualizerBarCount={appConfig.audioVisualizerBarCount}
          audioVisualizerGridRowCount={appConfig.audioVisualizerGridRowCount}
          audioVisualizerGridColumnCount={appConfig.audioVisualizerGridColumnCount}
          audioVisualizerRadialBarCount={
            appConfig.audioVisualizerRadialBarCount
          }
          audioVisualizerRadialRadius={appConfig.audioVisualizerRadialRadius}
          audioVisualizerWaveLineWidth={appConfig.audioVisualizerWaveLineWidth}
          className="fixed inset-0"
        />
      )}
    </AnimatePresence>
  );
}
