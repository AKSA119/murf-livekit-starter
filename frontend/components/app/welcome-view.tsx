'use client';

import { Button } from '@/components/ui/button';

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
}: WelcomeViewProps) => {
  return (
    <div className="relative flex min-h-screen w-full items-center justify-center overflow-hidden bg-[#05050b] text-white">
      {/* Background glow */}
      <div className="absolute left-1/2 top-1/2 h-[700px] w-[700px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-violet-600/10 blur-[140px]" />

      <div className="absolute -left-40 -top-40 h-[400px] w-[400px] rounded-full bg-blue-600/10 blur-[120px]" />

      <div className="absolute -bottom-40 -right-40 h-[400px] w-[400px] rounded-full bg-fuchsia-600/10 blur-[120px]" />

      {/* Top badge */}
      <div className="absolute right-8 top-7 hidden rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-xs font-medium tracking-wider text-white/60 backdrop-blur-md md:block">
        ✦ AI CAREER COMPANION
      </div>

      {/* Main content */}
      <div className="relative z-10 flex w-full max-w-4xl flex-col items-center px-6 text-center">
        
        {/* Voice Orb */}
        <div className="relative mb-10 flex h-36 w-36 items-center justify-center">
          {/* Outer rings */}
          <div className="absolute inset-0 animate-ping rounded-full border border-violet-400/20" />
          <div className="absolute -inset-4 rounded-full border border-violet-400/10" />
          <div className="absolute -inset-8 rounded-full border border-blue-400/5" />

          {/* Glow */}
          <div className="absolute inset-0 rounded-full bg-violet-500/30 blur-3xl" />

          {/* Orb */}
          <div className="relative flex h-28 w-28 items-center justify-center rounded-full border border-white/20 bg-gradient-to-br from-violet-500 via-indigo-500 to-blue-600 shadow-[0_0_80px_rgba(124,58,237,0.45)]">
            <div className="absolute inset-3 rounded-full bg-white/10 backdrop-blur-sm" />

            <span className="relative text-5xl">🎙️</span>
          </div>
        </div>

        {/* Small label */}
        <div className="mb-5 flex items-center gap-2 rounded-full border border-violet-400/20 bg-violet-500/10 px-4 py-2 text-xs font-medium text-violet-200">
          <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
          AI VOICE ASSISTANT
        </div>

        {/* Heading */}
        <h1 className="bg-gradient-to-r from-white via-violet-100 to-blue-200 bg-clip-text text-5xl font-bold tracking-tight text-transparent md:text-7xl">
          CareerPath AI
        </h1>

        <p className="mt-5 text-xl font-medium text-white/90 md:text-2xl">
          Your career. Your questions. One conversation.
        </p>

        <p className="mt-5 max-w-2xl text-sm leading-7 text-white/50 md:text-base">
          Talk naturally with your AI career companion. Explore careers,
          discover skills, find courses, and build a roadmap for your future.
        </p>

        {/* CTA */}
        <Button
          size="lg"
          onClick={onStartCall}
          className="group relative mt-9 h-14 w-72 overflow-hidden rounded-full border border-white/20 bg-gradient-to-r from-violet-600 to-blue-600 text-base font-semibold text-white shadow-[0_10px_40px_rgba(99,102,241,0.35)] transition-all duration-300 hover:scale-105 hover:shadow-[0_15px_50px_rgba(99,102,241,0.5)]"
        >
          <span className="relative z-10 flex items-center justify-center gap-3">
            <span className="text-lg">🎙️</span>
            {startButtonText}
            <span className="transition-transform duration-300 group-hover:translate-x-1">
              →
            </span>
          </span>

          <div className="absolute inset-0 -translate-x-full bg-white/10 transition-transform duration-500 group-hover:translate-x-full" />
        </Button>

        {/* Supported languages */}
        <div className="mt-6 flex items-center gap-3 text-xs text-white/40">
          <span>English</span>
          <span className="h-1 w-1 rounded-full bg-white/30" />
          <span>Hindi</span>
          <span className="h-1 w-1 rounded-full bg-white/30" />
          <span>Hinglish</span>
        </div>

        {/* Feature cards */}
        <div className="mt-14 grid w-full max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 backdrop-blur-md transition-all hover:border-violet-400/30 hover:bg-white/[0.06]">
            <div className="mb-2 text-xl">💬</div>
            <p className="text-sm font-semibold text-white/80">
              Natural Conversation
            </p>
            <p className="mt-1 text-xs text-white/40">
              Just speak normally
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 backdrop-blur-md transition-all hover:border-violet-400/30 hover:bg-white/[0.06]">
            <div className="mb-2 text-xl">🧭</div>
            <p className="text-sm font-semibold text-white/80">
              Career Guidance
            </p>
            <p className="mt-1 text-xs text-white/40">
              Find your direction
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 backdrop-blur-md transition-all hover:border-violet-400/30 hover:bg-white/[0.06]">
            <div className="mb-2 text-xl">⚡</div>
            <p className="text-sm font-semibold text-white/80">
              Instant Answers
            </p>
            <p className="mt-1 text-xs text-white/40">
              Powered by AI
            </p>
          </div>
        </div>

        <p className="mt-8 text-[11px] tracking-wide text-white/25">
          Your conversation starts with one simple “Hello”.
        </p>
      </div>
    </div>
  );
};