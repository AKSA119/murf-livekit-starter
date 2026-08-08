'use client';

import { type ComponentProps } from 'react';
import { AnimatePresence } from 'motion/react';
import { type AgentState, type ReceivedMessage } from '@livekit/components-react';

import { AgentChatIndicator } from '@/components/agents-ui/agent-chat-indicator';

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';

import {
  Message,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message';

export interface AgentChatTranscriptProps extends ComponentProps<'div'> {
  agentState?: AgentState;
  messages?: ReceivedMessage[];
}

export function AgentChatTranscript({
  agentState,
  messages = [],
  className,
  ...props
}: AgentChatTranscriptProps) {
  return (
    <Conversation className={className} {...props}>
      <ConversationContent>
        {messages.map((receivedMessage) => {
          const { id, timestamp, from, message } = receivedMessage;

          const messageOrigin = from?.isLocal ? 'user' : 'assistant';

          const time = new Date(timestamp);
          const title = time.toLocaleTimeString();

          return (
            <Message
              key={id}
              title={title}
              from={messageOrigin}
            >
              <MessageContent>
                <MessageResponse>
                  {message}
                </MessageResponse>
              </MessageContent>
            </Message>
          );
        })}

        <AnimatePresence>
          {agentState === 'thinking' && (
            <AgentChatIndicator size="sm" />
          )}
        </AnimatePresence>
      </ConversationContent>

      <ConversationScrollButton />
    </Conversation>
  );
}