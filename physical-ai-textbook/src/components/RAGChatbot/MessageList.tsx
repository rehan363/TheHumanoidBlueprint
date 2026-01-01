/**
 * Message list component for displaying chat history
 */

import React, { useEffect, useRef } from 'react';
import type { ChatMessage } from './types';
import { Citation } from './Citation';
import styles from './styles.module.css';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading: boolean;
}

export const MessageList: React.FC<MessageListProps> = ({ messages, isLoading }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (messages.length === 0 && !isLoading) {
    return (
      <div className={styles.messageList}>
        <div className={styles.emptyState}>
          <div className={styles.emptyStateIcon}>💬</div>
          <div className={styles.emptyStateTitle}>Welcome to AI Assistant!</div>
          <div className={styles.emptyStateText}>
            Ask me anything about the Physical AI textbook.
            <br />
            I'll provide answers with source citations.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.messageList}>
      {messages.map((message) => (
        <div
          key={message.id}
          className={`${styles.message} ${
            message.role === 'user' ? styles.userMessage : styles.assistantMessage
          }`}
        >
          <div
            className={`${styles.messageBubble} ${
              message.role === 'user' ? styles.userBubble : styles.assistantBubble
            }`}
          >
            <p className={styles.messageContent}>{message.content}</p>

            {message.citations && message.citations.length > 0 && (
              <div className={styles.citations}>
                <div className={styles.citationsTitle}>Sources:</div>
                {message.citations.map((citation, index) => (
                  <Citation key={index} citation={citation} />
                ))}
              </div>
            )}

            <div className={styles.messageTime}>
              {formatTime(message.timestamp)}
              {message.processing_time_ms && ` • ${message.processing_time_ms}ms`}
            </div>
          </div>
        </div>
      ))}

      {isLoading && (
        <div className={`${styles.message} ${styles.assistantMessage}`}>
          <div className={`${styles.messageBubble} ${styles.assistantBubble}`}>
            <div className={styles.loadingDots}>
              <div className={styles.dot} />
              <div className={styles.dot} />
              <div className={styles.dot} />
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
};
