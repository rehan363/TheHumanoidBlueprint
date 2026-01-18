/**
 * @file MessageList.tsx
 * @description React component for displaying the chronological list of chat messages.
 * It handles automatic scrolling, displays user and assistant messages, citations,
 * and loading indicators.
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import type { ChatMessage } from './types';
import { Citation } from './Citation';
import styles from './styles.module.css';

/**
 * Props for the MessageList component.
 */
interface MessageListProps {
  /**
   * An array of chat messages to be displayed.
   */
  messages: ChatMessage[];
  /**
   * Indicates if a message is currently being loaded (e.g., from the AI assistant).
   */
  isLoading: boolean;
}

/**
 * `MessageList` is a functional React component that renders the conversation history.
 * It displays messages from both the user and the assistant, including any relevant citations.
 * The component also manages scroll behavior, automatically scrolling to the bottom when
 * new messages are added, unless the user has scrolled up to view older messages.
 *
 * @param {MessageListProps} props The properties for the MessageList component.
 * @returns {React.FC<MessageListProps>} The rendered message list component.
 */
export const MessageList: React.FC<MessageListProps> = ({ messages, isLoading }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);

  // Auto-scroll to bottom when new messages arrive, but only if user is already at the bottom
  useEffect(() => {
    if (isScrolledToBottom) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading, isScrolledToBottom]);

  /**
   * Checks the scroll position of the message container to determine if the user is
   * currently scrolled near the bottom.
   */
  const handleScroll = useCallback(() => {
    if (scrollContainerRef.current) {
      const { scrollTop, clientHeight, scrollHeight } = scrollContainerRef.current;
      const threshold = 10; // Pixels from bottom to consider "at bottom"
      setIsScrolledToBottom(scrollTop + clientHeight >= scrollHeight - threshold);
    }
  }, []);

  useEffect(() => {
    const scrollContainer = scrollContainerRef.current;
    if (scrollContainer) {
      scrollContainer.addEventListener('scroll', handleScroll);
      // Initial check
      handleScroll();
    }
    return () => {
      if (scrollContainer) {
        scrollContainer.removeEventListener('scroll', handleScroll);
      }
    };
  }, [handleScroll]);

  /**
   * Formats a given Date object into a localized time string (e.g., "HH:MM AM/PM").
   * @param {Date} date The Date object to format.
   * @returns {string} The formatted time string.
   */
  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (messages.length === 0 && !isLoading) {
    return (
      <div className={styles.messageList} ref={scrollContainerRef}>
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
    <div className={styles.messageList} ref={scrollContainerRef}>
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
