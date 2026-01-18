/**
 * @file ChatModal.tsx
 * @description React component for displaying the RAG chatbot modal dialog.
 * This component handles the modal's visibility, keyboard accessibility (Escape key to close),
 * and integrates with other chat-related components like MessageList and QueryInput.
 */

import React, { useEffect, useCallback } from 'react';
import type { ChatMessage } from './types';
import type { SelectionData } from '../../hooks/useTextSelection';
import { MessageList } from './MessageList';
import { QueryInput } from './QueryInput';
import styles from './styles.module.css';

/**
 * Props for the ChatModal component.
 */
interface ChatModalProps {
  /**
   * Determines if the chat modal is currently open and visible.
   */
  isOpen: boolean;
  /**
   * Callback function to close the chat modal.
   */
  onClose: () => void;
  /**
   * An array of chat messages to display in the MessageList.
   */
  messages: ChatMessage[];
  /**
   * Indicates if a query is currently being processed by the RAG system.
   */
  isLoading: boolean;
  /**
   * An error message string to display, or null if no error.
   */
  error: string | null;
  /**
   * Callback function to submit a new query to the RAG system.
   * @param query The user's query string.
   */
  onSubmitQuery: (query: string) => void;
  /**
   * The number of seconds remaining until the rate limit resets, if applicable.
   */
  rateLimitSeconds: number;
  /**
   * Data about currently selected text, used for context-aware queries.
   */
  selection: SelectionData | null;
  /**
   * Callback function to clear the chat history.
   */
  onClearHistory: () => void;
}

/**
 * `ChatModal` is a functional React component that renders the main chat interface
 * within a modal dialog. It provides functionality for displaying messages,
 * submitting queries, handling errors and rate limits, and managing text selection context.
 *
 * @param {ChatModalProps} props The properties for the ChatModal component.
 * @returns {React.FC<ChatModalProps>} The rendered chat modal component.
 */
export const ChatModal: React.FC<ChatModalProps> = ({
  isOpen,
  onClose,
  messages,
  isLoading,
  error,
  onSubmitQuery,
  rateLimitSeconds,
  selection,
  onClearHistory,
}) => {
  /**
   * Handles the `keydown` event to close the modal when the Escape key is pressed.
   * @param {KeyboardEvent} e The keyboard event object.
   */
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    },
    [onClose]
  );

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen) {
    return null;
  }

  /**
   * Handles clicks on the modal overlay to close the modal if the click is outside the modal content.
   * @param {React.MouseEvent<HTMLDivElement>} e The mouse event object.
   */
  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    // Close if clicking on overlay (not on modal content)
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div className={styles.modalOverlay} onClick={handleOverlayClick}>
      <div className={styles.modalContainer} role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <div className={styles.modalHeader}>
          <h2 id="modal-title" className={styles.modalTitle}>
            AI Assistant
          </h2>
          <button
            className={styles.clearHistoryButton}
            onClick={onClearHistory}
            aria-label="Clear chat history"
            title="Clear Chat History"
          >
            Clear History
          </button>
          <button
            className={styles.closeButton}
            onClick={onClose}
            aria-label="Close chat"
          >
            ×
          </button>
        </div>

        {error && (
          <div className={styles.errorMessage} role="alert">
            {error}
          </div>
        )}

        <MessageList messages={messages} isLoading={isLoading} />

        <QueryInput
          onSubmit={onSubmitQuery}
          isLoading={isLoading}
          disabled={!!error}
          rateLimitSeconds={rateLimitSeconds}
          selection={selection}
        />
      </div>
    </div>
  );
};
