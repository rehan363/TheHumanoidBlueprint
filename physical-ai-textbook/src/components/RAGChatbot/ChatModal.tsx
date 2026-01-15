/**
 * Chat modal dialog component
 */

import React, { useEffect, useCallback } from 'react';
import type { ChatMessage } from './types';
import type { SelectionData } from '../../hooks/useTextSelection';
import { MessageList } from './MessageList';
import { QueryInput } from './QueryInput';
import styles from './styles.module.css';

interface ChatModalProps {
  isOpen: boolean;
  onClose: () => void;
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  onSubmitQuery: (query: string) => void;
  rateLimitSeconds: number;
  selection: SelectionData | null;
}

export const ChatModal: React.FC<ChatModalProps> = ({
  isOpen,
  onClose,
  messages,
  isLoading,
  error,
  onSubmitQuery,
  rateLimitSeconds,
  selection,
}) => {
  // Close on Escape key
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
