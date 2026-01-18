/**
 * @file QueryInput.tsx
 * @description React component for the chat input field, handling user queries,
 * character limits, debouncing, and integration with text selection.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { SelectionData } from '../../hooks/useTextSelection';
import styles from './styles.module.css';

/**
 * Props for the QueryInput component.
 */
interface QueryInputProps {
  /**
   * Callback function to submit the query.
   * @param query The query string to be submitted.
   */
  onSubmit: (query: string) => void;
  /**
   * Indicates if a query is currently being processed.
   */
  isLoading: boolean;
  /**
   * Disables the input and submit button if true.
   */
  disabled?: boolean;
  /**
   * Placeholder text for the input field.
   */
  placeholder?: string;
  /**
   * The number of seconds remaining until the rate limit resets. If > 0, input is disabled.
   */
  rateLimitSeconds?: number;
  /**
   * Data about currently selected text, used to pre-fill the query input.
   */
  selection: SelectionData | null;
}

/**
 * `QueryInput` is a functional React component that provides the text input area
 * for users to type their questions. It features:
 * - A character limit of 500.
 * - Auto-resizing textarea.
 * - Submission on Enter key (Shift+Enter for new line).
 * - Integration with text selection to pre-fill queries.
 * - Disabling based on loading state, external `disabled` prop, or rate limits.
 *
 * @param {QueryInputProps} props The properties for the QueryInput component.
 * @returns {React.FC<QueryInputProps>} The rendered query input component.
 */
export const QueryInput: React.FC<QueryInputProps> = ({
  onSubmit,
  isLoading,
  disabled = false,
  placeholder = 'Ask a question about the textbook...',
  rateLimitSeconds = 0,
  selection,
}) => {
  const [query, setQuery] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Pre-fill and focus when a text selection is made
  useEffect(() => {
    if (selection && textareaRef.current) {
      const newQuery = `What does this mean: "${selection.text}"?`;
      setQuery(newQuery);
      textareaRef.current.focus();
      // Auto-resize textarea
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [selection]);

  /**
   * Handles the submission of the query. Trims the query, checks validity,
   * calls the `onSubmit` prop, and clears the input field.
   */
  const handleSubmit = useCallback(() => {
    const trimmedQuery = query.trim();
    if (trimmedQuery && !isLoading && !disabled && rateLimitSeconds === 0) {
      onSubmit(trimmedQuery);
      setQuery('');

      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  }, [query, isLoading, disabled, rateLimitSeconds, onSubmit]);

  /**
   * Handles key down events in the textarea, specifically for submitting on Enter.
   * @param {React.KeyboardEvent<HTMLTextAreaElement>} e The keyboard event object.
   */
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (but allow Shift+Enter for new line)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  /**
   * Handles changes to the textarea input, updates the query state,
   * enforces a character limit, and auto-resizes the textarea.
   * @param {React.ChangeEvent<HTMLTextAreaElement>} e The change event object.
   */
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;

    // Limit to 500 characters
    if (value.length <= 500) {
      setQuery(value);

      // Auto-resize textarea
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
        textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
      }
    }
  };

  const isDisabled = disabled || isLoading || rateLimitSeconds > 0;
  const canSubmit = query.trim().length > 0 && !isDisabled;

  return (
    <div className={styles.inputContainer}>
      {rateLimitSeconds > 0 && (
        <div className={styles.rateLimitNotice}>
          ⏳ Rate limit: Please wait {rateLimitSeconds}s before sending another message
        </div>
      )}

      <div className={styles.inputWrapper}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          value={query}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isDisabled}
          rows={1}
          maxLength={500}
          aria-label="Chat query input"
        />
        <button
          className={styles.sendButton}
          onClick={handleSubmit}
          disabled={!canSubmit}
          aria-label="Send message"
        >
          {isLoading ? '...' : 'Send'}
        </button>
      </div>

      <div style={{ fontSize: '11px', color: '#a0aec0', marginTop: '4px', textAlign: 'right' }}>
        {query.length}/500
      </div>
    </div>
  );
};
