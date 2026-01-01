/**
 * Query input component with debouncing and character limit
 */

import React, { useState, useRef, useEffect } from 'react';
import styles from './styles.module.css';

interface QueryInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  disabled?: boolean;
  placeholder?: string;
  rateLimitSeconds?: number;
}

export const QueryInput: React.FC<QueryInputProps> = ({
  onSubmit,
  isLoading,
  disabled = false,
  placeholder = 'Ask a question about the textbook...',
  rateLimitSeconds = 0,
}) => {
  const [query, setQuery] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const trimmedQuery = query.trim();
    if (trimmedQuery && !isLoading && !disabled && rateLimitSeconds === 0) {
      onSubmit(trimmedQuery);
      setQuery('');

      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (but allow Shift+Enter for new line)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

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
