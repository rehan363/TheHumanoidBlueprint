/**
 * @file index.tsx
 * @description Main RAG Chatbot component that orchestrates the chat experience,
 * including a floating action button, text selection handling, and the chat modal.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { ChatModal } from './ChatModal';
import { useChatAPI } from '../../hooks/useChatAPI';
import useSessionStorage from '../../hooks/useSessionStorage';
import { useTextSelection } from '../../hooks/useTextSelection';
import type { ChatMessage } from './types';
import styles from './styles.module.css';

/**
 * `RAGChatbot` is the main entry component for the Retrieval-Augmented Generation (RAG) chatbot.
 * It manages the state of the chat modal, integrates with `useChatAPI` for backend communication,
 * `useSessionStorage` for chat history persistence, and `useTextSelection` for context-aware queries.
 *
 * It renders a floating action button to open the chat and a context-sensitive button
 * for asking AI about selected text.
 *
 * @returns {React.FC} The rendered RAGChatbot component.
 */
export const RAGChatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  // useSessionStorage for persisting chat history. The key 'rag-chat-history' is used for storage.
  const [chatHistory, setChatHistory] = useSessionStorage<ChatMessage[]>('rag-chat-history', []);

  // useChatAPI hook provides functionality for sending queries to the backend.
  const { isLoading, error, sendQuery, clearError, rateLimitSeconds } = useChatAPI(chatHistory, setChatHistory);
  // useTextSelection hook detects and manages user's text selections on the page.
  const { selection, clearSelection } = useTextSelection();

  /**
   * Opens the chat modal and clears any previous error messages.
   */
  const handleOpenChat = () => {
    setIsOpen(true);
    clearError();
  };

  /**
   * Closes the chat modal.
   */
  const handleCloseChat = () => {
    setIsOpen(false);
  };

  /**
   * Submits a user query to the RAG backend via the `sendQuery` function from `useChatAPI`.
   * @param {string} query The user's input query.
   */
  const handleSubmitQuery = async (query: string) => {
    await sendQuery(query);
  };

  /**
   * Clears the entire chat history, both from state and session storage, and also clears any active errors.
   */
  const handleClearHistory = () => {
    setChatHistory([]);
    clearError(); // Also clear any errors
  };

  /**
   * Handles the action when the "Ask AI about this" button is clicked after text selection.
   * It opens the chat modal and clears the selection.
   */
  const handleAskAboutSelection = () => {
    if (!selection) return;

    // Open the chat modal
    handleOpenChat();

    // Clear the on-screen selection button
    clearSelection();

    // The logic to pre-fill the input will be handled in QueryInput.tsx (T051)
    // For now, we can pass the selection context to the modal.
    // This part of the implementation anticipates the next steps.
  };

  return (
    <>
      {/* Floating Action Button */}
      <button
        className={styles.chatButton}
        onClick={handleOpenChat}
        aria-label="Open AI chat assistant"
        title="Ask AI Assistant"
      >
        💬
      </button>

      {/* "Ask AI about this" button on text selection */}
      {selection && (
        <button
          className={styles.selectionButton}
          style={{
            top: `${selection.position.y}px`,
            left: `${selection.position.x}px`,
          }}
          onClick={handleAskAboutSelection}
        >
          Ask AI about this
        </button>
      )}

      {/* Chat Modal */}
      <ChatModal
        isOpen={isOpen}
        onClose={handleCloseChat}
        messages={chatHistory}
        isLoading={isLoading}
        error={error}
        onSubmitQuery={handleSubmitQuery}
        rateLimitSeconds={rateLimitSeconds}
        selection={selection}
        onClearHistory={handleClearHistory}
      />
    </>
  );
};

export default RAGChatbot;
