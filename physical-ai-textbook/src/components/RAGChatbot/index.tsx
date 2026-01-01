/**
 * Main RAG Chatbot component with floating action button
 */

import React, { useState, useEffect } from 'react';
import { ChatModal } from './ChatModal';
import { useChatAPI } from '../../hooks/useChatAPI';
import { useSessionStorage } from '../../hooks/useSessionStorage';
import styles from './styles.module.css';

export const RAGChatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { messages, isLoading, error, sendQuery, clearError, clearMessages, rateLimitSeconds } = useChatAPI();
  const { clearHistory } = useSessionStorage(messages);

  // Load history on mount
  useEffect(() => {
    try {
      const stored = sessionStorage.getItem('rag-chat-history');
      if (stored) {
        // Messages are already loaded by the hook
        console.log('Chat history loaded');
      }
    } catch (err) {
      console.error('Error loading history:', err);
    }
  }, []);

  const handleOpenChat = () => {
    setIsOpen(true);
    clearError();
  };

  const handleCloseChat = () => {
    setIsOpen(false);
  };

  const handleSubmitQuery = async (query: string) => {
    await sendQuery(query);
  };

  const handleClearHistory = () => {
    clearMessages();
    clearHistory();
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

      {/* Chat Modal */}
      <ChatModal
        isOpen={isOpen}
        onClose={handleCloseChat}
        messages={messages}
        isLoading={isLoading}
        error={error}
        onSubmitQuery={handleSubmitQuery}
        rateLimitSeconds={rateLimitSeconds}
      />
    </>
  );
};

export default RAGChatbot;
