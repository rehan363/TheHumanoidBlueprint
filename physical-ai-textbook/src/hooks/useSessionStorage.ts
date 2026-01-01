/**
 * Custom hook for persisting chat history in sessionStorage
 */

import { useEffect } from 'react';
import type { ChatMessage } from '../components/RAGChatbot/types';

const STORAGE_KEY = 'rag-chat-history';

export function useSessionStorage(messages: ChatMessage[]) {
  // Load messages from sessionStorage on mount
  useEffect(() => {
    const loadMessages = (): ChatMessage[] => {
      try {
        const stored = sessionStorage.getItem(STORAGE_KEY);
        if (!stored) return [];

        const parsed = JSON.parse(stored);

        // Convert timestamp strings back to Date objects
        return parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        }));
      } catch (error) {
        console.error('Error loading chat history:', error);
        return [];
      }
    };

    const loadedMessages = loadMessages();
    return loadedMessages;
  }, []); // Only run on mount

  // Save messages to sessionStorage whenever they change
  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch (error) {
      console.error('Error saving chat history:', error);
    }
  }, [messages]);

  // Function to clear history
  const clearHistory = () => {
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error('Error clearing chat history:', error);
    }
  };

  return { clearHistory };
}
