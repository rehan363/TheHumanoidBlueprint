/**
 * @file useChatAPI.ts
 * @description Custom React hook for interacting with the RAG chat API.
 * It handles sending queries, managing loading states, errors, rate limiting,
 * and persisting session information.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { ChatMessage, QueryRequest, QueryResponse, ErrorResponse } from '../components/RAGChatbot/types';
import { authClient } from '../lib/auth-client';

/**
 * Generates a UUID v4 string.
 * @returns {string} A UUID v4 string.
 */
function uuidv4(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/**
 * Represents the return value of the `useChatAPI` hook.
 */
interface UseChatAPIReturn {
  /**
   * Indicates if a chat query is currently being processed.
   */
  isLoading: boolean;
  /**
   * Any error message from the API or network, or null if no error.
   */
  error: string | null;
  /**
   * Function to send a query to the RAG backend.
   * @param query The user's query string.
   * @param queryType The type of query (e.g., 'full_text', 'text_selection').
   * @param context Optional context for the query (e.g., selected text data).
   */
  sendQuery: (query: string, queryType?: 'full_text' | 'text_selection', context?: any) => Promise<void>;
  /**
   * Function to clear any active error messages.
   */
  clearError: () => void;
  /**
   * The number of seconds remaining until the rate limit resets.
   */
  rateLimitSeconds: number;
}

const API_BASE_URL = '/api/chat';

/**
 * `useChatAPI` is a custom React hook that provides an interface to the RAG chatbot backend.
 * It manages the lifecycle of chat API requests, including:
 * - Handling loading and error states.
 * - Implementing client-side rate limiting with retry-after headers.
 * - Generating and persisting a session ID for conversation history.
 * - Updating the chat messages state with user and assistant responses.
 *
 * @param {ChatMessage[]} messages The current list of chat messages.
 * @param {(newMessages: ChatMessage[]) => void} setMessages Function to update the list of chat messages.
 * @returns {UseChatAPIReturn} An object containing the current loading state, error, functions to send queries and clear errors, and the rate limit countdown.
 */
export function useChatAPI(messages: ChatMessage[], setMessages: (newMessages: ChatMessage[]) => void): UseChatAPIReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rateLimitSeconds, setRateLimitSeconds] = useState(0);
  const rateLimitTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  useEffect(() => {
    let currentSessionId = sessionStorage.getItem('sessionId');
    if (!currentSessionId) {
      currentSessionId = uuidv4();
      sessionStorage.setItem('sessionId', currentSessionId);
    }
    setSessionId(currentSessionId);
  }, []);

  /**
   * Starts a countdown timer for rate limiting.
   * @param {number} seconds The number of seconds to countdown from.
   */
  const startRateLimitCountdown = useCallback((seconds: number) => {
    setRateLimitSeconds(seconds);

    if (rateLimitTimerRef.current) {
      clearInterval(rateLimitTimerRef.current);
    }

    rateLimitTimerRef.current = setInterval(() => {
      setRateLimitSeconds((prev) => {
        if (prev <= 1) {
          if (rateLimitTimerRef.current) {
            clearInterval(rateLimitTimerRef.current);
            rateLimitTimerRef.current = null;
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  }, []);

  /**
   * Sends a chat query to the RAG backend API.
   * Handles loading states, errors (including rate limiting and service unavailability),
   * and updates the chat messages with the assistant's response.
   *
   * @param {string} query The user's query string.
   * @param {'full_text' | 'text_selection'} [queryType='full_text'] The type of query.
   * @param {any} [context] Optional context data for the query.
   * @returns {Promise<void>} A promise that resolves when the query is processed.
   */
  const sendQuery = useCallback(
    async (query: string, queryType: 'full_text' | 'text_selection' = 'full_text', context?: any) => {
      // Clear previous error
      setError(null);
      setIsLoading(true);

      // Add user message immediately
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: query,
        timestamp: new Date(),
      };

      setMessages([...messages, userMessage]);

      try {
        // Get the current session token from Better-Auth cookies
        const tokenMatch = document.cookie.match(/better-auth\.session_token=([^;]+)/) ||
          document.cookie.match(/better-auth\.session-token=([^;]+)/);
        const token = tokenMatch ? tokenMatch[1] : null;

        const requestBody: QueryRequest = {
          query,
          query_type: queryType,
          context,
          session_id: sessionId,
        };

        const response = await fetch(`${API_BASE_URL}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': token ? `Bearer ${token}` : '',
          },
          body: JSON.stringify(requestBody),
        });

        // Handle rate limiting (429)
        if (response.status === 429) {
          const retryAfter = response.headers.get('Retry-After');
          const seconds = retryAfter ? parseInt(retryAfter, 10) : 60;
          startRateLimitCountdown(seconds);
          setError('Rate limit exceeded. Please wait before sending another message.');
          setIsLoading(false);
          return;
        }

        // Handle service unavailable (503)
        if (response.status === 503) {
          setError('AI assistant temporarily offline. Browse content normally.');
          setIsLoading(false);
          return;
        }

        // Handle other errors
        if (!response.ok) {
          const errorData: ErrorResponse = await response.json();
          setError(errorData.message || 'An error occurred while processing your query.');
          setIsLoading(false);
          return;
        }

        // Parse successful response
        const data: QueryResponse = await response.json();

        // Update session ID if backend provided a different one
        if (data.session_id && data.session_id !== sessionId) {
          setSessionId(data.session_id);
          sessionStorage.setItem('sessionId', data.session_id);
        }

        // Add assistant message
        const assistantMessage: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: data.answer,
          citations: data.citations,
          timestamp: new Date(data.timestamp),
          processing_time_ms: data.processing_time_ms,
        };

        setMessages([...messages, userMessage, assistantMessage]); // Update messages with both user and assistant
      } catch (err) {
        console.error('Chat API error:', err);

        // Network error or service down
        if (err instanceof TypeError && err.message === 'Failed to fetch') {
          setError('AI assistant temporarily offline. Browse content normally.');
        } else {
          setError('An unexpected error occurred. Please try again.');
        }
      } finally {
        setIsLoading(false);
      }
    },
    [startRateLimitCountdown, messages, setMessages, sessionId]
  );

  /**
   * Clears the current error message.
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    isLoading,
    error,
    sendQuery,
    clearError,
    rateLimitSeconds,
  };
}
