/**
 * Custom hook for chat API client with error handling, rate limiting, and session persistence.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { ChatMessage, QueryRequest, QueryResponse, ErrorResponse } from '../components/RAGChatbot/types';

// Simple UUID generator for session_id
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/**
 * Custom hook for chat API client with error handling, rate limiting, and session persistence.
 */

interface UseChatAPIReturn {
  isLoading: boolean;
  error: string | null;
  sendQuery: (query: string, queryType?: 'full_text' | 'text_selection', context?: any) => Promise<void>;
  clearError: () => void;
  rateLimitSeconds: number;
}

const API_BASE_URL = '/api/chat';

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

  // Start rate limit countdown
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
        const requestBody: QueryRequest = {
          query,
          query_type: queryType,
          context,
          session_id: sessionId, // Add session_id to the request
        };

        const response = await fetch(`${API_BASE_URL}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
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
