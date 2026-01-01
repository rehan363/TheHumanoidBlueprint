/**
 * Custom hook for chat API client with error handling and rate limiting
 */

import { useState, useCallback, useRef } from 'react';
import type { QueryRequest, QueryResponse, ErrorResponse, ChatMessage } from '../components/RAGChatbot/types';

interface UseChatAPIReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  sendQuery: (query: string, queryType?: 'full_text' | 'text_selection', context?: any) => Promise<void>;
  clearError: () => void;
  clearMessages: () => void;
  rateLimitSeconds: number;
}

const API_BASE_URL = '/api/chat';

export function useChatAPI(): UseChatAPIReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rateLimitSeconds, setRateLimitSeconds] = useState(0);
  const rateLimitTimerRef = useRef<NodeJS.Timeout | null>(null);

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

      setMessages((prev) => [...prev, userMessage]);

      try {
        const requestBody: QueryRequest = {
          query,
          query_type: queryType,
          context,
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

        setMessages((prev) => [...prev, assistantMessage]);
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
    [startRateLimitCountdown]
  );

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendQuery,
    clearError,
    clearMessages,
    rateLimitSeconds,
  };
}
