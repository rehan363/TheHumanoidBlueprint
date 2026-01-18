/**
 * @file useSessionStorage.ts
 * @description Custom React hook for persisting state to sessionStorage,
 * with an added feature to fetch initial chat history from a backend API
 * and fallback to local storage.
 */

import { useState, useEffect, useCallback } from 'react';
import { ChatMessage, Citation } from '../components/RAGChatbot/types';

const API_BASE_URL = '/api/chat';

/**
 * `useSessionStorage` is a custom React hook for managing state that needs to be
 * persisted across browser sessions (within a tab) and potentially initialized
 * from a backend API.
 *
 * It attempts to load chat history from a backend API using a session ID.
 * If the API call fails or no session ID is found, it falls back to loading
 * from browser's `sessionStorage`.
 *
 * @template T The type of the state to be stored.
 * @param {string} key The key under which the data is stored in `sessionStorage`.
 * @param {T} initialValue The initial value to use if no value is found in `sessionStorage` or from the API.
 * @returns {[T, (value: T | ((val: T) => T)) => void]} A tuple containing the current stored value and a function to update it.
 */
function useSessionStorage<T>(key: string, initialValue: T): [T, (value: T | ((val: T) => T)) => void] {
  // State to store our value. Initialized with initialValue to prevent hydration mismatches.
  const [storedValue, setStoredValue] = useState<T>(initialValue);

  /**
   * Reads the value from `sessionStorage` for the given key.
   * @returns {T} The parsed value from `sessionStorage` or the `initialValue` if not found or an error occurs.
   */
  const readValue = useCallback((): T => {
    // Prevent build error "window is undefined" but keep working for SSR
    if (typeof window === 'undefined') {
      return initialValue;
    }
    try {
      const item = window.sessionStorage.getItem(key);
      return item ? (JSON.parse(item) as T) : initialValue;
    } catch (error) {
      console.warn(`Error reading sessionStorage key “${key}”:`, error);
      return initialValue;
    }
  }, [initialValue, key]);

  /**
   * Fetches chat history from the backend API using the stored `sessionId`.
   * If successful, updates the component state and `sessionStorage`.
   * Falls back to `readValue()` from `sessionStorage` if the API call fails or no `sessionId` is found.
   */
  const fetchHistoryFromBackend = useCallback(async () => {
    if (typeof window === 'undefined') {
      return;
    }

    const sessionId = sessionStorage.getItem('sessionId');
    if (!sessionId) {
      console.log('No session ID found, skipping backend history fetch.');
      setStoredValue(readValue()); // Fallback to local storage if no session ID
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/history/${sessionId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: ChatMessage[] = await response.json();

      // Ensure timestamps are Date objects for client-side use
      const messagesWithDates = data.map(msg => ({
        ...msg,
        timestamp: new Date(msg.timestamp),
      }));

      // Update state with fetched history. Type assertion `as unknown as T` is used
      // assuming T can be assigned from ChatMessage[] when key is 'rag-chat-history'.
      setStoredValue(messagesWithDates as unknown as T);
      // Also update sessionStorage with fetched data to maintain consistency and serve as a quick fallback
      sessionStorage.setItem(key, JSON.stringify(messagesWithDates));
      console.log('History fetched from backend and stored.');
    } catch (error) {
      console.error('Failed to fetch history from backend, falling back to sessionStorage:', error);
      setStoredValue(readValue()); // Fallback to local storage on error
    }
  }, [key, readValue]);

  // Effect to read initial value from backend or sessionStorage on mount
  useEffect(() => {
    fetchHistoryFromBackend();
  }, [fetchHistoryFromBackend]);

  /**
   * Updates the stored value in both React state and `sessionStorage`.
   * @param {T | ((val: T) => T)} value The new value to store, or a function that receives the previous value and returns the new value.
   */
  const setValue = useCallback((value: T | ((val: T) => T)) => {
    // Prevent build error "window is undefined" but keeps working for SSR
    if (typeof window == 'undefined') {
      console.warn(
        `Tried setting sessionStorage key “${key}” even though environment is not a client`,
      );
      return;
    }

    try {
      // Allow value to be a function so we have same API as useState
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      // Save state
      setStoredValue(valueToStore);
      // Save to session storage
      window.sessionStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.warn(`Error setting sessionStorage key “${key}”:`, error);
    }
  }, [key, storedValue]);
  

  return [storedValue, setValue];
}

export default useSessionStorage;
