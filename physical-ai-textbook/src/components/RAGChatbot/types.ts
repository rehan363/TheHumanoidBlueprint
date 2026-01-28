/**
 * @file types.ts
 * @description Defines TypeScript interfaces for the RAG Chatbot components,
 * including data structures for messages, citations, queries, and responses.
 */

/**
 * Represents a source citation for an AI-generated answer.
 */
export interface Citation {
  /**
   * The source document or section from which the information was retrieved (e.g., "Week 3, ROS 2 Architecture").
   */
  source: string;
  /**
   * The index of the chunk within its document.
   */
  chunk_index: number;
  /**
   * The total number of chunks in the source document.
   */
  total_chunks: number;
  /**
   * A score indicating the relevance of this citation to the query (0.0 to 1.0).
   */
  relevance_score: number;
  /**
   * Optional: The file path of the source document.
   */
  file_path?: string;
  /**
   * Optional: A snippet of content from the source chunk.
   */
  content_preview?: string;
}

/**
 * Represents a single message in the chat conversation.
 */
export interface ChatMessage {
  /**
   * A unique identifier for the message.
   */
  id: string;
  /**
   * The role of the sender, either 'user' or 'assistant'.
   */
  role: 'user' | 'assistant';
  /**
   * The textual content of the message.
   */
  content: string;
  /**
   * Optional: A list of citations relevant to the assistant's message.
   */
  citations?: Citation[];
  /**
   * The timestamp when the message was created.
   */
  timestamp: Date;
  /**
   * Optional: The time taken in milliseconds to process the assistant's response.
   */
  processing_time_ms?: number;
}

/**
 * Represents the context of a user's text selection from the document.
 */
export interface SelectedTextContext {
  /**
   * The actual text selected by the user.
   */
  text: string;
  /**
   * The file path of the document where the text was selected.
   */
  file_path: string;
  /**
   * Optional: The start and end positions of the selection within the document.
   */
  selection_range?: {
    start: number;
    end: number;
  };
}

/**
 * Represents the structure of a query request sent to the RAG backend.
 */
export interface QueryRequest {
  /**
   * The user's query string.
   */
  query: string;
  /**
   * The type of query, either 'full_text' for general questions or 'text_selection' for context-specific queries.
   */
  query_type: 'full_text' | 'text_selection';
  /**
   * Optional: The context of selected text, required if `query_type` is 'text_selection'.
   */
  context?: SelectedTextContext;
  /**
   * Optional: A unique session identifier for the conversation.
   */
  session_id?: string;
  /**
   * Optional: Better-Auth session token for personalization.
   */
  auth_token?: string;
}

/**
 * Represents the structure of a query response received from the RAG backend.
 */
export interface QueryResponse {
  /**
   * The AI-generated answer.
   */
  answer: string;
  /**
   * A list of citations supporting the answer.
   */
  citations: Citation[];
  /**
   * The type of query that was processed.
   */
  query_type: 'full_text' | 'text_selection';
  /**
   * Optional: The session identifier for the conversation.
   */
  session_id?: string;
  /**
   * Optional: The time taken in milliseconds to process the query.
   */
  processing_time_ms?: number;
  /**
   * The timestamp when the response was generated (ISO 8601 string).
   */
  timestamp: string;
}

/**
 * Represents the structure of an error response from the API.
 */
export interface ErrorResponse {
  /**
   * A general error code or type.
   */
  error: string;
  /**
   * A human-readable message describing the error.
   */
  message: string;
  /**
   * Optional: Additional details about the error.
   */
  details?: Record<string, any>;
  /**
   * The timestamp when the error occurred (ISO 8601 string).
   */
  timestamp: string;
}

/**
 * Represents the state structure of the chat interface.
 */
export interface ChatState {
  /**
   * The list of messages in the current conversation.
   */
  messages: ChatMessage[];
  /**
   * Indicates if the chat is currently loading a response.
   */
  isLoading: boolean;
  /**
   * Any error message to display to the user.
   */
  error: string | null;
  /**
   * Indicates if the chat modal is open.
   */
  isOpen: boolean;
}
