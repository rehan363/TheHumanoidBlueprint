/**
 * TypeScript interfaces for RAG Chatbot components
 */

export interface Citation {
  source: string;
  chunk_index: number;
  total_chunks: number;
  relevance_score: number;
  file_path?: string;
  content_preview?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  timestamp: Date;
  processing_time_ms?: number;
}

export interface SelectedTextContext {
  text: string;
  file_path: string;
  selection_range?: {
    start: number;
    end: number;
  };
}

export interface QueryRequest {
  query: string;
  query_type: 'full_text' | 'text_selection';
  context?: SelectedTextContext;
  session_id?: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  query_type: 'full_text' | 'text_selection';
  session_id?: string;
  processing_time_ms?: number;
  timestamp: string;
}

export interface ErrorResponse {
  error: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
}

export interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  isOpen: boolean;
}
