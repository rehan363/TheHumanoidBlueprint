/**
 * Root component - Docusaurus swizzled component
 * This wraps the entire app and adds the global RAG chatbot
 */

import React from 'react';
import { RAGChatbot } from '../components/RAGChatbot';

export default function Root({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <>
      {children}
      <RAGChatbot />
    </>
  );
}
