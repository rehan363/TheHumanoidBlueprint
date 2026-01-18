/**
 * @file Citation.tsx
 * @description React component for displaying a single source citation within the RAG chatbot.
 * This component visualizes the source, a content preview, and the relevance score of a citation.
 */

import React from 'react';
import type { Citation as CitationType } from './types';
import styles from './styles.module.css';

/**
 * Props for the Citation component.
 */
interface CitationProps {
  /**
   * The citation object containing source, content preview, and relevance score.
   */
  citation: CitationType;
}

/**
 * `Citation` is a functional React component that renders details of a single source citation.
 * It takes a `CitationType` object as a prop and displays its source, a preview of the content,
 * and its calculated relevance score.
 *
 * @param {CitationProps} props The properties for the Citation component.
 * @returns {React.FC<CitationProps>} The rendered citation component.
 */
export const Citation: React.FC<CitationProps> = ({ citation }) => {
  return (
    <div className={styles.citation}>
      <div className={styles.citationSource}>
        📖 {citation.source}
      </div>
      {citation.content_preview && (
        <div className={styles.citationPreview}>
          "{citation.content_preview}..."
        </div>
      )}
      <div className={styles.citationScore}>
        Relevance: {(citation.relevance_score * 100).toFixed(0)}%
      </div>
    </div>
  );
};
