/**
 * Citation component for displaying source references
 */

import React from 'react';
import type { Citation as CitationType } from './types';
import styles from './styles.module.css';

interface CitationProps {
  citation: CitationType;
}

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
