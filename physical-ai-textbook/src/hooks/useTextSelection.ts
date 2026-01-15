/**
 * Custom hook for tracking user text selection.
 * Returns the selected text and its position on the screen.
 */

import { useState, useEffect, useCallback } from 'react';

/**
 * Data returned by the useTextSelection hook.
 *
 * @property text - The selected text content.
 * @property position - The x/y coordinates for positioning UI near the selection.
 */
export interface SelectionData {
  text: string;
  position: { x: number; y: number };
}

/**
 * A React hook that detects and returns information about user text selection.
 *
 * @returns An object containing the current selection data and a function to clear it.
 */
export const useTextSelection = () => {
  const [selection, setSelection] = useState<SelectionData | null>(null);

  /**
   * Handles the mouseup event to capture text selection.
   */
  const handleSelection = useCallback(() => {
    const selectionObj = window.getSelection();
    const selectedText = selectionObj?.toString().trim();

    if (selectedText) {
      const range = selectionObj.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      setSelection({
        text: selectedText,
        // Position the popover just below the selection
        position: {
          x: rect.left + window.scrollX,
          y: rect.bottom + window.scrollY + 5,
        },
      });
    } else {
      // If there's no selection, or selection is cleared, reset the state
      setSelection(null);
    }
  }, []);

  useEffect(() => {
    // We listen for 'mouseup' to capture the final selection
    document.addEventListener('mouseup', handleSelection);
    // We also listen for 'mousedown' to clear the selection when the user starts a new action
    document.addEventListener('mousedown', () => setSelection(null));

    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('mousedown', () => setSelection(null));
    };
  }, [handleSelection]);

  const clearSelection = useCallback(() => {
    setSelection(null);
  }, []);

  return { selection, clearSelection };
};
