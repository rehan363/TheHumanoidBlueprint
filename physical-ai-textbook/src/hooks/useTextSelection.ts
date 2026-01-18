/**
 * @file useTextSelection.ts
 * @description Custom React hook for tracking user text selection on the page
 * and providing its content and screen position.
 */

import { useState, useEffect, useCallback } from 'react';

/**
 * Interface for the data representing a user's text selection.
 */
export interface SelectionData {
  /**
   * The actual text content that was selected by the user.
   */
  text: string;
  /**
   * The x and y coordinates on the screen, typically used for positioning a UI element near the selection.
   * `x`: The horizontal position.
   * `y`: The vertical position.
   */
  position: { x: number; y: number };
}

/**
 * `useTextSelection` is a React hook that detects and provides information
 * about the user's text selection within the document.
 *
 * It listens to `mouseup` events to capture the selection and `mousedown`
 * events to clear it.
 *
 * @returns {{selection: SelectionData | null, clearSelection: () => void}} An object containing:
 * - `selection`: An object of type `SelectionData` if text is selected, otherwise `null`.
 * - `clearSelection`: A function to manually clear the current text selection.
 */
export const useTextSelection = () => {
  const [selection, setSelection] = useState<SelectionData | null>(null);

  /**
   * Handles the `mouseup` event to capture text selection.
   * If text is selected, it updates the `selection` state with the text content
   * and its calculated screen position. If no text is selected, it clears the `selection` state.
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

  /**
   * Clears the current text selection by setting the `selection` state to `null`.
   */
  const clearSelection = useCallback(() => {
    setSelection(null);
  }, []);

  return { selection, clearSelection };
};
