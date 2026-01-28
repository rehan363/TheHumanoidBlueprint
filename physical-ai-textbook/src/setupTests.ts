import '@testing-library/jest-dom';

// Mock IntersectionObserver for Three.js components
(global as any).IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
};

// Mock ResizeObserver for responsive components
(global as any).ResizeObserver = class ResizeObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
};

// Mock WebGL context for Three.js
HTMLCanvasElement.prototype.getContext = jest.fn((contextType) => {
  if (contextType === 'webgl' || contextType === 'webgl2') {
    return {
      canvas: {},
      drawingBufferWidth: 1024,
      drawingBufferHeight: 768,
      getExtension: jest.fn(),
      getParameter: jest.fn(),
      createShader: jest.fn(),
      shaderSource: jest.fn(),
      compileShader: jest.fn(),
      createProgram: jest.fn(),
      attachShader: jest.fn(),
      linkProgram: jest.fn(),
      useProgram: jest.fn(),
      createBuffer: jest.fn(),
      bindBuffer: jest.fn(),
      bufferData: jest.fn(),
      enableVertexAttribArray: jest.fn(),
      vertexAttribPointer: jest.fn(),
      drawArrays: jest.fn(),
      viewport: jest.fn(),
      clearColor: jest.fn(),
      clear: jest.fn(),
    };
  }
  return null;
}) as any;

// Mock Speech Recognition API for voice input testing
(global as any).SpeechRecognition = class SpeechRecognition {
  constructor() {}
  start() {}
  stop() {}
  abort() {}
};

(global as any).webkitSpeechRecognition = (global as any).SpeechRecognition;