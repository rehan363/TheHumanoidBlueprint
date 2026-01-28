/**
 * @file infrastructure.test.ts
 * @description Property-based tests for infrastructure and dependency setup
 * Feature: interactive-textbook-ui, Property 26: Page Load Performance
 */

import * as fc from 'fast-check';

describe('Infrastructure Setup', () => {
  // Feature: interactive-textbook-ui, Property 26: Page Load Performance
  test('should have all required dependencies available', () => {
    fc.assert(
      fc.property(fc.constantFrom('react', 'react-dom', '@reduxjs/toolkit', 'three'), (dependency) => {
        // Test that each dependency can be required/imported
        expect(() => {
          require(dependency);
        }).not.toThrow();
      })
    );
  });

  test('should have Redux store configured correctly', () => {
    const { store } = require('../store');
    
    expect(store).toBeDefined();
    expect(store.getState).toBeDefined();
    expect(store.dispatch).toBeDefined();
    
    // Test initial state structure
    const state = store.getState();
    expect(state).toHaveProperty('progress');
    expect(state).toHaveProperty('user');
    expect(state).toHaveProperty('ui');
  });

  test('should have Three.js available for 3D components', () => {
    const THREE = require('three');
    
    expect(THREE).toBeDefined();
    expect(THREE.Scene).toBeDefined();
    expect(THREE.WebGLRenderer).toBeDefined();
    expect(THREE.PerspectiveCamera).toBeDefined();
  });

  test('should have Sandpack available for code playgrounds', () => {
    const sandpack = require('@codesandbox/sandpack-react');
    
    expect(sandpack).toBeDefined();
    expect(sandpack.Sandpack).toBeDefined();
  });

  test('should have React Three Fiber available', () => {
    const fiber = require('@react-three/fiber');
    
    expect(fiber).toBeDefined();
    expect(fiber.Canvas).toBeDefined();
  });

  test('should have accessibility testing tools available', () => {
    const axe = require('axe-core');
    
    expect(axe).toBeDefined();
    expect(axe.run).toBeDefined();
  });

  // Property test for performance baseline
  test('dependency loading should be performant', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(
          'react',
          'react-dom',
          '@reduxjs/toolkit',
          'three',
          '@codesandbox/sandpack-react',
          '@react-three/fiber'
        ),
        (dependency) => {
          const startTime = performance.now();
          require(dependency);
          const endTime = performance.now();
          const loadTime = endTime - startTime;
          
          // Dependencies should load within reasonable time (100ms)
          expect(loadTime).toBeLessThan(100);
        }
      ),
      { numRuns: 10 } // Reduced runs for performance test
    );
  });
});