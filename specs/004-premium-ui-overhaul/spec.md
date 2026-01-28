# Specification: 004-premium-ui-overhaul (STRICT COMPLIANCE)

## Overview
Implement a world-class, premium aesthetic for the Physical AI & Humanoid Robotics Textbook. The design will follow the "Agent Factory" inspiration provided by the user, utilizing a deep purple base (`#461D34`) with modern high-contrast accents, custom animations, and a sophisticated theme that supports both light and dark modes.

## Goals
- **Premium Aesthetic**: Implement a "high-end tech" look using the Deep Plum (`#461D34`) palette.
- **Brand Identity**: Use the provided book image in the hero section and ensure the favicon is correctly integrated.
- **Dynamic UX**: Add smooth animations (transitions, hover effects, slide-ins) to make the interface feel alive.
- **Consistency**: Ensure the new design system flows through the homepage, documentation, and authentication pages.

## Success Criteria
- [ ] Homepage hero section features the book image on the right with a professional layout.
- [ ] Color palette centered around `#461D34` with complementary high-contrast colors (e.g., Electric Blue).
- [ ] Responsive design that looks premium on both mobile and desktop.
- [ ] Functioning light and dark modes with proper contrast and theme persistence.
- [ ] Hover effects and entrance animations applied to key components (cards, buttons).

## Tech Stack
- **Styling**: Vanilla CSS (Docusaurus `custom.css` and CSS Modules).
- **Icons**: Lucide React or similar (if needed) or custom SVGs.
- **Animations**: CSS Keyframes and Transitions.

## Design Tokens
- **Primary**: `#461D34` (Deep Plum)
- **Primary-Light**: `#6b2d4f`
- **Action/Highlight**: `#3b82f6` (Vibrant Blue)
- **Dark Background**: `#0a0508`
- **Surface**: `#1a0f14` (for cards/modals)

## Implementation Plan
1. **Core Styles**: Update `custom.css` with the new color system and global resets.
2. **Hero Section**: Redesign `index.tsx` homepage to match the "Agent Factory" hero layout with the book image.
3. **Feature Cards**: Update `HomepageFeatures` components with the new border-glow and badge aesthetic.
4. **Auth Styling**: Aesthetic pass on the Login/Signup forms to match the premium theme.
5. **Animations**: Add global entrance animations for page elements.
