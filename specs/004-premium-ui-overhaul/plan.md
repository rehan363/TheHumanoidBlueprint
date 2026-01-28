# Plan: 004-premium-ui-overhaul

## Phase 1: Foundation & Color System
- [ ] Update `src/css/custom.css` with the new design tokens.
- [ ] Define CSS variables for Light and Dark modes using `#461D34` as the base.
- [ ] Implement global scrollbar and typography (Inter/Outfit) styling.

## Phase 2: Homepage Transformation
- [ ] Modify `src/pages/index.tsx` hero section.
    - Move text to the left.
    - Add the book image (`/img/book-image.jpg`) to the right with a subtle glow/shadow.
    - Style the action buttons (Primary Blue / Outline White).
- [ ] Update `src/components/HomepageFeatures` cards.
    - Use the "Agent Factory" spectrum card look: thin borders, accent headers, and badges.
    - Add hover lift and border glow animations.

## Phase 3: Global Branding & Auth
- [ ] Update `docusaurus.config.ts` with the new favicon and color mode settings.
- [ ] Aesthetic pass on `src/components/Auth/Auth.module.css`.
    - Apply glassmorphism (blur + semi-transparent dark background).
    - Match the purple/blue accent colors.

## Phase 4: Animations & Polish
- [ ] Add `@keyframes` for fade-in and slide-up effects.
- [ ] Ensure the RAG Chatbot modal matches the new aesthetic (Plum headers, Blue accents).
- [ ] Final responsive check for mobile views.
