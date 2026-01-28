/**
 * Root component - Docusaurus swizzled component
 * This wraps the entire app and adds authentication gatekeeping
 */

import React, { useEffect, useState } from 'react';
import { useSession } from '../lib/auth-client';
import { RAGChatbot } from '../components/RAGChatbot';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

// Helper to check if we are on an auth page
const isAuthPage = () => {
  if (!ExecutionEnvironment.canUseDOM) return false;
  const path = window.location.pathname;
  return path.includes('/login') || path.includes('/signup');
};

export default function Root({ children }: { children: React.ReactNode }): JSX.Element {
  const { data: session, isPending } = useSession();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Don't render anything until mounted on client to avoid hydration mismatch
  if (!mounted) {
    return <div style={{ visibility: 'hidden' }}>{children}</div>;
  }

  // Show loading state while session is being checked
  if (isPending) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        background: 'var(--ifm-background-color)'
      }}>
        <div className="loading-spinner">Loading Student Session...</div>
      </div>
    );
  }

  // Gatekeeping logic: Allow viewing the landing page without login.
  // Redirect to login only when trying to access documentation or personal features.
  const isLandingPage = ExecutionEnvironment.canUseDOM &&
    (window.location.pathname === '/TheHumanoidBlueprint/' ||
      window.location.pathname === '/TheHumanoidBlueprint' ||
      window.location.pathname === '/');

  const accessDenied = !session && !isAuthPage() && !isLandingPage;

  if (accessDenied) {
    if (ExecutionEnvironment.canUseDOM) {
      window.location.href = window.location.pathname.includes('/TheHumanoidBlueprint/')
        ? '/TheHumanoidBlueprint/login'
        : '/login';
    }
    return null;
  }

  return (
    <>
      {children}
      {session && <RAGChatbot />}
    </>
  );
}
