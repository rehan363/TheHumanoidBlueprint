import React from 'react';
import { useSession, signOut } from '../../lib/auth-client';
import Link from '@docusaurus/Link';

export default function AuthButton() {
    const { data: session, isPending } = useSession();

    if (isPending) {
        return <div style={{ width: '40px' }} />;
    }

    if (session) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                    {session.user.name}
                </span>
                <button
                    onClick={() => signOut()}
                    className="button button--primary button--sm"
                    style={{ padding: '4px 12px' }}
                >
                    Logout
                </button>
            </div>
        );
    }

    return (
        <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'center' }}>
            <Link
                className="navbar__item navbar__link"
                to="/login"
                style={{ fontWeight: 600, color: 'var(--brand-primary)' }}
            >
                Sign In
            </Link>
            <Link
                className="button button--secondary button--sm"
                to="/signup"
                style={{
                    borderRadius: '8px',
                    padding: '8px 20px',
                    fontWeight: 700,
                    textTransform: 'none'
                }}
            >
                Sign Up
            </Link>
        </div>
    );
}
