import React, { useState } from 'react';
import { signIn } from '../../lib/auth-client';
import styles from './Auth.module.css';

export default function LoginForm() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            const { data, error } = await signIn.email({
                email,
                password,
                callbackURL: "/panaversity-hackathon-I/",
            });

            if (error) {
                setError(error.message || 'Failed to login');
            } else {
                window.location.href = '/panaversity-hackathon-I/';
            }
        } catch (err) {
            setError('An unexpected error occurred');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={styles.authCard}>
            <h2>Student Login</h2>
            <p>Welcome back! Let's continue your robotics training.</p>

            <form onSubmit={handleLogin}>
                <div className={styles.inputGroup}>
                    <label>Email</label>
                    <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required placeholder="john@example.com" />
                </div>

                <div className={styles.inputGroup}>
                    <label>Password</label>
                    <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required placeholder="••••••••" />
                </div>

                {error && <div className={styles.errorMessage}>{error}</div>}

                <button type="submit" className="button button--primary button--block" disabled={loading}>
                    {loading ? 'Logging In...' : 'Login'}
                </button>

                <div className={styles.footer}>
                    Don't have an account? <a href="/signup">Sign up</a>
                </div>
            </form>
        </div>
    );
}
