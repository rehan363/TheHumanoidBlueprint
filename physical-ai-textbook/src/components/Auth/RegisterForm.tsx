import React, { useState } from 'react';
import { signUp } from '../../lib/auth-client';
import styles from './Auth.module.css';

export default function RegisterForm() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');
    const [pythonExp, setPythonExp] = useState(5);
    const [hardwareExp, setHardwareExp] = useState(5);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSignUp = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            const { data, error } = await signUp.email({
                email,
                password,
                name,
                callbackURL: "/TheHumanoidBlueprint/",
                fetchOptions: {
                    body: {
                        python_experience: pythonExp,
                        hardware_experience: hardwareExp,
                        onboarding_complete: true
                    }
                }
            });

            if (error) {
                setError(error.message || 'Failed to sign up');
            } else {
                window.location.href = '/TheHumanoidBlueprint/';
            }
        } catch (err) {
            setError('An unexpected error occurred');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={styles.authCard}>
            <h2>Create Student Account</h2>
            <p>Join the Physical AI journey and personalize your learning.</p>

            <form onSubmit={handleSignUp}>
                <div className={styles.inputGroup}>
                    <label>Full Name</label>
                    <input type="text" value={name} onChange={(e) => setName(e.target.value)} required placeholder="John Doe" />
                </div>

                <div className={styles.inputGroup}>
                    <label>Email</label>
                    <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required placeholder="john@example.com" />
                </div>

                <div className={styles.inputGroup}>
                    <label>Password</label>
                    <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required placeholder="••••••••" />
                </div>

                <hr className={styles.divider} />

                <div className={styles.profileSection}>
                    <h3>Background Profile</h3>
                    <p className={styles.helpText}>Adjust sliders to describe your current experience level (0-10).</p>

                    <div className={styles.rangeGroup}>
                        <label>Python Proficiency: {pythonExp}</label>
                        <input type="range" min="0" max="10" value={pythonExp} onChange={(e) => setPythonExp(parseInt(e.target.value))} />
                        <div className={styles.rangeLabels}>
                            <span>Beginner</span>
                            <span>Expert</span>
                        </div>
                    </div>

                    <div className={styles.rangeGroup}>
                        <label>Hardware/Robotics Exp: {hardwareExp}</label>
                        <input type="range" min="0" max="10" value={hardwareExp} onChange={(e) => setHardwareExp(parseInt(e.target.value))} />
                        <div className={styles.rangeLabels}>
                            <span>None</span>
                            <span>Roboticist</span>
                        </div>
                    </div>
                </div>

                {error && <div className={styles.errorMessage}>{error}</div>}

                <button type="submit" className="button button--primary button--block" disabled={loading}>
                    {loading ? 'Creating Account...' : 'Get Started'}
                </button>
            </form>
        </div>
    );
}
