import React from 'react';
import Layout from '@theme/Layout';
import LoginForm from '../components/Auth/LoginForm';

export default function LoginPage() {
    return (
        <Layout title="Login" description="Login to your Physical AI Student account">
            <main className="auth-page">
                <LoginForm />
            </main>
        </Layout>
    );
}
