import React, { ReactNode, useState, useRef } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  const [rotate, setRotate] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    // Very subtle parallax effect (max 10 degrees)
    const rotateX = ((y - centerY) / centerY) * -10;
    const rotateY = ((x - centerX) / centerX) * 10;

    setRotate({ x: rotateX, y: rotateY });
  };

  const handleMouseLeave = () => {
    setRotate({ x: 0, y: 0 });
  };

  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className="row">
          <div className={clsx('col col--5', styles.heroTextContainer, 'animate-fade-in')}>
            <div className={styles.categoryLabel}>A COMPLETE TECHNICAL TEXTBOOK</div>
            <Heading as="h1" className={styles.heroTitle}>
              The Humanoid <br />
              <span className={styles.highlight}>Blueprint</span>
            </Heading>
            <p className={styles.heroSubtitle}>
              Mastering the Era of Physical AI.
              Bridging the gap between the digital brain and the physical body through
              Vision-Language-Action (VLA) models and robotic autonomy.
            </p>
            <div className={styles.buttons}>
              <Link
                className="button button--secondary button--lg"
                to="/module1-ros2/chapter1-foundations">
                START READING →
              </Link>
            </div>

            <div className={styles.coAuthors}>
              <small>POWERED BY NVIDIA ISAAC™</small>
            </div>
          </div>

          <div
            className={clsx('col col--7', styles.heroImageContainer, 'animate-fade-in')}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            ref={containerRef}
          >
            <div
              className={styles.bookWrapper}
              style={{
                transform: `rotateY(-15deg) rotateX(10deg) rotateX(${rotate.x}deg) rotateY(${rotate.y}deg)`,
                transition: rotate.x === 0 ? 'transform 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275)' : 'transform 0.1s ease-out'
              }}
            >
              <img
                src={require('@site/static/img/book-blueprint-3d-removebg-preview.png').default}
                alt="The Humanoid Blueprint Textbook"
                className={styles.bookImage}
              />
              <div className={styles.glowEffect}></div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics Textbook - A Spec-Driven Blueprint.">
      <HomepageHeader />
      <main>
        {/* Characteristics Section */}
        <section className={styles.characteristicsSection}>
          <div className="container">
            <div className="row">
              <div className="col col--6">
                <h2 className={styles.sectionTitle}>Why This Textbook?</h2>
                <p className={styles.sectionLead}>
                  Unlike traditional robotics manuals, this book focuses on the <strong>Physical AI Era</strong>.
                  We don't just teach code; we teach the orchestration of physical movement via intelligence.
                </p>
                <div className={styles.charList}>
                  <div className={styles.charItem}>
                    <div className={styles.charIcon}>✓</div>
                    <div>
                      <strong>Spec-Driven Development</strong>
                      <p>Build according to industry standards used by NVIDIA and Tesla.</p>
                    </div>
                  </div>
                  <div className={styles.charItem}>
                    <div className={styles.charIcon}>✓</div>
                    <div>
                      <strong>ISAAC Sim Integration</strong>
                      <p>Directly utilize NVIDIA's high-fidelity physics for digital twins.</p>
                    </div>
                  </div>
                  <div className={styles.charItem}>
                    <div className={styles.charIcon}>✓</div>
                    <div>
                      <strong>VLA Orchestration</strong>
                      <p>Master Vision-Language-Action models for natural autonomy.</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="col col--6">
                <div className={styles.infoBox}>
                  <h3>The Humanoid Blueprint</h3>
                  <p>
                    The journey to humanoid robotics requires an interdisciplinary approach.
                    This book bridges the gap between:
                  </p>
                  <ul className={styles.blueprintList}>
                    <li>Mechanical Kinematics & Bio-mimicry</li>
                    <li>Edge AI Inference (Jetson/Orin)</li>
                    <li>Low-latency Actuator Control</li>
                    <li>Sim-to-Real Transfer Learning</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className={styles.sectionDivider}>
          <span className={styles.dividerText}>CURRICULUM OVERVIEW</span>
        </div>
        <HomepageFeatures />

        {/* Pillars Section */}
        <section className={styles.pillarsSection}>
          <div className="container">
            <h2 className={styles.centerTitle}>The Pillars of Physical AI</h2>
            <div className="row">
              <div className="col col--4">
                <div className={styles.pillarCard}>
                  <div className={styles.pillarNumber}>01</div>
                  <h3>Perception</h3>
                  <p>High-speed visual processing and tactile feedback integration for environment sensing.</p>
                </div>
              </div>
              <div className="col col--4">
                <div className={styles.pillarCard}>
                  <div className={styles.pillarNumber}>02</div>
                  <h3>Cognition</h3>
                  <p>Large language models acting as the "Brain" to translate natural intent into robotic tasks.</p>
                </div>
              </div>
              <div className="col col--4">
                <div className={styles.pillarCard}>
                  <div className={styles.pillarNumber}>03</div>
                  <h3>Action</h3>
                  <p>Converting cognitive decisions into ultra-precise motor trajectories and force control.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Visual Showcase Section */}
        <section className={styles.visualShowcase}>
          <div className="container">
            <div className="row align-items-center">
              <div className="col col--6">
                <div className={styles.peaceRobotWrapper}>
                  <img
                    src={require('@site/static/img/robot-doing-peace-sign.jpg').default}
                    alt="Robot Interaction"
                    className={styles.largeRobotImage}
                  />
                </div>
              </div>
              <div className="col col--6">
                <h2 className={styles.sectionTitle}>Human-Robot Interaction</h2>
                <p className={styles.sectionLead}>
                  We prioritize robots that can safely and naturally interact with humans.
                  Our curriculum covers the social and physical etiquette of autonomous humanoids.
                </p>
                <div className={styles.infoBoxGold}>
                  <h3>Real-World Ready</h3>
                  <p>
                    From laboratory benchmarks to real-world deployment.
                    Master the safety protocols and interactive behaviors that make humanoid robots viable in human spaces.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Community / Final CTA */}
        <section className={styles.finalCta}>
          <div className="container">
            <div className={styles.ctaCard}>
              <div className={styles.ctaContent}>
                <h2>Ready to build the future?</h2>
                <p>Join thousands of engineers mastering the physical AI stack.</p>
                <Link className="button button--secondary button--lg" to="/module1-ros2/chapter1-foundations">
                  GET STARTED NOW
                </Link>
              </div>
              <div className={styles.ctaImage}>
                <img
                  src={require('@site/static/img/vecteezy_cartoon-robot-presenting-data-with-green-eyes-and-white_58693450.png').default}
                  alt="Join the mission"
                  className={styles.cartoonRobot}
                />
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
