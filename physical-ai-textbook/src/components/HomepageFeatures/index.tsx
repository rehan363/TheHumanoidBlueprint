import type { ReactNode } from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  badge: string;
  imgSrc: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Nervous System',
    badge: 'MODULE 1: ROS 2',
    imgSrc: require('@site/static/img/module-ros2.png').default,
    description: (
      <>
        Construct the foundational communication architecture of a humanoid robot.
        Learn real-time node orchestration, custom interfaces, and lifecycle management.
      </>
    ),
  },
  {
    title: 'Digital Twins',
    badge: 'MODULE 2: SIMULATION',
    imgSrc: require('@site/static/img/module-simulation.png').default,
    description: (
      <>
        Bridge the gap with high-fidelity simulation. Develop in Gazebo and Isaac Sim
        to test control algorithms and sensor fusion before deploying to physical hardware.
      </>
    ),
  },
  {
    title: 'Robot Cognitive Brain',
    badge: 'MODULE 3: AUTONOMY',
    imgSrc: require('@site/static/img/module-autonomy.png').default,
    description: (
      <>
        Implement the VLA (Vision-Language-Action) framework. Use deep learning to
        enable semantic understanding and autonomous decision making in physical space.
      </>
    ),
  },
  {
    title: 'Physical Deployment',
    badge: 'MODULE 4: HARDWARE',
    imgSrc: require('@site/static/img/module-hardware.png').default,
    description: (
      <>
        Bring it all to life on real actuators. Calibrate sensors, tune PID loops,
        and optimize model inference for real-time edge performance on humanoid platforms.
      </>
    ),
  },
];

function Feature({ title, badge, imgSrc, description }: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <div className={styles.featureCard}>
        <div className={styles.badge}>{badge}</div>
        <div className={styles.featureIconWrapper}>
          <img src={imgSrc} className={styles.featureImage} alt={title} />
        </div>
        <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
        <p className={styles.featureDescription}>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
