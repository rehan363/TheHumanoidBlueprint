import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Module 1: ROS 2 Foundations',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Master the "Robotic Nervous System". Learn ROS 2 nodes, topics, services,
        and how to build the foundational architecture for Physical AI systems.
      </>
    ),
  },
  {
    title: 'Module 2: The Digital Twin',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Bridge reality and simulation. Explore high-fidelity physics with Gazebo
        and Unity, simulating complex environments before deploying to hardware.
      </>
    ),
  },
  {
    title: 'Module 3: AI-Robot Brain',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Powered by NVIDIA Isaac™. Develop advanced perception, VSLAM, and
        autonomous navigation for humanoid robots using hardware-accelerated AI.
      </>
    ),
  },
  {
    title: 'Module 4: VLA Integration',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        The future of Robotics. Integrate Vision-Language-Action models (VLA)
        and LLMs like GPT-4 to enable natural human-robot interaction.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
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
