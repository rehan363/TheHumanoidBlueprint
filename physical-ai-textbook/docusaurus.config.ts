import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Learn Physical AI, Embodied Intelligence, and Humanoid Robotics',
  favicon: 'img/favicon.ico',

  // Enable Mermaid for diagrams
  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://rehan363.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  // Use '/' for local development, change to '/panaversity-hackathon-I/' for production
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'rehan363', // Usually your GitHub org/user name.
  projectName: 'panaversity-hackathon-I', // Usually your repo name.

  // Add KaTeX CSS for math rendering
  stylesheets: [
    'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
  ],

  onBrokenLinks: 'warn', // Changed to warn for now during development

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/', // Set homepage to docs/intro.md
          // Configure math rendering plugins
          remarkPlugins: [require('remark-math')],
          rehypePlugins: [require('rehype-katex')],
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/rehan363/panaversity-hackathon-I/tree/main/physical-ai-textbook',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/rehan363/panaversity-hackathon-I/tree/main/physical-ai-textbook',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI Textbook Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'curriculumSidebar',
          position: 'left',
          label: 'Curriculum',
        },
        {
          href: 'https://github.com/rehan363/panaversity-hackathon-I',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Resources',
          items: [
            {
              label: 'Course Overview',
              to: '/', // Homepage (docs/intro.md with routeBasePath '/')
            },
            {
              label: 'GitHub Repository',
              href: 'https://github.com/rehan363/panaversity-hackathon-I',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Report Issue',
              href: 'https://github.com/rehan363/panaversity-hackathon-I/issues',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Licensed under CC BY 4.0`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
