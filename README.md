# 🤖 The Humanoid Blueprint: Physical AI & Humanoid Robotics

Welcome to **The Humanoid Blueprint**, a professional, interactive technical textbook and educational platform designed for the next generation of roboticists. This project was developed as part of the **Panaversity Hackathon I**, focusing on the convergence of Generative AI and Physical Robotics.

[![Live Site](https://img.shields.io/badge/Live-Website-8A2BE2?style=for-the-badge&logo=github)](https://rehan363.github.io/TheHumanoidBlueprint/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/rehan363/TheHumanoidBlueprint)

---

## 🌟 Executive Summary

The future of work is a partnership between humans and intelligent physical agents. **The Humanoid Blueprint** bridges the gap between the digital brain and the physical body. It provides a comprehensive 13-chapter curriculum covering everything from ROS 2 fundamentals to Vision-Language-Action (VLA) models.

This isn't just a book—it's an **AI-Powered Learning Ecosystem** featuring:
- **Interactive 3D Visualizations**: Real-time 3D rendered book covers and robotic models.
- **Embedded RAG Chatbot**: A specialized AI tutor that understands the entire textbook and provides context-aware assistance.
- **Modern Auth System**: Personalization through [Better-Auth](https://www.better-auth.com/) and student onboarding.
- **Premium UI/UX**: Professional light/dark mode design with a focus on technical readability.

---

## 🏗️ Architecture Overview

The project is built on a modern, distributed architecture:

### 📱 Frontend: [Docusaurus 3.9](https://docusaurus.io/)
- **13 Specialized Chapters**: Comprehensive technical content on robotics and Physical AI.
- **React-Powered Components**: Interactive elements including 3D models (Three.js/Fiber).
- **Embedded RAG Widget**: A persistent chat interface for live content queries.
- **Theme Engine**: Professional light/dark mode support using Deep Plum and Gold accents.

### ⚙️ Backend: [FastAPI](https://fastapi.tiangolo.com/) & [Python 3.10+](https://www.python.org/)
- **Multi-Agent RAG Pipeline**: Orchestrates specialized agents for Retrieval, Explanation, and Summarization.
- **Vector Intelligence**: Powered by **Qdrant Cloud** for semantic indexing of the entire textbook.
- **Database Layer**: **Neon Serverless Postgres** for chat history and user session persistence.
- **Auth Service**: Node.js/TypeScript sidecar running **Better-Auth** for high-security student accounts.

### 🧠 AI Core
- **LLM Orchestration**: OpenRouter (DeepSeek, Mistral) with Gemini API fallbacks.
- **Embeddings**: Google Gemini `text-embedding-004` (768 dimensions).
- **SDKs**: OpenAI Agents SDK for modular intelligence.

---

## 📁 Project Structure

```text
TheHumanoidBlueprint/
├── physical-ai-textbook/     # Docusaurus Frontend
│   ├── docs/                 # 13 Chapters of technical content
│   ├── src/                  # React components & UI logic
│   └── docusaurus.config.ts  # Site & Deployment configuration
├── backend/                  # FastAPI & RAG Infrastructure
│   ├── rag_backend/          # RAG pipeline logic & AI Agents
│   ├── auth/                 # Better-Auth service (Node.js/Drizzle)
│   └── scripts/              # Data indexing and setup tools
├── specs/                    # Spec-Driven Development documentation
└── .github/workflows/        # Automated CI/CD (GitHub Actions)
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Node.js 18+** & **npm/yarn**
- **Python 3.10+** (with `uv` recommended)
- **NVIDIA GPU** (Optional, for running local simulations mentioned in the book)

### 2. Frontend Setup
```bash
cd physical-ai-textbook
npm install
npm start
```

### 3. Backend Setup
```bash
cd backend
pip install -e .
# Configure your .env with OpenRouter, Qdrant, and Neon keys
uvicorn rag_backend.main:app --reload
```

---

## 📚 Curriculum Highlights

1.  **Introduction to Physical AI**: Foundations of embodied intelligence.
2.  **ROS 2 Fundamentals**: Nodes, topics, services, and actions.
3.  **Robot Simulation**: Master Gazebo and URDF modeling.
4.  **The AI-Robot Brain**: Professional development with NVIDIA Isaac Sim.
5.  **Vision-Language-Action (VLA)**: Convergence of LLMs and Robotics.
6.  **Conversational Robotics**: Multi-modal interaction (Speech, Gesture, Vision).

---

## 👑 Hackathon Features (Spec-Driven)

This project follows strict **Spec-Driven Development (SDD)**:
- [x] **AI-Driven Book**: Fully generated and curated using Spec-Kit Plus.
- [x] **Integrated RAG**: Semantic search across all 13 chapters.
- [x] **Auth Integration**: Secure student login and personalized profiles.
- [x] **Text Selection Queries**: Highlight text to ask the AI specifically about a snippet.
- [x] **Automated CI/CD**: Seamless deployment to GitHub Pages via professional Actions.

---

## 👥 Authors & Acknowledgments

- **Rehan Ahmed** - *Lead AI Developer & Roboticist*
- Developed for **Panaversity Hackathon I**.
- Content inspired by the latest research in VLA models and humanoid autonomy.

---

## ⚖️ License

Distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**. See `LICENSE` for more information.

---

<p align="center">
  <i>"Mastering the Era of Physical AI"</i>
</p>
