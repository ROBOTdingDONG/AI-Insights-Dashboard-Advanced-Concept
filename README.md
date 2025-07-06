# 🔮 AI Insights Dashboard

**An intelligent data visualization platform that transforms real-time external research into interactive dashboards and animated insights.**

## 🎯 Project Overview

This dashboard combines Perplexity API search capabilities with LLM-powered analysis to create dynamic, visual insights from external data sources. Users can input topics, gather real-time information, and generate professional visualizations with automated storytelling capabilities.

## 🧩 Core Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Real-time Data Collection** | Perplexity API integration for external source gathering | 🔄 In Development |
| **LLM-Powered Analysis** | OpenAI/Claude integration for data extraction and summarization | 🔄 In Development |
| **Interactive Visualizations** | Dynamic charts, graphs, and trend analysis | 📋 Planned |
| **Animated Storytelling** | Time-based visual narratives and data stories | 📋 Planned |
| **Multi-format Export** | PDF reports, shareable dashboards, animated videos | 📋 Planned |
| **Business Intelligence** | KPI overlays and comparative analysis | 📋 Planned |

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Perplexity API  │───▶│  Data Processor │
│  (Topics/Query) │    │    Integration   │    │   & LLM Layer   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             ▼
│  Export Engine  │◀───│  Visualization   │◀───┌─────────────────┐
│ (PDF/Video/Web) │    │     Engine       │    │   PostgreSQL    │
└─────────────────┘    └──────────────────┘    │   + TimescaleDB │
                                               └─────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI (Python) - High performance, automatic API documentation
- **Database**: PostgreSQL + TimescaleDB - Optimized for time-series analytics
- **LLM Integration**: OpenAI API + LangChain - Advanced NLP and RAG capabilities
- **External APIs**: Perplexity API - Real-time search and data collection

### Frontend
- **Framework**: React + TypeScript - Modern, scalable frontend architecture
- **Visualization**: Recharts - React-native charting with excellent performance
- **Styling**: Tailwind CSS - Utility-first, responsive design system
- **State Management**: Zustand - Lightweight, TypeScript-friendly state management

### Infrastructure
- **Containerization**: Docker + Docker Compose - Consistent development environment
- **Authentication**: JWT + OAuth2 - Secure user authentication
- **File Storage**: AWS S3 - Scalable asset and export storage
- **Deployment**: AWS ECS/Vercel - Production-ready hosting

## 📁 Repository Structure

```
ai-insights-dashboard/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── core/
│   │   │   ├── config.py          # Configuration management
│   │   │   ├── security.py        # Authentication & authorization
│   │   │   └── database.py        # Database connection & models
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── search.py      # Perplexity API integration
│   │   │   │   ├── analyze.py     # LLM analysis endpoints
│   │   │   │   ├── visualize.py   # Data visualization APIs
│   │   │   │   └── export.py      # Export functionality
│   │   │   └── dependencies.py    # FastAPI dependencies
│   │   ├── services/
│   │   │   ├── perplexity.py      # External API service
│   │   │   ├── summarizer.py      # LLM processing service
│   │   │   ├── visualizer.py      # Chart generation service
│   │   │   └── export_service.py  # Multi-format export service
│   │   ├── models/
│   │   │   ├── analytics.py       # Data models
│   │   │   ├── users.py          # User management models
│   │   │   └── exports.py         # Export tracking models
│   │   └── utils/
│   │       ├── data_processor.py  # Data transformation utilities
│   │       └── validators.py      # Input validation
│   ├── tests/
│   └── alembic/                   # Database migrations
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard/         # Main dashboard components
│   │   │   ├── Charts/           # Visualization components
│   │   │   ├── Search/           # Search interface
│   │   │   └── Export/           # Export controls
│   │   ├── hooks/                # Custom React hooks
│   │   ├── services/             # API service layer
│   │   ├── stores/               # State management
│   │   ├── types/                # TypeScript definitions
│   │   └── utils/                # Frontend utilities
│   ├── public/
│   └── package.json
├── prompts/
│   ├── insight_extraction.txt     # LLM prompts for data analysis
│   ├── summarization.txt         # Content summarization prompts
│   └── trend_analysis.txt        # Trend identification prompts
└── docs/
    ├── api.md                    # API documentation
    ├── deployment.md             # Deployment guide
    └── development.md            # Development setup
```

## 🔐 Security Considerations

- **API Key Management**: Environment-based configuration with rotation support
- **Input Validation**: Comprehensive sanitization and rate limiting
- **Authentication**: JWT tokens with refresh mechanism
- **Data Privacy**: Encrypted storage and secure data transmission
- **CORS Policy**: Strict origin validation for production environments

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Docker & Docker Compose

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd ai-insights-dashboard

# Copy environment variables
cp .env.example .env

# Start development environment
docker-compose up -d

# Install backend dependencies
cd backend && pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install

# Run development servers
npm run dev:backend    # FastAPI server
npm run dev:frontend   # React development server
```

## 📊 Use Cases

### 1. Trend Analysis
**Example**: "AI regulation in Europe"
- **Output**: Time series analysis, policy stance tracking, source credibility assessment
- **Visualization**: Interactive timeline with sentiment analysis overlay

### 2. Market Research
**Example**: "Top LLM providers in healthcare"
- **Output**: Funding analysis, adoption metrics, competitive landscape
- **Visualization**: Comparative bar charts, market share evolution

### 3. Narrative Reports
**Example**: Quarterly technology trend summary
- **Output**: Automated visual summaries with presentation-ready narratives
- **Visualization**: Animated data stories with voiceover script generation

## 🔄 Development Phases

- **Phase 1**: Repository setup and core architecture ✅
- **Phase 2**: Perplexity API integration and LLM processing 🔄
- **Phase 3**: Interactive dashboard and visualization engine 📋
- **Phase 4**: Animation and storytelling capabilities 📋
- **Phase 5**: Export functionality and sharing features 📋

## 📝 Contributing

Please read our [Contributing Guidelines](docs/contributing.md) and [Code of Conduct](docs/code-of-conduct.md) before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for intelligent data visualization and insights generation.**
