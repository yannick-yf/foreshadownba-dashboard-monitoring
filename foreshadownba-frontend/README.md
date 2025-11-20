# ForeshadowNBA Dashboard

A professional NBA game prediction dashboard built with Next.js, featuring real-time analytics, machine learning predictions, and interactive visualizations.

## Features

- **Real-time Predictions**: View NBA game predictions powered by machine learning
- **Interactive Filtering**: Filter games by team or conference
- **Confusion Matrix**: Visual representation of prediction accuracy
- **Data Table**: Sortable, searchable table with pagination
- **Modern UI**: Premium design with dark mode, glassmorphism effects, and smooth animations
- **Responsive**: Fully responsive design for desktop, tablet, and mobile

## Tech Stack

- **Framework**: Next.js 16 (App Router)
- **Language**: TypeScript
- **Styling**: CSS with custom design system
- **Icons**: Lucide React
- **Data Source**: AWS S3
- **CSV Parsing**: PapaParse

## Getting Started

### Prerequisites

- Node.js 18+ installed
- AWS credentials with S3 access
- Access to the `foreshadownba` S3 bucket

### Installation

1. Clone the repository and navigate to the frontend directory:
```bash
cd foreshadownba-dashboard-monitoring/foreshadownba-frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env.local` file in the root directory:
```bash
cp .env.local.example .env.local
```

4. Update the `.env.local` file with your AWS credentials:
```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
S3_BUCKET_NAME=foreshadownba
S3_FILE_KEY=inference-pipeline-output/nba_games_inseasonn_w_pred.csv
```

### Running the Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to see the dashboard.

### Building for Production

```bash
npm run build
npm start
```

## Project Structure

```
foreshadownba-frontend/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   └── games/
│   │   │       └── route.ts       # API route for fetching data
│   │   ├── globals.css            # Global styles and design system
│   │   ├── layout.tsx             # Root layout
│   │   └── page.tsx               # Main dashboard page
│   ├── components/
│   │   ├── StatCard.tsx           # Metric display cards
│   │   ├── TeamFilter.tsx         # Team selection dropdown
│   │   ├── ConfusionMatrix.tsx    # Confusion matrix visualization
│   │   └── GamesTable.tsx         # Interactive data table
│   └── lib/
│       ├── types.ts               # TypeScript type definitions
│       ├── utils.ts               # Utility functions
│       ├── teams.ts               # NBA team data
│       └── s3.ts                  # AWS S3 integration
├── public/                        # Static assets
├── .env.local.example             # Environment variables template
└── package.json
```

## Design System

The dashboard features a comprehensive design system with:

- **Color Palette**: NBA-inspired colors with primary orange (#FF6B35), secondary blue (#1E3A8A), and accent gold (#F59E0B)
- **Typography**: Inter for body text, Outfit for headings
- **Animations**: Smooth transitions, fade-ins, and hover effects
- **Components**: Reusable card, button, and badge styles
- **Dark Mode**: Default dark theme optimized for readability

## Key Components

### StatCard
Displays key metrics with icons and optional trend indicators.

### TeamFilter
Multi-select dropdown for filtering games by NBA teams, organized by conference.

### ConfusionMatrix
2x2 grid showing prediction accuracy breakdown (TP, FP, TN, FN).

### GamesTable
Interactive table with sorting, searching, and pagination for viewing game predictions.

## Data Flow

1. **Data Fetching**: API route (`/api/games`) fetches CSV data from AWS S3
2. **Data Processing**: Utility functions prepare and calculate metrics
3. **Filtering**: Client-side filtering by selected teams
4. **Visualization**: Components render processed data with interactive features

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AWS_REGION` | AWS region for S3 bucket | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key | Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Yes |
| `S3_BUCKET_NAME` | S3 bucket name | Yes |
| `S3_FILE_KEY` | Path to CSV file in bucket | Yes |

## Performance

- **Build Time**: ~6 seconds
- **Bundle Size**: Optimized with Next.js automatic code splitting
- **Caching**: API responses cached for 5 minutes
- **Responsive**: Mobile-first design with optimized layouts

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

This project is part of the ForeshadowNBA prediction system.
