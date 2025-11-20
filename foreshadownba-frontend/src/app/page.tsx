'use client';

import { useEffect, useState, useMemo } from 'react';
import { Target, Trophy, BarChart3 } from 'lucide-react';
import StatCard from '@/components/StatCard';
import TeamFilter from '@/components/TeamFilter';
import ConfusionMatrix from '@/components/ConfusionMatrix';
import GamesTable from '@/components/GamesTable';
import { NBAGame } from '@/lib/types';
import { 
  prepareNBAGamesData, 
  calculateSeasonAccuracy, 
  filterGamesByTeam,
  calculateConfusionMatrix 
} from '@/lib/utils';

export default function Home() {
  const [games, setGames] = useState<NBAGame[]>([]);
  const [selectedTeams, setSelectedTeams] = useState<string[]>(['All']);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        // For now, we'll use the local CSV file as a fallback
        // In production, this would fetch from S3
        const response = await fetch('/api/games');
        if (!response.ok) {
          throw new Error('Failed to fetch games data');
        }
        const data = await response.json();
        const preparedData = prepareNBAGamesData(data);
        setGames(preparedData);
      } catch (err) {
        console.error('Error loading data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  const filteredGames = useMemo(() => 
    filterGamesByTeam(games, selectedTeams), 
    [games, selectedTeams]
  );

  const seasonAccuracy = useMemo(() => {
    const accuracy = calculateSeasonAccuracy(filteredGames);
    return accuracy[0] || { row_count: 0, sum_row_accuracy: 0, season_accuracy: 0 };
  }, [filteredGames]);

  const confusionMatrix = useMemo(() => 
    calculateConfusionMatrix(filteredGames), 
    [filteredGames]
  );

  if (loading) {
    return (
      <div style={{ 
        minHeight: '100vh', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: 'var(--background)'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div className="animate-pulse" style={{
            width: '64px',
            height: '64px',
            margin: '0 auto var(--spacing-lg)',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, var(--primary), var(--primary-dark))',
          }} />
          <p style={{ color: 'var(--foreground-muted)', fontSize: '1.125rem' }}>
            Loading NBA predictions...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ 
        minHeight: '100vh', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: 'var(--background)',
        padding: 'var(--spacing-lg)'
      }}>
        <div className="card" style={{ maxWidth: '500px', textAlign: 'center' }}>
          <h2 style={{ color: 'var(--error)', marginBottom: 'var(--spacing-md)' }}>
            Error Loading Data
          </h2>
          <p style={{ color: 'var(--foreground-muted)' }}>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--background)', paddingBottom: 'var(--spacing-2xl)' }}>
      {/* Header */}
      <header style={{ 
        background: 'linear-gradient(135deg, var(--surface), var(--surface-light))',
        borderBottom: '1px solid var(--border)',
        padding: 'var(--spacing-2xl) 0'
      }}>
        <div className="container">
          <h1 className="animate-fade-in" style={{ 
            marginBottom: 'var(--spacing-md)',
            background: 'linear-gradient(135deg, var(--primary), var(--accent))',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text'
          }}>
            ForeshadowNBA Dashboard
          </h1>
          <p className="animate-fade-in" style={{ 
            fontSize: '1.125rem', 
            color: 'var(--foreground-muted)',
            maxWidth: '600px',
            animationDelay: '100ms'
          }}>
            Real-time NBA game prediction analytics powered by machine learning
          </p>
        </div>
      </header>

      <div className="container" style={{ marginTop: 'var(--spacing-2xl)' }}>
        {/* Team Filter */}
        <div style={{ marginBottom: 'var(--spacing-2xl)' }}>
          <h3 style={{ marginBottom: 'var(--spacing-md)' }}>Filter by Team</h3>
          <TeamFilter 
            selectedTeams={selectedTeams} 
            onTeamsChange={setSelectedTeams} 
          />
        </div>

        {/* Stats Cards */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: 'var(--spacing-lg)',
          marginBottom: 'var(--spacing-2xl)'
        }}>
          <StatCard
            title="Games Correctly Predicted"
            value={seasonAccuracy.sum_row_accuracy}
            icon={Trophy}
          />
          <StatCard
            title="Total Number of Games"
            value={seasonAccuracy.row_count}
            icon={BarChart3}
          />
          <StatCard
            title="Prediction Accuracy"
            value={`${(seasonAccuracy.season_accuracy * 100).toFixed(1)}%`}
            icon={Target}
          />
        </div>

        {/* Confusion Matrix */}
        <div className="card" style={{ marginBottom: 'var(--spacing-2xl)' }}>
          <h3 className="card-header">
            Confusion Matrix
          </h3>
          <ConfusionMatrix data={confusionMatrix} />
        </div>

        {/* Games Table */}
        <div className="card">
          <h3 className="card-header">
            NBA Games Data
          </h3>
          <GamesTable games={filteredGames} />
        </div>
      </div>
    </div>
  );
}
