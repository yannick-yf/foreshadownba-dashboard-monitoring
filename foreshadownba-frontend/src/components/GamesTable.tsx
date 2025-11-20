'use client';

import { useState, useMemo } from 'react';
import { NBAGame } from '@/lib/types';
import { formatDate } from '@/lib/utils';
import { ArrowUpDown, CheckCircle2, XCircle } from 'lucide-react';

interface GamesTableProps {
  games: NBAGame[];
}

type SortField = 'game_date' | 'tm' | 'opp' | 'results' | 'prediction_value';
type SortDirection = 'asc' | 'desc';

export default function GamesTable({ games }: GamesTableProps) {
  const [sortField, setSortField] = useState<SortField>('game_date');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const filteredAndSortedGames = useMemo(() => {
    let filtered = games;

    if (searchTerm) {
      filtered = games.filter(game =>
        game.tm.toLowerCase().includes(searchTerm.toLowerCase()) ||
        game.opp.toLowerCase().includes(searchTerm.toLowerCase()) ||
        game.game_date.includes(searchTerm)
      );
    }

    return [...filtered].sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];

      if (sortField === 'game_date') {
        aVal = new Date(a.game_date).getTime();
        bVal = new Date(b.game_date).getTime();
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });
  }, [games, searchTerm, sortField, sortDirection]);

  const totalPages = Math.ceil(filteredAndSortedGames.length / itemsPerPage);
  const paginatedGames = filteredAndSortedGames.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const SortButton = ({ field, label }: { field: SortField; label: string }) => (
    <button
      onClick={() => handleSort(field)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.25rem',
        background: 'none',
        border: 'none',
        color: 'var(--foreground)',
        fontSize: '0.875rem',
        fontWeight: 700,
        cursor: 'pointer',
        padding: '0.5rem',
        borderRadius: 'var(--radius-sm)',
        transition: 'background var(--transition-fast)'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--surface-lighter)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'none';
      }}
    >
      {label}
      <ArrowUpDown size={14} style={{ opacity: sortField === field ? 1 : 0.3 }} />
    </button>
  );

  return (
    <div>
      {/* Search Bar */}
      <div style={{ marginBottom: 'var(--spacing-lg)' }}>
        <input
          type="text"
          placeholder="Search by team or date..."
          value={searchTerm}
          onChange={(e) => {
            setSearchTerm(e.target.value);
            setCurrentPage(1);
          }}
          style={{
            width: '100%',
            padding: 'var(--spacing-md)',
            background: 'var(--surface-light)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--foreground)',
            fontSize: '0.875rem',
            transition: 'all var(--transition-fast)'
          }}
          onFocus={(e) => {
            e.currentTarget.style.borderColor = 'var(--primary)';
            e.currentTarget.style.boxShadow = '0 0 0 3px rgba(255, 107, 53, 0.1)';
          }}
          onBlur={(e) => {
            e.currentTarget.style.borderColor = 'var(--border)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        />
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto', marginBottom: 'var(--spacing-lg)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--border)' }}>
              <th style={{ textAlign: 'left', padding: 'var(--spacing-md)' }}>
                <SortButton field="game_date" label="Date" />
              </th>
              <th style={{ textAlign: 'left', padding: 'var(--spacing-md)' }}>Season</th>
              <th style={{ textAlign: 'left', padding: 'var(--spacing-md)' }}>
                <SortButton field="tm" label="Team" />
              </th>
              <th style={{ textAlign: 'left', padding: 'var(--spacing-md)' }}>
                <SortButton field="opp" label="Opponent" />
              </th>
              <th style={{ textAlign: 'center', padding: 'var(--spacing-md)' }}>
                <SortButton field="results" label="Result" />
              </th>
              <th style={{ textAlign: 'center', padding: 'var(--spacing-md)' }}>
                <SortButton field="prediction_value" label="Prediction" />
              </th>
              <th style={{ textAlign: 'center', padding: 'var(--spacing-md)' }}>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {paginatedGames.map((game, index) => {
              const isCorrect = game.results === game.prediction_value;
              return (
                <tr
                  key={game.id}
                  style={{
                    borderBottom: '1px solid var(--border)',
                    background: index % 2 === 0 ? 'transparent' : 'var(--surface-light)',
                    transition: 'background var(--transition-fast)'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'var(--surface-lighter)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = index % 2 === 0 ? 'transparent' : 'var(--surface-light)';
                  }}
                >
                  <td style={{ padding: 'var(--spacing-md)', fontSize: '0.875rem', color: 'var(--foreground-muted)' }}>
                    {formatDate(game.game_date)}
                  </td>
                  <td style={{ padding: 'var(--spacing-md)', fontSize: '0.875rem', color: 'var(--foreground-muted)' }}>
                    {game.id_season}
                  </td>
                  <td style={{ padding: 'var(--spacing-md)', fontSize: '0.875rem', fontWeight: 600, color: 'var(--foreground)' }}>
                    {game.tm}
                  </td>
                  <td style={{ padding: 'var(--spacing-md)', fontSize: '0.875rem', fontWeight: 600, color: 'var(--foreground)' }}>
                    {game.opp}
                  </td>
                  <td style={{ padding: 'var(--spacing-md)', textAlign: 'center' }}>
                    <span className={game.results === 1 ? 'badge-success' : 'badge-error'} style={{
                      display: 'inline-block',
                      padding: '0.25rem 0.75rem',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      borderRadius: '9999px'
                    }}>
                      {game.results === 1 ? 'Win' : 'Loss'}
                    </span>
                  </td>
                  <td style={{ padding: 'var(--spacing-md)', textAlign: 'center' }}>
                    <span className={game.prediction_value === 1 ? 'badge-success' : 'badge-error'} style={{
                      display: 'inline-block',
                      padding: '0.25rem 0.75rem',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      borderRadius: '9999px'
                    }}>
                      {game.prediction_value === 1 ? 'Win' : 'Loss'}
                    </span>
                  </td>
                  <td style={{ padding: 'var(--spacing-md)', textAlign: 'center' }}>
                    {isCorrect ? (
                      <CheckCircle2 size={20} color="var(--success)" />
                    ) : (
                      <XCircle size={20} color="var(--error)" />
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 'var(--spacing-md)'
      }}>
        <p style={{ fontSize: '0.875rem', color: 'var(--foreground-muted)' }}>
          Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, filteredAndSortedGames.length)} of {filteredAndSortedGames.length} games
        </p>
        <div style={{ display: 'flex', gap: 'var(--spacing-sm)' }}>
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="btn btn-secondary"
            style={{ opacity: currentPage === 1 ? 0.5 : 1, cursor: currentPage === 1 ? 'not-allowed' : 'pointer' }}
          >
            Previous
          </button>
          <span style={{ 
            padding: 'var(--spacing-sm) var(--spacing-lg)', 
            display: 'flex', 
            alignItems: 'center',
            fontSize: '0.875rem',
            fontWeight: 600,
            color: 'var(--foreground)'
          }}>
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="btn btn-secondary"
            style={{ opacity: currentPage === totalPages ? 0.5 : 1, cursor: currentPage === totalPages ? 'not-allowed' : 'pointer' }}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
