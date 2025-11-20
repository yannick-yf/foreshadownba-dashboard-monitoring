'use client';

import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { NBA_TEAMS, EASTERN_CONFERENCE, WESTERN_CONFERENCE } from '@/lib/teams';

interface TeamFilterProps {
  selectedTeams: string[];
  onTeamsChange: (teams: string[]) => void;
}

export default function TeamFilter({ selectedTeams, onTeamsChange }: TeamFilterProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleTeamToggle = (teamValue: string) => {
    if (teamValue === 'All') {
      onTeamsChange(['All']);
      return;
    }

    const newSelection = selectedTeams.includes('All') 
      ? [teamValue]
      : selectedTeams.includes(teamValue)
        ? selectedTeams.filter(t => t !== teamValue)
        : [...selectedTeams, teamValue];

    onTeamsChange(newSelection.length === 0 ? ['All'] : newSelection);
  };

  const displayText = selectedTeams.includes('All') || selectedTeams.length === 0
    ? 'All Teams'
    : selectedTeams.length === 1
      ? NBA_TEAMS.find(t => t.value === selectedTeams[0])?.label || 'Select Teams'
      : `${selectedTeams.length} Teams Selected`;

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          width: '100%',
          padding: 'var(--spacing-md)',
          background: 'var(--surface-light)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--foreground)',
          fontSize: '0.875rem',
          fontWeight: 600,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          transition: 'all var(--transition-fast)',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = 'var(--border-light)';
          e.currentTarget.style.background = 'var(--surface-lighter)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = 'var(--border)';
          e.currentTarget.style.background = 'var(--surface-light)';
        }}
      >
        <span>{displayText}</span>
        <ChevronDown 
          size={16} 
          style={{ 
            transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform var(--transition-fast)'
          }} 
        />
      </button>

      {isOpen && (
        <div style={{
          position: 'absolute',
          top: 'calc(100% + 0.5rem)',
          left: 0,
          right: 0,
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          boxShadow: 'var(--shadow-xl)',
          maxHeight: '400px',
          overflowY: 'auto',
          zIndex: 50,
          animation: 'fadeIn 0.2s ease-out'
        }}>
          {/* All Teams Option */}
          <div
            onClick={() => {
              handleTeamToggle('All');
              setIsOpen(false);
            }}
            style={{
              padding: 'var(--spacing-md)',
              cursor: 'pointer',
              background: selectedTeams.includes('All') ? 'var(--surface-light)' : 'transparent',
              borderBottom: '1px solid var(--border)',
              transition: 'background var(--transition-fast)'
            }}
            onMouseEnter={(e) => {
              if (!selectedTeams.includes('All')) {
                e.currentTarget.style.background = 'var(--surface-light)';
              }
            }}
            onMouseLeave={(e) => {
              if (!selectedTeams.includes('All')) {
                e.currentTarget.style.background = 'transparent';
              }
            }}
          >
            <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--foreground)' }}>
              All Teams
            </span>
          </div>

          {/* Eastern Conference */}
          <div style={{ padding: 'var(--spacing-sm) var(--spacing-md)', background: 'var(--surface-lighter)' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--foreground-subtle)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Eastern Conference
            </span>
          </div>
          {EASTERN_CONFERENCE.map((team) => (
            <div
              key={team.value}
              onClick={() => handleTeamToggle(team.value)}
              style={{
                padding: 'var(--spacing-md)',
                cursor: 'pointer',
                background: selectedTeams.includes(team.value) ? 'var(--surface-light)' : 'transparent',
                transition: 'background var(--transition-fast)'
              }}
              onMouseEnter={(e) => {
                if (!selectedTeams.includes(team.value)) {
                  e.currentTarget.style.background = 'var(--surface-light)';
                }
              }}
              onMouseLeave={(e) => {
                if (!selectedTeams.includes(team.value)) {
                  e.currentTarget.style.background = 'transparent';
                }
              }}
            >
              <span style={{ fontSize: '0.875rem', color: 'var(--foreground)' }}>
                {team.label}
              </span>
            </div>
          ))}

          {/* Western Conference */}
          <div style={{ padding: 'var(--spacing-sm) var(--spacing-md)', background: 'var(--surface-lighter)', marginTop: 'var(--spacing-sm)' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--foreground-subtle)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Western Conference
            </span>
          </div>
          {WESTERN_CONFERENCE.map((team) => (
            <div
              key={team.value}
              onClick={() => handleTeamToggle(team.value)}
              style={{
                padding: 'var(--spacing-md)',
                cursor: 'pointer',
                background: selectedTeams.includes(team.value) ? 'var(--surface-light)' : 'transparent',
                transition: 'background var(--transition-fast)'
              }}
              onMouseEnter={(e) => {
                if (!selectedTeams.includes(team.value)) {
                  e.currentTarget.style.background = 'var(--surface-light)';
                }
              }}
              onMouseLeave={(e) => {
                if (!selectedTeams.includes(team.value)) {
                  e.currentTarget.style.background = 'transparent';
                }
              }}
            >
              <span style={{ fontSize: '0.875rem', color: 'var(--foreground)' }}>
                {team.label}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
