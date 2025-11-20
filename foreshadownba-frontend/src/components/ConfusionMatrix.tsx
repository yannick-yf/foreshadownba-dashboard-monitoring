'use client';

import { ConfusionMatrixData } from '@/lib/types';

interface ConfusionMatrixProps {
  data: ConfusionMatrixData;
}

export default function ConfusionMatrix({ data }: ConfusionMatrixProps) {
  const total = data.truePositive + data.falsePositive + data.trueNegative + data.falseNegative;
  const accuracy = total > 0 ? ((data.truePositive + data.trueNegative) / total * 100).toFixed(1) : 0;

  const cells = [
    {
      label: 'True Positive',
      value: data.truePositive,
      description: 'Correctly predicted wins',
      color: 'var(--success)',
      bgColor: 'rgba(16, 185, 129, 0.15)'
    },
    {
      label: 'False Positive',
      value: data.falsePositive,
      description: 'Predicted win, actual loss',
      color: 'var(--error)',
      bgColor: 'rgba(239, 68, 68, 0.15)'
    },
    {
      label: 'False Negative',
      value: data.falseNegative,
      description: 'Predicted loss, actual win',
      color: 'var(--warning)',
      bgColor: 'rgba(245, 158, 11, 0.15)'
    },
    {
      label: 'True Negative',
      value: data.trueNegative,
      description: 'Correctly predicted losses',
      color: 'var(--info)',
      bgColor: 'rgba(59, 130, 246, 0.15)'
    }
  ];

  return (
    <div>
      <div style={{ 
        marginBottom: 'var(--spacing-xl)', 
        textAlign: 'center',
        padding: 'var(--spacing-lg)',
        background: 'var(--surface-light)',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border)'
      }}>
        <p style={{ 
          fontSize: '0.875rem', 
          color: 'var(--foreground-muted)', 
          marginBottom: '0.5rem',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          fontWeight: 600
        }}>
          Overall Accuracy
        </p>
        <h3 style={{ 
          fontSize: '3rem', 
          fontWeight: 900, 
          color: 'var(--primary)',
          fontFamily: "'Outfit', sans-serif"
        }}>
          {accuracy}%
        </h3>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: 'var(--spacing-lg)'
      }}>
        {cells.map((cell, index) => (
          <div
            key={cell.label}
            className="animate-fade-in"
            style={{
              background: cell.bgColor,
              border: `2px solid ${cell.color}`,
              borderRadius: 'var(--radius-lg)',
              padding: 'var(--spacing-xl)',
              textAlign: 'center',
              transition: 'all var(--transition-base)',
              animationDelay: `${index * 100}ms`
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-4px)';
              e.currentTarget.style.boxShadow = `0 8px 20px ${cell.color}40`;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <p style={{
              fontSize: '0.75rem',
              fontWeight: 700,
              color: cell.color,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              marginBottom: 'var(--spacing-sm)'
            }}>
              {cell.label}
            </p>
            <h4 style={{
              fontSize: '2.5rem',
              fontWeight: 900,
              color: cell.color,
              fontFamily: "'Outfit', sans-serif",
              marginBottom: 'var(--spacing-sm)'
            }}>
              {cell.value}
            </h4>
            <p style={{
              fontSize: '0.75rem',
              color: 'var(--foreground-muted)'
            }}>
              {cell.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
