import { LucideIcon } from "lucide-react";

interface StatCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

export default function StatCard({ title, value, icon: Icon, trend }: StatCardProps) {
  return (
    <div className="glass-strong card animate-fade-in">
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <div style={{ flex: 1 }}>
          <p style={{ 
            fontSize: '0.875rem', 
            fontWeight: 600, 
            color: 'var(--foreground-muted)', 
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            marginBottom: '0.5rem'
          }}>
            {title}
          </p>
          <h2 style={{ 
            fontSize: 'clamp(2rem, 4vw, 3rem)', 
            fontWeight: 900, 
            color: 'var(--foreground)',
            lineHeight: 1,
            fontFamily: "'Outfit', sans-serif"
          }}>
            {value}
          </h2>
          {trend && (
            <div style={{ 
              marginTop: '0.75rem', 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.25rem' 
            }}>
              <span style={{ 
                fontSize: '0.875rem', 
                fontWeight: 600,
                color: trend.isPositive ? 'var(--success)' : 'var(--error)'
              }}>
                {trend.isPositive ? '↑' : '↓'} {Math.abs(trend.value)}%
              </span>
              <span style={{ 
                fontSize: '0.75rem', 
                color: 'var(--foreground-subtle)' 
              }}>
                vs last season
              </span>
            </div>
          )}
        </div>
        <div style={{
          width: '64px',
          height: '64px',
          borderRadius: 'var(--radius-lg)',
          background: 'linear-gradient(135deg, var(--primary), var(--primary-dark))',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 8px 16px rgba(255, 107, 53, 0.3)',
          flexShrink: 0
        }}>
          <Icon size={32} color="white" strokeWidth={2.5} />
        </div>
      </div>
    </div>
  );
}
