import { useEffect, useRef } from 'react';

interface ChartProps {
  data: any;
  type: 'line' | 'bar' | 'pie' | 'doughnut';
  options?: any;
  height?: number;
  className?: string;
}

export default function Chart({ 
  data, 
  type, 
  options = {}, 
  height = 300,
  className = '' 
}: ChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<any>(null);

  useEffect(() => {
    if (!canvasRef.current || !data) return;

    // Dynamic import of Chart.js to avoid SSR issues
    import('chart.js/auto').then((Chart) => {
      const ctx = canvasRef.current?.getContext('2d');
      if (!ctx) return;

      // Destroy existing chart
      if (chartRef.current) {
        chartRef.current.destroy();
      }

      // Default options
      const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top' as const,
          },
          title: {
            display: false,
          },
        },
        scales: type === 'pie' || type === 'doughnut' ? undefined : {
          y: {
            beginAtZero: true,
          },
        },
      };

      // Merge options
      const mergedOptions = {
        ...defaultOptions,
        ...options,
      };

      // Create new chart
      chartRef.current = new Chart.default(ctx, {
        type,
        data,
        options: mergedOptions,
      });
    });

    // Cleanup
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [data, type, options]);

  return (
    <div className={`relative ${className}`} style={{ height }}>
      <canvas ref={canvasRef} />
    </div>
  );
}