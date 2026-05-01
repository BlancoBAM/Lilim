import { useEffect, useRef } from 'react';

interface Flame {
  x: number;
  y: number;
  height: number;
  width: number;
  sway: number;
  swaySpeed: number;
  flickerSpeed: number;
  opacity: number;
}

export function FlameBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Create flame columns
    const flames: Flame[] = [];
    const flameCount = 15;
    
    for (let i = 0; i < flameCount; i++) {
      flames.push({
        x: (canvas.width / flameCount) * i,
        y: canvas.height,
        height: Math.random() * 250 + 300, // Much taller flames
        width: canvas.width / flameCount + 20,
        sway: Math.random() * Math.PI * 2,
        swaySpeed: Math.random() * 0.02 + 0.01,
        flickerSpeed: Math.random() * 0.05 + 0.03,
        opacity: Math.random() * 0.5 + 0.4, // More visible
      });
    }

    let time = 0;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      time += 0.016;

      flames.forEach((flame) => {
        flame.sway += flame.swaySpeed;
        
        const swayOffset = Math.sin(flame.sway) * 15;
        const flicker = Math.sin(time * flame.flickerSpeed * 50) * 0.3 + 0.7;
        
        // Draw flame gradient from bottom
        const gradient = ctx.createLinearGradient(
          flame.x + swayOffset,
          flame.y,
          flame.x + swayOffset,
          flame.y - flame.height * flicker
        );
        
        gradient.addColorStop(0, `rgba(255, 200, 50, ${flame.opacity * 0.8})`); // Brighter yellow-orange base
        gradient.addColorStop(0.2, `rgba(255, 140, 0, ${flame.opacity * 0.7})`);
        gradient.addColorStop(0.5, `rgba(255, 69, 0, ${flame.opacity * 0.5})`);
        gradient.addColorStop(0.7, `rgba(220, 38, 38, ${flame.opacity * 0.3})`);
        gradient.addColorStop(1, 'rgba(139, 0, 0, 0)');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        
        // Draw wavy flame shape
        const segments = 20;
        for (let i = 0; i <= segments; i++) {
          const t = i / segments;
          const y = flame.y - (flame.height * flicker * t);
          const waveOffset = Math.sin(t * Math.PI * 3 + time * 2) * 10 * (1 - t);
          const x = flame.x + swayOffset + waveOffset;
          const width = flame.width * (1 - t * 0.8);
          
          if (i === 0) {
            ctx.moveTo(x - width / 2, y);
          } else {
            ctx.lineTo(x - width / 2, y);
          }
        }
        
        for (let i = segments; i >= 0; i--) {
          const t = i / segments;
          const y = flame.y - (flame.height * flicker * t);
          const waveOffset = Math.sin(t * Math.PI * 3 + time * 2) * 10 * (1 - t);
          const x = flame.x + swayOffset + waveOffset;
          const width = flame.width * (1 - t * 0.8);
          
          ctx.lineTo(x + width / 2, y);
        }
        
        ctx.closePath();
        ctx.fill();
      });

      requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none z-0"
    />
  );
}