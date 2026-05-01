import { useEffect, useRef } from 'react';

interface Ember {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  hue: number;
  life: number;
  swirl: number;
  swirlSpeed: number;
}

export function EmberOverlay() {
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

    const embers: Ember[] = [];
    const maxEmbers = 25; // Much fewer embers

    const createEmber = () => {
      return {
        x: Math.random() * canvas.width,
        y: canvas.height + 10,
        vx: (Math.random() - 0.5) * 4, // Faster horizontal movement
        vy: -(Math.random() * 3 + 1.5), // Faster upward movement
        size: Math.random() * 2.5 + 0.5,
        opacity: Math.random() * 0.8 + 0.2,
        hue: Math.random() * 40 + 5, // Orange to red
        life: 1,
        swirl: Math.random() * Math.PI * 2,
        swirlSpeed: (Math.random() - 0.5) * 0.15, // More swirl
      };
    };

    // Initialize embers
    for (let i = 0; i < maxEmbers; i++) {
      const ember = createEmber();
      ember.y = Math.random() * canvas.height;
      embers.push(ember);
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Spawn new embers from bottom - less often
      if (embers.length < maxEmbers && Math.random() < 0.08) {
        embers.push(createEmber());
      }

      // Update and draw embers
      for (let i = embers.length - 1; i >= 0; i--) {
        const ember = embers[i];
        
        // Update position with swirl effect
        ember.swirl += ember.swirlSpeed;
        ember.x += ember.vx + Math.sin(ember.swirl) * 1.2; // More pronounced swirl
        ember.y += ember.vy;
        
        // Add turbulence
        ember.vx += (Math.random() - 0.5) * 0.2;
        ember.vy += (Math.random() - 0.5) * 0.1;
        
        // Fade out as it rises
        ember.life -= 0.005;
        ember.opacity = ember.life * (Math.random() * 0.3 + 0.7);

        // Remove dead embers or those out of bounds
        if (ember.life <= 0 || ember.y < -20 || ember.x < -20 || ember.x > canvas.width + 20) {
          embers.splice(i, 1);
          continue;
        }

        // Draw glow
        const gradient = ctx.createRadialGradient(
          ember.x,
          ember.y,
          0,
          ember.x,
          ember.y,
          ember.size * 4
        );
        gradient.addColorStop(0, `hsla(${ember.hue}, 100%, 60%, ${ember.opacity})`);
        gradient.addColorStop(0.4, `hsla(${ember.hue}, 100%, 50%, ${ember.opacity * 0.5})`);
        gradient.addColorStop(1, `hsla(${ember.hue - 10}, 100%, 40%, 0)`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(ember.x, ember.y, ember.size * 4, 0, Math.PI * 2);
        ctx.fill();

        // Draw bright core
        ctx.fillStyle = `hsla(${ember.hue + 30}, 100%, 80%, ${ember.opacity})`;
        ctx.beginPath();
        ctx.arc(ember.x, ember.y, ember.size, 0, Math.PI * 2);
        ctx.fill();

        // Occasional spark trail
        if (Math.random() < 0.1) {
          ctx.fillStyle = `hsla(${ember.hue + 20}, 100%, 70%, ${ember.opacity * 0.3})`;
          ctx.beginPath();
          ctx.arc(ember.x - ember.vx * 3, ember.y - ember.vy * 3, ember.size * 0.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }

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
      className="absolute inset-0 pointer-events-none z-20"
      style={{ mixBlendMode: 'screen' }}
    />
  );
}