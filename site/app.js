(() => {
  'use strict';
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const $ = (id) => document.getElementById(id);
  const percentage = (value) => `${(100 * value).toFixed(2)}%`;
  const signed = (value) => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`;
  const state = { split: 'heldout', metric: 'strict' };

  // Values are public saved measurements. The HTML keeps the complete default
  // comparison readable when JavaScript or the data request is unavailable.
  function renderComparison(data) {
    const heldout = state.split === 'heldout';
    const source = data.v1;
    const metric = heldout ? source.heldout.metrics[`prompt_${state.metric}`] : null;
    const values = {};
    ['baseline', 'candidate'].forEach((arm) => {
      const numerator = heldout ? metric[arm].numerator : source.development[arm][state.metric];
      const denominator = heldout ? metric[arm].denominator : source.development[arm].total;
      const rate = numerator / denominator;
      values[arm] = rate;
      $(`${arm}-score`).textContent = numerator;
      $(`${arm}-denominator`).textContent = `/${denominator}`;
      $(`${arm}-rate`).textContent = `${percentage(rate)} of prompts`;
      $(`${arm}-bar-label`).textContent = percentage(rate);
      const bar = $(`${arm}-bar`);
      // The CSS transition is deliberately interruptible and never delays data.
      bar.style.transition = reducedMotion.matches ? 'none' : 'transform 320ms cubic-bezier(.22,1,.36,1)';
      bar.style.transform = `scaleX(${rate})`;
    });
    const splitLabel = heldout ? 'held-out IFEval' : 'development Ai2 RLVR-IFeval';
    const metricLabel = state.metric === 'strict' ? 'Strict' : 'Loose';
    $('comparison-context').textContent = `V1 ${splitLabel} · ${metricLabel} prompt compliance · Higher is better`;
    $('quality-bars').setAttribute('aria-label', `${metricLabel} ${heldout ? 'held-out' : 'development'} compliance: original ${percentage(values.baseline)}; edited ${percentage(values.candidate)}.`);
    const delta = 100 * (values.candidate - values.baseline);
    $('comparison-note').textContent = heldout
      ? `${signed(delta)} percentage points. 95% interval: ${signed(metric.ci95_percentage_points[0])} to ${signed(metric.ci95_percentage_points[1])}. Improvement and equivalence are unestablished.`
      : `${signed(delta)} percentage points on the development set used for candidate selection. This is not held-out confirmation.`;
    document.querySelectorAll('[data-split]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.split === state.split)));
    document.querySelectorAll('[data-metric]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.metric === state.metric)));
  }

  fetch('results.json')
    .then((response) => { if (!response.ok) throw new Error('Public results unavailable'); return response.json(); })
    .then((data) => {
      if (data.schema_version !== 1 || !data.v1?.heldout?.metrics?.prompt_strict) throw new Error('Unsupported result data');
      $('comparison-controls').hidden = false;
      document.querySelectorAll('[data-split]').forEach((button) => button.addEventListener('click', () => { state.split = button.dataset.split; renderComparison(data); }));
      document.querySelectorAll('[data-metric]').forEach((button) => button.addEventListener('click', () => { state.metric = button.dataset.metric; renderComparison(data); }));
      if (data.v2?.description) $('v2-description').textContent = data.v2.description;
      if (data.v2?.resource_status_label) $('v2-resource-status').textContent = data.v2.resource_status_label;
      if (data.v2?.limitation) $('v2-limitation').textContent = data.v2.limitation;
      renderComparison(data);
    })
    .catch(() => { /* Static evidence remains complete and controls stay hidden. */ });

  if (!reducedMotion.matches && typeof Element.prototype.animate === 'function') {
    document.querySelectorAll('.headline-line>span').forEach((line, index) => {
      line.animate([{ transform: 'translateY(36px)', opacity: .35 }, { transform: 'translateY(0)', opacity: 1 }], {
        duration: 720, delay: index * 90, easing: 'cubic-bezier(.16,1,.3,1)', fill: 'backwards'
      });
    });
    document.querySelectorAll('.hero-intro,.hero .button,.hero-context p').forEach((part, index) => {
      part.animate([{ transform: 'translateY(16px)', opacity: .4 }, { transform: 'translateY(0)', opacity: 1 }], {
        duration: 500, delay: 100 + index * 60, easing: 'cubic-bezier(.22,1,.36,1)', fill: 'backwards'
      });
    });
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.filter((entry) => entry.isIntersecting).forEach((entry, index) => {
          observer.unobserve(entry.target);
          if (reducedMotion.matches) return;
          const isFigure = entry.target.tagName === 'FIGURE';
          const isExchange = entry.target.classList.contains('exchange');
          const from = isFigure ? 'scale(.985)' : isExchange ? 'translateX(14px)' : 'translateY(20px)';
          entry.target.animate([{ transform: from, opacity: .45 }, { transform: 'none', opacity: 1 }], {
            duration: isFigure ? 650 : 540, delay: Math.min(index, 3) * 65,
            easing: 'cubic-bezier(.16,1,.3,1)', fill: 'backwards'
          });
        });
      }, { threshold: .08, rootMargin: '0px 0px -3% 0px' });
      document.querySelectorAll('.reveal').forEach((part) => observer.observe(part));
    }
  }
})();
