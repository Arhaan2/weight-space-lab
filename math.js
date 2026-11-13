/* Local filtering only. The complete report is ordinary HTML without scripts. */
(() => {
  const filter = document.getElementById('outcome-filter');
  const controls = document.querySelector('.math-filter');
  const count = document.getElementById('outcome-count');
  const empty = document.getElementById('outcome-empty');
  const records = document.getElementById('problem-records');
  const rows = [...document.querySelectorAll('#problem-records tbody tr[data-outcome]')];
  if (!filter || !controls || !count || !empty || !rows.length) return;
  controls.hidden = false;
  const revealRecords = () => {
    if (location.hash === '#problem-records' && records) records.open = true;
  };
  document.querySelectorAll('a[href="#problem-records"]').forEach(link => {
    link.addEventListener('click', () => { if (records) records.open = true; });
  });
  window.addEventListener('hashchange', revealRecords);
  revealRecords();
  filter.addEventListener('change', () => {
    let shown = 0;
    rows.forEach(row => {
      row.hidden = filter.value !== 'all' && row.dataset.outcome !== filter.value;
      shown += !row.hidden;
    });
    count.textContent = `${shown} of ${rows.length} problems`;
    empty.hidden = shown !== 0;
  });
})();
