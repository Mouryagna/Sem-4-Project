// ── Navigation Toggle ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('navToggle');
  const menu = document.getElementById('navMenu');
  if (toggle && menu) {
    toggle.addEventListener('click', () => menu.classList.toggle('open'));
    document.querySelectorAll('.nav-link').forEach(l =>
      l.addEventListener('click', () => menu.classList.remove('open'))
    );
  }

  // Navbar scroll effect
  window.addEventListener('scroll', () => {
    const nav = document.getElementById('navbar');
    if (nav) nav.style.background = window.scrollY > 50
      ? 'rgba(10,14,23,.95)' : 'rgba(10,14,23,.8)';
  });

  // Counter animation
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseFloat(el.dataset.count);
    const decimals = (el.dataset.decimals || '0');
    animateCounter(el, target, parseInt(decimals));
  });
});

function animateCounter(el, target, decimals = 0) {
  let current = 0;
  const duration = 1500;
  const step = target / (duration / 16);
  const timer = setInterval(() => {
    current += step;
    if (current >= target) { current = target; clearInterval(timer); }
    el.textContent = current.toFixed(decimals);
  }, 16);
}

// ── Predict Form Handler ───────────────────────────────────
function handlePredict(e) {
  e.preventDefault();
  const btn = document.getElementById('predictBtn');
  const resultCard = document.getElementById('resultCard');
  const form = e.target;

  const weekdayMap = { Monday:0, Tuesday:1, Wednesday:2, Thursday:3, Friday:4, Saturday:5, Sunday:6 };

  const data = {
    pm25: parseFloat(form.pm25.value),
    pm10: parseFloat(form.pm10.value),
    no2:  parseFloat(form.no2.value),
    so2:  parseFloat(form.so2.value),
    co:   parseFloat(form.co.value),
    o3:   parseFloat(form.o3.value),
    hour:   parseInt(form.hour.value),
    day:    parseInt(form.day.value),
    month:  parseInt(form.month.value),
    weekday: weekdayMap[form.weekday.value],
    season: form.season.value
  };

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting...';

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  })
  .then(r => r.json())
  .then(res => {
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-brain"></i> Predict AQI';
    if (res.success) {
      document.getElementById('resultAqi').textContent = res.aqi;
      document.getElementById('resultAqi').style.color = res.color;
      document.getElementById('resultCat').textContent = res.category;
      document.getElementById('resultCat').style.color = res.color;
      document.getElementById('resultIcon').className = res.icon;
      document.getElementById('resultIcon').style.color = res.color;
      resultCard.classList.add('show');
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else {
      alert('Prediction failed: ' + (res.error || 'Unknown error'));
    }
  })
  .catch(err => {
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-brain"></i> Predict AQI';
    alert('Error: ' + err.message);
  });
}

// ── Clear History ──────────────────────────────────────────
function clearHistory() {
  if (!confirm('Clear all prediction history?')) return;
  fetch('/api/clear-history', { method: 'POST' })
    .then(r => r.json())
    .then(() => location.reload())
    .catch(err => alert('Error: ' + err.message));
}

// ── Export History to CSV ──────────────────────────────────
function exportCSV() {
  const table = document.querySelector('.data-table');
  if (!table) return;
  let csv = [];
  table.querySelectorAll('tr').forEach(row => {
    const cols = [];
    row.querySelectorAll('th, td').forEach(c => cols.push('"' + c.textContent.trim() + '"'));
    csv.push(cols.join(','));
  });
  const blob = new Blob([csv.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'aqi_predictions.csv';
  a.click();
}
