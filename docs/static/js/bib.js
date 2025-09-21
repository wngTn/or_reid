
(function () {
  function dedent(text) {
    const lines = text.replace(/\r\n/g, '\n').split('\n');
    while (lines.length && !lines[0].trim()) lines.shift();
    while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
    let min = Infinity;
    lines.forEach(l => { if (l.trim()) { const m = l.match(/^[ \t]*/); if (m) min = Math.min(min, m[0].length); } });
    if (!isFinite(min)) return text.trim();
    return lines.map(l => l.slice(min)).join('\n');
  }

  function parseFields(body) {
    const fields = {};
    let i = 0, n = body.length;
    function ws() { while (i < n && /\s|,/.test(body[i])) i++; }
    function ident() { const m = body.slice(i).match(/^([A-Za-z][\w\-]*)/); if (!m) return null; i += m[0].length; return m[0].toLowerCase(); }
    function value() {
      ws(); if (i >= n) return '';
      const ch = body[i];
      if (ch === '{') {
        let depth = 0, start = ++i;
        while (i < n) { if (body[i] === '{') depth++; else if (body[i] === '}') { if (depth === 0) break; depth--; } i++; }
        const val = body.slice(start, i); i++; return val.trim().replace(/\s+/g, ' ');
      } else if (ch === '"') {
        let start = ++i, out = '', escaped = false;
        while (i < n) { const c = body[i]; if (!escaped && c === '"') break; escaped = (!escaped && c === '\\'); i++; }
        out = body.slice(start, i); i++; return out.trim().replace(/\s+/g, ' ');
      } else {
        const m = body.slice(i).match(/^([^,}]+)/); if (!m) return ''; i += m[0].length; return m[0].trim().replace(/\s+/g, ' ');
      }
    }
    while (i < n) {
      ws(); const k = ident(); if (!k) break; ws(); if (body[i] === '=') i++;
      const v = value(); fields[k] = v; ws(); if (body[i] === ',') i++;
    }
    return fields;
  }

  function parseBibtex(text) {
    const entries = [];
    const t = dedent(text);
    const re = /@([A-Za-z]+)\s*\{\s*([^,]+)\s*,([\s\S]*?)\}\s*(?=@|$)/g;
    let m;
    while ((m = re.exec(t)) !== null) {
      const type = m[1], key = m[2].trim(), body = m[3].trim();
      const fields = parseFields(body);
      entries.push({ type, key, fields, raw: '@' + type + '{' + key + ',\n' + body + '\n}' });
    }
    return entries;
  }

  function authorsFull(authorField) {
    if (!authorField) return 'Unknown authors';
    const parts = authorField.split(/\s+and\s+/i).map(s => s.trim()).filter(Boolean);
    if (parts.length <= 1) return parts[0] || 'Unknown authors';
    // Join all authors with commas; last joined with ' and '
    return parts.slice(0, -1).join(', ') + ' and ' + parts[parts.length - 1];
  }

  function venueText(fields) {
    if (fields.journal) return fields.journal;
    if (fields.booktitle) return fields.booktitle;
    if (fields.organization) return fields.organization;
    return '';
  }

  function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(text); }
    else {
      const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); document.body.removeChild(ta);
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    const bibSection = document.getElementById('BibTeX');
    if (!bibSection) return;

    const codeEl = bibSection.querySelector('pre code') || bibSection.querySelector('pre');
    if (!codeEl) return;
    const raw = codeEl.textContent;

    const entries = parseBibtex(raw);
    if (!entries.length) {
      const warn = document.createElement('p');
      warn.style.cssText = 'color:#b91c1c;background:#fff0f0;border:1px solid #fecaca;border-radius:8px;padding:.5rem .75rem;';
      warn.textContent = 'Could not parse BibTeX. Please check the syntax.';
      codeEl.parentElement.insertAdjacentElement('afterend', warn);
      return;
    }

    // Mark and hide the raw <pre>
    const pre = codeEl.closest('pre');
    if (pre) { pre.classList.add('bib-raw'); }

    // Ensure / create container
    let container = bibSection.querySelector('.bib-entries');
    if (!container) {
      container = document.createElement('div');
      container.className = 'bib-entries';
      if (pre) pre.insertAdjacentElement('afterend', container);
      else bibSection.appendChild(container);
    } else {
      container.innerHTML = '';
    }

    // Intro
    const intro = document.createElement('div');
    intro.className = 'bib-intro';
    intro.textContent = 'Citations are numbered below. Use the copy button to grab BibTeX. Link directly with #<key> (e.g., #wang2025_beyond_role).';
    container.insertAdjacentElement('beforebegin', intro);

    // Render entries
    entries.forEach((e, idx) => {
      const card = document.createElement('article');
      card.className = 'bib-card';
      card.id = e.key;

      const title = document.createElement('h3');
      title.className = 'bib-title';
      const a = document.createElement('a');
      a.href = '#' + e.key;
      a.textContent = '[' + (idx + 1) + '] ' + (e.fields.title || e.key);
      title.appendChild(a);

      const meta = document.createElement('div');
      meta.className = 'bib-meta';
      const authors = authorsFull(e.fields.author);
      const venue = venueText(e.fields);
      const year = e.fields.year ? (' (' + e.fields.year + ')') : '';
      meta.innerHTML = authors + (venue ? ' · <span class="venue">' + venue + '</span>' : '') + year;

      // Actions: only Copy button (no Link button)
      const actions = document.createElement('div');
      actions.className = 'bib-actions';
      const copyBtn = document.createElement('button');
      copyBtn.className = 'button is-small is-light';
      copyBtn.textContent = 'Copy BibTeX';
      copyBtn.addEventListener('click', function () {
        copyToClipboard(e.raw);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy BibTeX'; }, 1200);
      });
      actions.appendChild(copyBtn);

      // Collapsible BibTeX
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = 'Show BibTeX';
      const preEl = document.createElement('pre');
      const code = document.createElement('code');
      code.textContent = e.raw;
      preEl.appendChild(code);
      details.appendChild(summary);
      details.appendChild(preEl);

      card.appendChild(title);
      card.appendChild(meta);
      card.appendChild(actions);
      card.appendChild(details);
      container.appendChild(card);
    });

    // In-text citations
    const keyToIndex = new Map(entries.map((e, i) => [e.key, i + 1]));
    document.querySelectorAll('a.cite[data-key]').forEach((el) => {
      const k = el.getAttribute('data-key');
      const n = keyToIndex.get(k);
      if (n) {
        el.textContent = '[' + n + ']';
        el.setAttribute('href', '#' + k);
        el.setAttribute('title', (entries[n - 1].fields.title || k));
      }
    });
  });
})();
