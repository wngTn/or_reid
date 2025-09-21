
(function () {
  function slugify(text) {
    return text.toString().toLowerCase()
      .replace(/\s+/g, '-')
      .replace(/[^\w\-]+/g, '')
      .replace(/\-\-+/g, '-')
      .replace(/^-+/, '')
      .replace(/-+$/, '');
  }

  function ensureId(el, prefix) {
    if (el.id) return el.id;
    const base = prefix + '-' + slugify(el.textContent.trim().slice(0, 80));
    let candidate = base || (prefix + '-' + Math.random().toString(36).slice(2,8));
    let i = 1;
    while (document.getElementById(candidate)) {
      candidate = base + '-' + i++;
    }
    el.id = candidate;
    return candidate;
  }

  function stripLeadingNumbering(text) {
    // Remove patterns like "1) ", "2.1) ", "3. ", "4 - ", "5- ", etc., at the start
    return text.replace(/^\s*\d+(?:\.\d+)*\s*[\)\.\-–:]?\s*/, '');
  }

  document.addEventListener('DOMContentLoaded', function () {
    const contentRoot = document.getElementById('article-content');
    const tocList = document.getElementById('toc-list');

    if (!contentRoot || !tocList) return;

    // Reset TOC list in case of re-runs
    tocList.innerHTML = '';

    // Collect sections (H2) and their subsections (H3)
    const sections = Array.from(contentRoot.querySelectorAll('section.blog-section'));
    let sectionCounter = 0;

    sections.forEach((sec) => {
      const h2 = sec.querySelector('h2.title');
      if (!h2) return;
      sectionCounter += 1;

      // Normalize heading text by removing any manual numbering first
      const h2Base = stripLeadingNumbering(h2.textContent.trim());
      h2.textContent = sectionCounter + ') ' + h2Base;
      const h2Id = ensureId(h2, 's' + sectionCounter);

      // Create TOC item for H2
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = '#' + h2Id;
      a.textContent = h2.textContent;
      li.appendChild(a);
      tocList.appendChild(li);

      // Subsections (H3) within this section
      const h3s = Array.from(sec.querySelectorAll('h3.subtitle'));
      if (h3s.length) {
        let subCounter = 0;
        const ol = document.createElement('ol');
        h3s.forEach((h3) => {
          subCounter += 1;
          const h3Base = stripLeadingNumbering(h3.textContent.trim());
          h3.textContent = sectionCounter + '.' + subCounter + ') ' + h3Base;
          const h3Id = ensureId(h3, 's' + sectionCounter + '-' + subCounter);

          const subLi = document.createElement('li');
          const subA = document.createElement('a');
          subA.href = '#' + h3Id;
          subA.textContent = h3.textContent;
          subLi.appendChild(subA);
          ol.appendChild(subLi);
        });
        li.appendChild(ol);
      }
    });
  });
})();
