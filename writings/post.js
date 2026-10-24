// Metadata lives in the generated HTML; the runtime only renders the Markdown body.
function stripFrontMatter(text) {
  return text.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, "");
}

// Math extraction for raw HTML layout blocks.
// Markdown math and code are distinguished by the token extensions below.
const PLACEHOLDER = '\x00MATH';
const PLACEHOLDER_END = '\x00';

function extractMath(content, placeholder) {
  // Display math: $$...$$
  content = content.replace(/\$\$([\s\S]*?)\$\$/g,
    (_, math) => placeholder({ display: true, math }));

  // Inline math: $...$  (not preceded or followed by $)
  content = content.replace(/(?<!\$)\$(?!\$)((?:[^$\n]|\\.)+?)\$(?!\$)/g,
    (_, math) => placeholder({ display: false, math }));

  // Equation references share the equation font and link to the tagged display.
  content = content.replace(/\\eqref\{(\d+)\}/g,
    (_, reference) => placeholder({ reference }));

  return content;
}

// KaTeX macros shared by the article equations.
const KATEX_MACROS = {
  '\\E': '\\mathbb{E}',
  '\\pit': '{\\textcolor{#2e639b}{\\pi_\\theta}}',
  '\\pib': '{\\textcolor{#a33d32}{\\pi_b}}',
  '\\Qt': '{Q_{\\textcolor{#2e639b}{\\theta}}}',
  '\\Qb': '{Q_{\\textcolor{#a33d32}{b}}}',
};

function restoreMath(html, blocks) {
  // Placeholders may have been wrapped in <p> or other tags by marked;
  // use a global replace to find them wherever they ended up.
  const re = /\x00MATH(\d+)\x00([.,;:!?]*)/g;
  return html.replace(re, (_, i, punctuation) => {
    const { display, math, reference } = blocks[parseInt(i)];
    const macros = { ...KATEX_MACROS };
    const withPunctuation = rendered => {
      if (display || !punctuation) return rendered + punctuation;
      // Bind punctuation to the last KaTeX segment, retaining earlier breakpoints.
      // The complete MathML stays in the first fragment for assistive readers.
      const template = document.createElement('template');
      template.innerHTML = rendered;
      const mathRoot = template.content.querySelector('.katex');
      const visual = mathRoot && mathRoot.querySelector('.katex-html');
      const lastBase = visual && visual.lastElementChild;
      if (!reference && visual && visual.children.length > 1 && lastBase.classList.contains('base')) {
        const tail = mathRoot.cloneNode(false);
        const tailVisual = visual.cloneNode(false);
        tailVisual.append(lastBase);
        tail.append(tailVisual);
        return template.innerHTML + `<span class="math-with-punctuation">${tail.outerHTML}${punctuation}</span>`;
      }
      return `<span class="math-with-punctuation">${rendered}${punctuation}</span>`;
    };
    try {
      if (reference) {
        return withPunctuation(`<a class="eq-ref" href="#eq-${reference}" aria-label="Equation ${reference}">${katex.renderToString('(' + reference + ')', { throwOnError: false })}</a>`);
      }
      const rendered = katex.renderToString(math, { displayMode: display, throwOnError: false, macros });
      const tag = display && math.match(/\\tag\{(\d+)\}/);
      return withPunctuation(tag ? `<span class="equation-anchor" id="eq-${tag[1]}">${rendered}</span>` : rendered);
    } catch (e) {
      console.error('Equation rendering failed:', e.message);
      const escaped = String(math).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      return withPunctuation(`<code>${escaped}</code>`);
    }
  });
}

function extractCitations(content) {
  return content.replace(/[ \t]*\[@([a-z0-9]+(?:;[a-z0-9]+)*)\]/g, (_, refs) =>
    `<span class="citation-group" data-refs="${refs}"></span>`);
}

function renderCitations(root) {
  const references = new Map();
  root.querySelectorAll('[data-cite]').forEach(meta => {
    const item = meta.closest('li');
    const link = item && item.querySelector('a[href]');
    if (!item || !link) return;
    // Move the anchor from the marker span to the list item, so citation links land on the entry.
    item.id = meta.id;
    meta.removeAttribute('id');
    references.set(item.id.slice(4), { item, label: meta.dataset.cite, title: link.textContent });
  });
  const order = [];
  root.querySelectorAll('.citation-group').forEach(group => {
    group.dataset.refs.split(';').forEach(key => {
      if (references.has(key) && !order.includes(key)) order.push(key);
    });
  });
  references.forEach((_, key) => { if (!order.includes(key)) order.push(key); });
  order.forEach((key, index) => { references.get(key).number = index + 1; });

  root.querySelectorAll('.citation-group').forEach(group => {
    const refs = group.dataset.refs.split(';').map(key => references.get(key));
    group.replaceChildren();
    refs.forEach((ref, index) => {
      if (index) group.append(', ');
      if (!ref) { group.append('reference unavailable'); return; }
      const link = document.createElement('a');
      link.href = '#' + ref.item.id;
      link.title = ref.label + ' — ' + ref.title;
      link.setAttribute('aria-label', link.title);
      link.textContent = ref.number;
      group.append(link);
    });
  });

  if (order.length) {
    const oldList = references.get(order[0]).item.parentElement;
    const list = document.createElement('ol');
    list.className = 'reference-list';
    order.forEach(key => { list.append(references.get(key).item); });
    oldList.replaceWith(list);
  }
}

function configureExternalLinks(root) {
  root.querySelectorAll('a[href]').forEach(link => {
    const url = new URL(link.getAttribute('href'), document.baseURI);
    if ((url.protocol === 'https:' || url.protocol === 'http:') &&
        url.origin !== window.location.origin) {
      link.target = '_blank';
      link.relList.add('noopener', 'noreferrer');
    }
  });
}

function groupAuthorNotes(root) {
  const isAuthorNote = node => {
    if (!node || node.tagName !== 'P' || node.children.length !== 1) return false;
    const emphasis = node.firstElementChild;
    return emphasis.tagName === 'EM' &&
      node.textContent.trim() === emphasis.textContent.trim();
  };
  const notes = Array.from(root.children).filter(isAuthorNote);
  notes.forEach(paragraph => {
    if (paragraph.parentElement !== root) return;
    const preface = paragraph === root.firstElementChild;
    const group = document.createElement('aside');
    group.className = 'author-note-block' + (preface ? ' author-preface' : '');
    group.setAttribute('aria-label', preface ? "Author's note and acknowledgments" : "Author's note");
    paragraph.before(group);
    let node = paragraph;
    while (isAuthorNote(node)) {
      const next = node.nextElementSibling;
      node.classList.add('author-note');
      group.append(node, document.createTextNode('\n'));
      node = next;
    }
  });
}

function foldAppendix(root) {
  const heading = root.querySelector('#derivations-and-extensions')?.closest('h2');
  if (!heading || heading.parentElement !== root) return;

  const disclosure = document.createElement('details');
  disclosure.className = 'article-appendix';
  const summary = document.createElement('summary');
  const content = document.createElement('div');
  content.className = 'appendix-content';

  heading.before(disclosure);
  summary.append(heading);
  disclosure.append(summary, content);

  // Stop at the next main section, leaving References visible.
  let node = disclosure.nextSibling;
  while (node && !(node.nodeType === 1 && node.tagName === 'H2')) {
    const next = node.nextSibling;
    content.append(node);
    node = next;
  }
}

function parsePostMarkdown(content) {
  const blocks = [];
  const placeholder = block => {
    blocks.push(block);
    return PLACEHOLDER + (blocks.length - 1) + PLACEHOLDER_END;
  };

  // Let Markdown recognize code before interpreting math or citations.
  // Raw HTML is used for equation layouts and the intuition box in this post.
  const parser = new marked.Marked({
    breaks: false,
    gfm: true,
    extensions: [
      {
        name: 'displayMath',
        level: 'block',
        start: source => source.search(/^ {0,3}\$\$/m),
        tokenizer(source) {
          const match = /^ {0,3}\$\$([\s\S]*?)\$\$[ \t]*(?:\n|$)/.exec(source);
          if (match) return { type: 'displayMath', raw: match[0], math: match[1] };
        },
        renderer: token => '<p>' + placeholder({ display: true, math: token.math }) + '</p>\n'
      },
      {
        name: 'inlineMath',
        level: 'inline',
        start: source => source.search(/\$|\\eqref\{/),
        tokenizer(source) {
          if (this.lexer.state.inRawBlock) return;
          const display = /^\$\$([\s\S]*?)\$\$/.exec(source);
          if (display) return { type: 'inlineMath', raw: display[0], block: { display: true, math: display[1] } };
          const inline = /^\$(?!\$)((?:[^$\n]|\\.)+?)\$(?!\$)/.exec(source);
          if (inline) return { type: 'inlineMath', raw: inline[0], block: { display: false, math: inline[1] } };
          const reference = /^\\eqref\{(\d+)\}/.exec(source);
          if (reference) return { type: 'inlineMath', raw: reference[0], block: { reference: reference[1] } };
        },
        renderer: token => placeholder(token.block)
      },
      {
        name: 'citation',
        level: 'inline',
        start: source => source.search(/[ \t]*\[@/),
        tokenizer(source) {
          if (this.lexer.state.inRawBlock) return;
          const match = /^[ \t]*\[@([a-z0-9]+(?:;[a-z0-9]+)*)\]/.exec(source);
          if (match) return { type: 'citation', raw: match[0] };
        },
        renderer: token => extractCitations(token.raw)
      }
    ],
    renderer: {
      html(source) {
        return source.split(/(<(?:pre|code|script|style|kbd)\b[^>]*>[\s\S]*?<\/(?:pre|code|script|style|kbd)\s*>)/gi)
          .map(part => {
            if (/^<(?:pre|code|script|style|kbd)\b/i.test(part)) return part;
            return extractCitations(extractMath(part, placeholder));
          }).join('');
      }
    }
  });

  return restoreMath(parser.parse(content), blocks);
}

function renderPost(content, sourceURL) {
  const root = document.getElementById('article-body');
  // Resolve paths against the Markdown source, before attaching images to the page.
  const template = document.createElement('template');
  template.innerHTML = parsePostMarkdown(content);
  template.content.querySelectorAll('a[href], img[src]').forEach(element => {
    const attribute = element.tagName === 'IMG' ? 'src' : 'href';
    const value = element.getAttribute(attribute);
    if (value && !value.startsWith('#')) {
      element.setAttribute(attribute, new URL(value, sourceURL).href);
    }
  });
  root.replaceChildren(template.content);
  renderCitations(root);
  configureExternalLinks(root);
  groupAuthorNotes(root);
  foldAppendix(root);
  buildArticleToc(root);
}

function buildArticleToc(root) {
  const panel = document.createElement('details');
  panel.className = 'article-toc';
  const summary = document.createElement('summary');
  summary.textContent = 'Contents';
  const title = document.createElement('div');
  title.className = 'toc-title';
  title.textContent = 'Contents';
  title.setAttribute('aria-hidden', 'true');
  const nav = document.createElement('nav');
  nav.setAttribute('aria-label', 'On this page');
  const list = document.createElement('ul');
  const entries = [...root.querySelectorAll('h2')].map(heading => {
    let target = heading.querySelector('[id]');
    if (!target) {
      target = document.createElement('span');
      const stem = heading.textContent.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
      let id = stem, suffix = 2;
      while (document.getElementById(id)) id = stem + '-' + suffix++;
      target.id = id;
      target.append(...heading.childNodes);
      heading.append(target);
    }
    const link = document.createElement('a');
    link.href = '#' + target.id;
    link.textContent = heading.textContent.trim();
    const item = document.createElement('li');
    item.append(link);
    list.append(item);
    return { target, link };
  });
  nav.append(list);
  panel.append(summary, title, nav);
  root.before(panel);
  const wide = window.matchMedia('(min-width: 1280px)');
  panel.open = wide.matches;
  wide.addEventListener('change', () => { panel.open = wide.matches; updateCurrent(); });

  function updateCurrent() {
    let current = entries[0];
    for (const entry of entries) {
      if (entry.target.getClientRects().length && entry.target.getBoundingClientRect().top <= 135) current = entry;
    }
    for (const entry of entries) {
      if (entry === current) entry.link.setAttribute('aria-current', 'location');
      else entry.link.removeAttribute('aria-current');
    }
  }
  let queued = false;
  window.addEventListener('scroll', () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; updateCurrent(); });
  }, { passive: true });
  window.addEventListener('resize', updateCurrent);
  root.querySelector('.article-appendix')?.addEventListener('toggle', updateCurrent);
  document.fonts.ready.then(updateCurrent);
  updateCurrent();
}

function postAnchorTarget(hash) {
  if (!hash) return null;
  try { return document.getElementById(decodeURIComponent(hash.slice(1))); }
  catch { return null; }
}

function revealDetailsForTarget(target) {
  for (let node = target; node; node = node.parentElement) {
    if (node.tagName === 'DETAILS') node.open = true;
  }
}

let highlightedAnchor = null;
let anchorFadeTimer;

function clearAnchorHighlight() {
  clearTimeout(anchorFadeTimer);
  highlightedAnchor?.classList.remove('anchor-target');
  highlightedAnchor = null;
}

function scheduleAnchorFade() {
  if (!highlightedAnchor) return;
  clearTimeout(anchorFadeTimer);
  anchorFadeTimer = setTimeout(clearAnchorHighlight, 1500);
}

async function restorePostAnchor() {
  const hash = window.location.hash;
  const target = postAnchorTarget(hash);
  clearAnchorHighlight();
  if (!target) return;
  revealDetailsForTarget(target);
  await document.fonts.ready;
  requestAnimationFrame(() => {
    if (window.location.hash !== hash) return;
    // Mark after rendering, including targets loaded from an initial URL fragment.
    highlightedAnchor = target;
    target.classList.add('anchor-target');
    const behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      ? 'instant' : 'smooth';
    target.scrollIntoView({ behavior, block: 'start' });
    scheduleAnchorFade();
  });
}

window.addEventListener('hashchange', restorePostAnchor);
// Keep the full highlight while scrolling; its 1.5-second hold starts when scrolling stops.
window.addEventListener('scroll', scheduleAnchorFade, { passive: true });
document.querySelector('article').addEventListener('click', event => {
  if (event.defaultPrevented || event.button !== 0 ||
      event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const link = event.target.closest('a[href^="#"]');
  const hash = link?.getAttribute('href');
  const target = postAnchorTarget(hash);
  if (!target) return;

  // Reveal before the browser's fragment navigation. Repeated links to the
  // current hash also work after the reader manually closes the appendix.
  revealDetailsForTarget(target);
  const toc = link.closest('.article-toc');
  if (toc && !window.matchMedia('(min-width: 1280px)').matches) toc.open = false;
  if (window.location.hash === hash) {
    event.preventDefault();
    restorePostAnchor();
  }
});

let appendixPrintState = null;
function expandAppendixForPrint() {
  if (appendixPrintState !== null) return;
  appendixPrintState = Array.from(document.querySelectorAll('.article-appendix'))
    .map(element => ({ element, open: element.open }));
  appendixPrintState.forEach(({ element }) => { element.open = true; });
}

function restoreAppendixAfterPrint() {
  if (appendixPrintState === null) return;
  appendixPrintState.forEach(({ element, open }) => { element.open = open; });
  appendixPrintState = null;
}

window.addEventListener('beforeprint', expandAppendixForPrint);
window.addEventListener('afterprint', restoreAppendixAfterPrint);
window.matchMedia('print').addEventListener('change', event => {
  if (event.matches) expandAppendixForPrint();
  else restoreAppendixAfterPrint();
});

// --- Main loader ---
async function loadPost() {
  const root = document.getElementById('article-body');
  try {
    const source = new URL(document.body.dataset.source, document.baseURI);
    const res = await fetch(source, { cache: 'no-cache' });
    if (!res.ok) throw new Error(res.status === 404 ? 'Article source not found.' : 'Could not load the article.');
    const content = stripFrontMatter(await res.text());
    renderPost(content, source);
    restorePostAnchor();
  } catch (error) {
    console.error('Article loading failed:', error);
    const message = document.createElement('p');
    message.textContent = 'The article could not be loaded. ';
    const retry = document.createElement('a');
    retry.href = window.location.href;
    retry.textContent = 'Reload the page';
    message.append(retry, '.');
    root.replaceChildren(message);
  } finally {
    root.removeAttribute('aria-busy');
  }
}

loadPost();
