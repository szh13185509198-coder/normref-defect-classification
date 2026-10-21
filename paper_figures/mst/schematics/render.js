// Render .drawio files to vector PDF and 300 dpi PNG with the official diagrams.net viewer in headless Chromium.
// Usage: node render.js <viewer-static.min.js> <out_dir> <file.drawio>...
// The PDF page is the diagram's bounding box at 96 px per inch, so a 567 px wide page prints at 15 cm.
// With TEXT_METRICS_DIR set, also writes <name>.text.json: the rendered box and font size of every text run, measured
// by the browser (used by check_figs.py for the font-size, overlap and page-bound checks).
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const [viewer, outDir, ...files] = process.argv.slice(2);
  const b = await chromium.launch({ args: ['--no-sandbox'],
    ...(process.env.HTTPS_PROXY ? { proxy: { server: process.env.HTTPS_PROXY } } : {}) });
  for (const f of files) {
    const xml = fs.readFileSync(f, 'utf8');
    const name = path.basename(f, '.drawio');
    const cfg = JSON.stringify({ xml, highlight: 'none', lightbox: false, border: 1 });
    const html = '<div id="g" class="mxgraph"></div>';
    for (const [kind, scale] of [['pdf', 1], ['png', 300 / 96]]) {
      const ctx = await b.newContext({ deviceScaleFactor: scale, viewport: { width: 1400, height: 1200 } });
      const p = await ctx.newPage();
      await p.setContent(html);
      await p.addStyleTag({ content: 'html,body{margin:0;padding:0;background:#fff}' });
      await p.evaluate((c) => document.getElementById('g').setAttribute('data-mxgraph', c), cfg);
      await p.addScriptTag({ path: viewer });
      await p.waitForTimeout(3000);
      await p.evaluate(() => new Promise((r) => window.GraphViewer.createViewerForElement(document.getElementById('g'), () => r(1))));
      await p.waitForFunction(() => document.querySelector('svg'), null, { timeout: 60000 });
      await p.waitForTimeout(1500);
      const box = await p.evaluate(() => { const r = document.querySelector('svg').getBoundingClientRect();
        return { w: Math.ceil(r.width), h: Math.ceil(r.height) }; });
      if (kind === 'pdf' && process.env.TEXT_METRICS_DIR) {
        const runs = await p.evaluate(() => {
          const svg = document.querySelector('svg');
          const o = svg.getBoundingClientRect();
          const out = [];
          const walk = document.createTreeWalker(svg, NodeFilter.SHOW_TEXT);
          let n;
          while ((n = walk.nextNode())) {
            if (!n.textContent.trim()) continue;
            const r = document.createRange();
            r.selectNodeContents(n);
            const rects = [...r.getClientRects()].filter((q) => q.width > 0 && q.height > 0);
            if (!rects.length) continue;
            const x0 = Math.min(...rects.map((q) => q.left)), y0 = Math.min(...rects.map((q) => q.top));
            const x1 = Math.max(...rects.map((q) => q.right)), y1 = Math.max(...rects.map((q) => q.bottom));
            const el = n.parentElement;
            out.push({ text: n.textContent.trim(), x: x0 - o.left, y: y0 - o.top, w: x1 - x0, h: y1 - y0,
                       lines: rects.map((q) => [q.left - o.left, q.top - o.top, q.width, q.height]),
                       font_px: parseFloat(getComputedStyle(el).fontSize), family: getComputedStyle(el).fontFamily });
          }
          return { page: { w: o.width, h: o.height }, runs: out };
        });
        fs.writeFileSync(path.join(process.env.TEXT_METRICS_DIR, `${name}.text.json`), JSON.stringify(runs));
      }
      if (kind === 'pdf') {
        await p.pdf({ path: path.join(outDir, `${name}.pdf`), width: `${box.w}px`, height: `${box.h}px`,
                      printBackground: true, pageRanges: '1' });
        console.log(`${name}: ${box.w} x ${box.h} px = ${(box.w / 96 * 2.54).toFixed(1)} x ${(box.h / 96 * 2.54).toFixed(1)} cm`);
      } else {
        await (await p.$('svg')).screenshot({ path: path.join(outDir, `${name}.png`), omitBackground: false });
      }
      await ctx.close();
    }
  }
  await b.close();
})();
