# Project website and paper publication

The public site is **https://yifanzhang-pro.github.io/KLPO/**. It uses static HTML, CSS, and JavaScript, using the same publication template as the [Recurrent Looped Transformer project](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/blob/master/index.html).

## Preview

From the repository root:

```bash
python -m http.server 8000
```

Open http://localhost:8000. No package installation or frontend build is required. Core content, links, and the default method remain readable without JavaScript. JavaScript adds the route/estimator explorer, citation copying, and mobile navigation. Styles, scripts, and figures are served locally. Space Grotesk and Inter load from Google Fonts, matching the reference template; system fonts are the fallback.

## Publish

GitHub Pages serves the root of the `main` branch. In repository **Settings → Pages**, the source is **Deploy from a branch → main → / (root)**. A push to `main` republishes the site. `.nojekyll` keeps the site static.

- `index.html`: page content and publication metadata.
- `assets/site/template.css`: the reference page’s stylesheet, copied from [RLT commit d0b24c4](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/commit/d0b24c4ae97300227f59aba648e4b1f0d3ba8098). Keep its monochrome palette, centered grid hero, navigation, typography, rounded buttons, section widths, and footer aligned with that template.
- `assets/site/style.css`: KLPO-specific method controls and responsive/accessibility additions.
- `assets/site/`: navigation script, favicon, and social preview. The method data and loss APIs are independent of the presentation template.
- `KLPO.pdf`: downloadable paper, also linked from the README.
- `assets/token-regression-mc.png`: the paper's Figure 1, exported from PDF page 3.
- `citation.bib`: downloadable citation; keep the README and site's citation in sync.

## Paper provenance

The current `KLPO.pdf` is an unmodified pdfLaTeX build of `main.tex` from [RPG-2-Overleaf commit 9a778da](https://github.com/yifanzhang-pro/RPG-2-Overleaf/commit/9a778da), compiled on Overleaf. It is a 58-page technical report, *On KL-Regularized Policy Optimization*, dated September 18, 2026, revised October 5, 2026. The paper is on arXiv as [2610.08963](https://arxiv.org/abs/2610.08963) (cs.LG); v1, submitted October 6, 2026, was built from [commit b485b29](https://github.com/yifanzhang-pro/RPG-2-Overleaf/commit/b485b29), so the PDF here is newer than v1 and adds the β → 0⁺ extension of the update. The site and citation reproduce the author line as “Yifan Zhang, Princeton University” and cite the arXiv preprint; no venue is asserted.

When publishing a paper revision:

1. Compile `main.tex` from the paper repository with pdfLaTeX (on Overleaf, or `latexmk -pdf main.tex`) and visually verify the PDF.
2. Copy it here as `KLPO.pdf`; record the source commit and revision date in this file, the README, and `index.html`.
3. Re-export Figure 1 if it changed, then check its crop and legibility. For the current page geometry, the export command is:

   ```bash
   pdftoppm -f 3 -l 3 -singlefile -r 200 -x 192 -y 192 -W 1316 -H 1189 \
     -png KLPO.pdf assets/token-regression-mc
   ```

4. Preview desktop and mobile layouts; check the method selector, copy button, navigation, and PDF/BibTeX downloads.
5. Push `main` and verify the live site and `/KLPO.pdf` after Pages finishes.

The default must remain **KLPO token regression + MC-KL** in the hero, initial explorer state, README, examples, and tables. Keep the name **TopK-KL** for the aggregated head/tail estimator. The website reports theoretical properties and implementation checks; do not present these as measured training or benchmark results.
