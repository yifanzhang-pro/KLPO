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
- `assets/token-regression-mc.png`: the paper's Figure 1, exported from PDF page 2.
- `citation.bib`: downloadable citation; keep the README and site's citation in sync.

## Paper provenance

The current `KLPO.pdf` is an unmodified copy of `report.pdf` from [RPG-2-Overleaf commit 1658d8d](https://github.com/yifanzhang-pro/RPG-2-Overleaf/commit/1658d8d). It is a 56-page technical report dated September 18, 2026, revised September 20, 2026. The site and citation reproduce the author line as “Yifan Zhang et al.”; no venue, DOI, or arXiv identifier is asserted.

When publishing a paper revision:

1. Build and visually verify `report.pdf` in the paper repository.
2. Copy it here as `KLPO.pdf`; record the source commit and revision date in this file, the README, and `index.html`.
3. Re-export Figure 1 if it changed, then check its crop and legibility. For the current page geometry, the export command is:

   ```bash
   pdftoppm -f 2 -l 2 -singlefile -r 200 -x 180 -y 178 -W 1340 -H 1272 \
     -png KLPO.pdf assets/token-regression-mc
   ```

4. Preview desktop and mobile layouts; check the method selector, copy button, navigation, and PDF/BibTeX downloads.
5. Push `main` and verify the live site and `/KLPO.pdf` after Pages finishes.

The default must remain **KLPO token regression + MC-KL** in the hero, initial explorer state, README, examples, and tables. Keep the name **TopK-KL** for the aggregated head/tail estimator. The website reports theoretical properties and implementation checks; do not present these as measured training or benchmark results.
