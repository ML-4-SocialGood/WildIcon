# WildIcon project page

Static page, no build step. Open `index.html` through any static server:

```bash
python3 -m http.server 8791 --directory /data/yil708/GenerativeModel/WildIcon-page
```

## Included videos

Open `index.html` directly, or serve it at `http://localhost:8791/`. The page includes **49 curated Wan 2.2 videos and their 49 reference images**. The hero wall uses 28 reference photographs, the carousel shows seven videos with reference-image insets, and the gallery includes all 49 examples with species filters and a "Show all" button.

Video files in `static/preview/` are included when adding the page to a Git repository. The page displays the available clips without URL parameters, model badges or layout-review notices. Model provenance remains in the media metadata and manifests. Added WildIcon outputs take precedence over the existing Wan 2.2 clips, so they can be replaced one at a time.

```bash
python3 /data/yil708/GenerativeModel/WildIcon-page/tools/build_data.py --all
python3 /data/yil708/GenerativeModel/WildIcon-page/tools/add_preview.py
```

MP4s and posters are copied from the verified `outputs/website` bundle without another encode. Reference images are copied byte for byte. Reference IDs are omitted from the page; stable source mappings, SHA256 and prompt provenance are retained in `static/preview/manifest.json` and `static/data.js`. The authoritative selection and reference catalog remain in `outputs/website/`.

## Layout

Every example is the video at its own aspect ratio, with a reference inset in the lower-left corner and its species below the clip. Reference insets retain the whole photograph, without a text label, and use the freed space for a larger image. Clicking the video opens the larger reference/video comparison. Prompts appear at the top on hover, or below the clip on touch screens. The gallery uses justified rows: clips of similar shape are grouped, and each row shares one height while filling the full width. Until a video exists, the reference photograph fills its frame.

Carousel arrows step through the seven examples in order, even when multiple clips are visible on a wide screen. Native scrolling selects the closest clip to the centre once scrolling settles.

| Path | Content |
|---|---|
| `index.html` | Text: title, authors, links, abstract, method, dataset, BibTeX |
| `static/data.js` | The examples (`items`: reference photo, stable reference ID, prompt, `wildicon` video or `null`, optional `preview`), which of them appear in the hero wall (`heroWall`), the carousel (`featured`) and the gallery (`gallery`), and the frequency-lens images (`lens`) |
| `static/ref/` | Exact copies of the original reference photographs |
| `static/thumb/` | 640-px copies of the reference photographs for the hero wall and the gallery, made by `tools/make_thumbs.py` (rerun after changing the selection) |
| `static/social.jpg`, `static/favicon.svg` | Link-preview image (1200×630, used by `og:image`) and browser-tab icon |
| `static/wildicon/` | WildIcon videos and posters, written by `tools/add_wildicon.py` |
| `static/preview/` | Included Wan 2.2 videos, posters and source mapping |
| `static/lens/` | Reference photo, SAM 3 foreground and Gaussian high-pass map per lens animal, made by `tools/make_lens.py` (run with the `sam3` conda env) |
| `static/figures/` | Method and dataset figures, rendered from the camera-ready PDFs |
| `static/css/style.css`, `static/js/page.js` | Style and behaviour; no framework |

Fonts come from Google Fonts (Instrument Serif, Inter). The hero is a static, tilted wall of seven columns of rounded reference photographs (from `heroWall`), without parallax or automatic drifting. Other decorative animations respect reduced-motion preferences. Videos autoplay muted and loop when they scroll into view, pause when they leave it or the tab is hidden, and resume when the reader returns. If the browser blocks autoplay, native playback controls appear; interacting with the page also retries visible clips. The gallery shows its first half until "Show all" is clicked.

## Adding WildIcon videos

```bash
cd /data/yil708/GenerativeModel/WildIcon-page
python3 tools/add_wildicon.py 01_tiger /path/to/tiger.mp4 06_zebra /path/to/zebra.mp4
```

Each clip is re-encoded for the web (H.264, `yuv420p`, CRF 18, `+faststart`, native size and frame rate), a poster is taken from the middle frame, and the slot in `static/data.js` is filled. Only WildIcon outputs go in these slots.

To change which reference photographs are on the page, pass ids from `outputs/website/selection_spec.json`:

```bash
python3 tools/build_data.py --ids 01_tiger 02_panda 05_nyala ...
```

WildIcon videos and temporary preview entries already added are kept. Use `--all` to include the complete curated selection.

## Deploying to GitHub Pages

Project pages live at `https://<owner>.github.io/<repo>/`, separately from the personal site `<user>.github.io`, and an account can have any number of them. On a Free plan the repository must be public.

Branch `gh-pages` of the code repository, served at `https://ml-4-socialgood.github.io/WildIcon/`:

```bash
cd /data/yil708/GenerativeModel/WildIcon-page
git init -b gh-pages
git add -A
git commit -m "Add project page"
git remote add origin https://github.com/ML-4-SocialGood/WildIcon.git
git push -u origin gh-pages
gh api -X POST repos/ML-4-SocialGood/WildIcon/pages -f "source[branch]=gh-pages" -f "source[path]=/"
```

`.nojekyll` is included so GitHub serves the files as they are. `_preview/` holds local screenshots and is ignored by git.
