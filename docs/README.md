# Academic Paper Project Page Template

A sleek, modern, and responsive template for academic research papers. Designed for high-impact visual presentation, featuring dark/light mode, synchronized video comparisons, and built-in research tools like BibTeX copy-to-clipboard and image lightboxes.

## ✨ Features

- **Responsive Design:** Mobile-friendly layout using Bulma CSS.
- **Dark/Light Mode:** Seamless theme switching with system preference detection.
- **Synchronized Videos:** Compare results side-by-side with a single toggle button.
- **Researcher Tools:** 
    - One-click BibTeX copy functionality.
    - Minimalist Image Lightbox for detailed figure viewing.
- **Modern UI:** Smooth scroll reveal animations, sticky navigation, and refined typography.
- **Privacy Conscious:** Includes a GDPR-compliant banner for analytics transparency.

## 🚀 Getting Started

### 1. Folder Structure
- `index.html`: The main content and structure.
- `static/css/blog.css`: All custom styling and theme variables.
- `static/js/main.js`: Interaction logic (videos, theme, lightbox, copy).
- `static/images/`: Store your figures, icons, and author avatars here.
- `static/videos/`: Store your result mp4s here.
- `static/materials/`: Place your PDFs (Paper, Poster, Supplementary) here.

### 2. Customization

#### Theme Colors
Open `static/css/blog.css` and modify the `:root` and `body.dark-theme` variables.
- `--accent`: The primary color for links and buttons.
- `--bg-primary`: Main background color.
- `--text-title`: Color for headings.

#### Paper Details
In `index.html`, search for the following sections to update:
- `<title>` and Meta tags for SEO.
- `.publication-title`: Your paper's title.
- `.authors-grid`: Author names, links, and avatars.
- `.publication-links`: Links to arXiv, Code, and PDF.
- `#abstract`: Your paper's abstract.
- `#BibTeX`: Your citation in BibTeX format.

### 3. Adding Figures & Videos
- **Figures:** Wrap images in a `<div class="figure">` with a `<div class="fig-caption">` for consistent styling and lightbox support.
- **Videos:** Use the `<div class="video-grid">` structure for side-by-side comparisons.

## 📄 License
This template is open-source. Feel free to use it for your research projects.

---
Created with ❤️ for the research community.
