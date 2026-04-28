/**
 * Main site interactions and logic.
 * Handles GDPR, Reading Progress, Scroll Reveal, Video Synchronization, and Theme Toggle.
 */

(function () {
    "use strict";

    // --- Configuration & Constants ---
    const CONFIG = {
        scrollThreshold: 0.1,
        gdprKey: "gdprAccepted",
        themeKey: "site-theme",
        revealClass: "reveal",
        revealActiveClass: "active",
        progressBarId: "progress-bar"
    };

    /**
     * Initializes GDPR banner logic.
     */
    function initGDPR() {
        const banner = document.getElementById("gdpr-banner");
        const acceptBtn = document.getElementById("accept-gdpr");

        if (!banner || !acceptBtn) return;

        if (localStorage.getItem(CONFIG.gdprKey) !== "true") {
            banner.style.display = "flex";
        }

        acceptBtn.addEventListener("click", () => {
            banner.style.display = "none";
            localStorage.setItem(CONFIG.gdprKey, "true");
        });
    }

    /**
     * Initializes scroll reveal animations.
     */
    function initScrollReveal() {
        const revealElements = document.querySelectorAll(`.${CONFIG.revealClass}`);
        if (!revealElements.length) return;

        const revealCallback = (entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add(CONFIG.revealActiveClass);
                }
            });
        };

        const observer = new IntersectionObserver(revealCallback, { 
            threshold: CONFIG.scrollThreshold 
        });

        revealElements.forEach(el => observer.observe(el));
    }

    /**
     * Initializes synchronized video playback for comparison.
     */
    function initVideoSync() {
        const video1 = document.getElementById("team1");
        const video2 = document.getElementById("team2");
        const toggleBtn = document.getElementById("videoToggleBtn");

        if (!video1 || !video2 || !toggleBtn) return;

        const icon = toggleBtn.querySelector("i");
        const btnText = toggleBtn.querySelector(".btn-text");

        const syncVideos = () => {
            const diff = Math.abs(video1.currentTime - video2.currentTime);
            if (diff > 0.1) {
                video2.currentTime = video1.currentTime;
            }
        };

        const updateUI = (isPaused) => {
            if (isPaused) {
                toggleBtn.classList.add("is-paused");
                btnText.textContent = "Play Both";
            } else {
                toggleBtn.classList.remove("is-paused");
                btnText.textContent = "Pause Both";
            }
        };

        toggleBtn.addEventListener("click", () => {
            if (video1.paused) {
                syncVideos();
                Promise.all([video1.play(), video2.play()])
                    .then(() => updateUI(false))
                    .catch(err => console.warn("Video playback was interrupted:", err));
            } else {
                video1.pause();
                video2.pause();
                updateUI(true);
            }
        });

        // Keep in sync when one is manually played
        video1.addEventListener("play", () => {
            syncVideos();
            updateUI(false);
        });
        video1.addEventListener("pause", () => updateUI(true));

        // Initial state check
        if (video1.paused) {
            updateUI(true);
            // Try to autoplay explicitly in case the attribute was blocked
            video1.play().then(() => updateUI(false)).catch(() => {});
            video2.play().catch(() => {});
        } else {
            updateUI(false);
        }
    }

    /**
     * Handles smooth scrolling to anchors with precise navbar offset.
     */
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                const href = this.getAttribute('href');
                if (href === "#") return;

                const target = document.querySelector(href);
                if (!target) return;

                e.preventDefault();

                // Get dynamic navbar height
                const navbar = document.querySelector('.navbar');
                const navbarHeight = navbar ? navbar.offsetHeight : 0;

                // Force reveal instantly so we can measure the "final" position
                // Otherwise, the translateY(30px) will throw off the measurement
                const isReveal = target.classList.contains("reveal");
                if (isReveal) {
                    target.style.transition = 'none';
                    target.classList.add("active");
                }

                // Calculate absolute position relative to document
                const rect = target.getBoundingClientRect();
                const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                const targetTop = rect.top + scrollTop;

                // Final offset
                const finalPosition = targetTop - navbarHeight;

                // Restore transition if it was a reveal element
                if (isReveal) {
                    // Trigger reflow
                    target.offsetHeight; 
                    target.style.transition = '';
                }

                window.scrollTo({
                    top: finalPosition,
                    behavior: 'smooth'
                });

                // Update URL hash without jumping
                history.pushState(null, null, href);
            });
        });
    }

    /**
     * Initializes hoverable anchor links for article headings.
     */
    function initAnchorLinks() {
        const headings = document.querySelectorAll("#article-content h2.title, #article-content h3.subtitle");
        
        headings.forEach(heading => {
            if (!heading.id) {
                heading.id = heading.textContent
                    .toLowerCase()
                    .replace(/[^\w\s-]/g, "")
                    .replace(/\s+/g, "-");
            }

            const anchor = document.createElement("a");
            anchor.className = "anchor-link";
            anchor.href = `#${heading.id}`;
            anchor.innerHTML = "#";
            anchor.setAttribute("aria-hidden", "true");
            heading.appendChild(anchor);
        });
    }

    /**
     * Initializes the BibTeX copy-to-clipboard functionality.
     */
    function initBibCopy() {
        const copyBtn = document.getElementById("copyBib");
        const bibCode = document.querySelector(".bibtex-container pre code");

        if (!copyBtn || !bibCode) return;

        copyBtn.addEventListener("click", () => {
            const text = bibCode.innerText;
            navigator.clipboard.writeText(text).then(() => {
                copyBtn.classList.add("copied");
                const icon = copyBtn.querySelector("i");
                const label = copyBtn.querySelector("span");
                
                icon.className = "fas fa-check";
                label.textContent = "Copied!";

                setTimeout(() => {
                    copyBtn.classList.remove("copied");
                    icon.className = "far fa-copy";
                    label.textContent = "Copy";
                }, 2000);
            });
        });
    }

    /**
     * Initializes the minimalist image lightbox.
     */
    function initLightbox() {
        const lightbox = document.getElementById("lightbox");
        const lightboxImg = lightbox.querySelector("img");
        const lightboxCaption = lightbox.querySelector(".lightbox-caption");
        const figures = document.querySelectorAll(".figure img");

        if (!lightbox || !lightboxImg || !lightboxCaption) return;

        figures.forEach(img => {
            img.addEventListener("click", () => {
                // Find the nearest caption
                const figureParent = img.closest(".figure");
                const caption = figureParent ? figureParent.querySelector(".fig-caption") : null;
                
                lightboxImg.src = img.src;
                lightboxCaption.textContent = caption ? caption.textContent : "";
                lightbox.classList.add("active");
                document.body.style.overflow = "hidden"; // Prevent scrolling
            });
        });

        lightbox.addEventListener("click", () => {
            lightbox.classList.remove("active");
            document.body.style.overflow = "";
        });

        // Close on Escape key
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && lightbox.classList.contains("active")) {
                lightbox.classList.remove("active");
                document.body.style.overflow = "";
            }
        });
    }

    /**
     * Initializes Dark/Light mode toggle.
     */
    function initThemeToggle() {
        const toggleBtn = document.getElementById("theme-toggle");
        if (!toggleBtn) return;

        const body = document.body;
        const savedTheme = localStorage.getItem(CONFIG.themeKey);

        // Apply saved theme on load
        if (savedTheme === "dark") {
            body.classList.add("dark-theme");
            body.classList.remove("light-theme");
        } else if (savedTheme === "light") {
            body.classList.add("light-theme");
            body.classList.remove("dark-theme");
        }

        toggleBtn.addEventListener("click", () => {
            const isDark = body.classList.contains("dark-theme");
            const isSystemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

            if (body.classList.contains("light-theme")) {
                // Currently forced light -> switch to dark
                body.classList.remove("light-theme");
                body.classList.add("dark-theme");
                localStorage.setItem(CONFIG.themeKey, "dark");
            } else if (body.classList.contains("dark-theme")) {
                // Currently forced dark -> switch to light
                body.classList.remove("dark-theme");
                body.classList.add("light-theme");
                localStorage.setItem(CONFIG.themeKey, "light");
            } else {
                // Currently following system -> toggle relative to system
                if (isSystemDark) {
                    body.classList.add("light-theme");
                    localStorage.setItem(CONFIG.themeKey, "light");
                } else {
                    body.classList.add("dark-theme");
                    localStorage.setItem(CONFIG.themeKey, "dark");
                }
            }
        });
    }

    // Initialize all modules on DOM ready
    document.addEventListener("DOMContentLoaded", () => {
        initThemeToggle(); // Run this early to avoid flash
        initGDPR();
        initScrollReveal();
        initVideoSync();
        initSmoothScroll();
        initAnchorLinks();
        initBibCopy();
        initLightbox();
    });

})();
