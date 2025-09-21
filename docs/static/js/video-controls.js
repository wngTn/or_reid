// docs/static/js/video-controls.js

document.addEventListener("DOMContentLoaded", () => {
    // Get the video elements
    const video1 = document.getElementById("team1");
    const video2 = document.getElementById("team2");

    // Get the control buttons
    const playBtn = document.getElementById("playBoth");
    const pauseBtn = document.getElementById("pauseBoth");
    const restartBtn = document.getElementById("restartBoth");

    // Check if all elements exist before adding listeners
    if (video1 && video2 && playBtn && pauseBtn && restartBtn) {

        // --- Add event listeners to the buttons ---

        // Play button
        playBtn.addEventListener("click", () => {
            video1.play();
            video2.play();
        });

        // Pause button
        pauseBtn.addEventListener("click", () => {
            video1.pause();
            video2.pause();
        });

        // Restart button
        restartBtn.addEventListener("click", () => {
            // Set current time to the beginning
            video1.currentTime = 0;
            video2.currentTime = 0;
            // Start playing again
            video1.play();
            video2.play();
        });
    }
});