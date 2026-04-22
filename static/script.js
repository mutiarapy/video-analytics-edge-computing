document.addEventListener('DOMContentLoaded', () => {
    console.log("Dashboard ISR Lab Ready!");

    // 1. Logika Tombol Login
    const loginBtn = document.querySelector('.btnLogin-popup');
    if (loginBtn) {
        loginBtn.addEventListener('click', () => {
            alert("Fitur Login sedang dikembangkan.");
        });
    }

    // 2. Logika Scroll Halus ke Live Camera
    const liveCameraLink = document.querySelector('a[href="#live-camera"]');
    const target = document.querySelector('#live-camera');

    if (liveCameraLink && target) {
        liveCameraLink.addEventListener('click', function(e) {
            e.preventDefault(); // Mencegah lompatan instan
            
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        });
    }
});