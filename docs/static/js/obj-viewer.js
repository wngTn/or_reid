

// OBJ viewer: robust checks + helpful inline error messages
(function () {
  function warn(container, msg) {
    const box = document.createElement('div');
    box.textContent = msg;
    box.style.cssText = 'margin:.5rem 0;color:#b91c1c;background:#fff0f0;border:1px solid #fecaca;border-radius:8px;padding:.5rem .75rem;font-size:.9rem;';
    container.appendChild(box);
  }

  function fitCameraToObject(camera, object, controls, offset = 1.25) {
    const box = new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1.0;
    const fov = camera.fov * (Math.PI / 180);
    const dist = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * offset;
    camera.position.set(center.x + dist, center.y + dist * 0.25, center.z + dist);
    camera.near = dist / 100;
    camera.far = dist * 100;
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();
  }

  function createViewer(container, src) {
    if (typeof THREE === 'undefined') {
      warn(container, 'Three.js failed to load (global THREE missing). Make sure the CDN script is reachable.');
      return;
    }
    if (typeof THREE.OrbitControls === 'undefined') {
      warn(container, 'OrbitControls not available. Ensure examples/js/controls/OrbitControls.js is included after three.min.js.');
      return;
    }
    if (typeof THREE.OBJLoader === 'undefined') {
      warn(container, 'OBJLoader not available. Ensure examples/js/loaders/OBJLoader.js is included after three.min.js.');
      return;
    }

    const width = container.clientWidth || 640;
    const height = Math.max(container.clientHeight, Math.floor(width * 0.6));

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(2.5, 1.5, 3.5);
    scene.add(camera);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.2;
    controls.maxDistance = 100;
    controls.enablePan = true;

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const dir = new THREE.DirectionalLight(0xffffff, 0.65);
    dir.position.set(5, 10, 7);
    scene.add(dir);

    // Subtle grid
    const grid = new THREE.GridHelper(10, 10);
    grid.material.opacity = 0.15;
    grid.material.transparent = true;
    scene.add(grid);

    const loader = new THREE.OBJLoader();
    loader.load(
      src,
      (obj) => {
        obj.traverse((c) => {
          if (c.isMesh) {
            if (!c.material) c.material = new THREE.MeshStandardMaterial({ color: 0xcccccc, metalness: 0.05, roughness: 0.7 });
            c.castShadow = true;
            c.receiveShadow = true;
          }
        });
        scene.add(obj);
        fitCameraToObject(camera, obj, controls, 1.6);
      },
      undefined,
      (err) => {
        console.error('Failed to load OBJ', err);
        warn(container, 'Failed to load model at ' + src + '. If you are viewing this page locally using the file:// protocol, many browsers block AJAX requests. Please run a local server (e.g., `python3 -m http.server`) or host via GitHub Pages.');
      }
    );

    function onResize() {
      const w = container.clientWidth || 640;
      const h = Math.max(container.clientHeight, Math.floor(w * 0.6));
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener('resize', onResize);

    (function animate() {
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    })();
  }

  function initViewers() {
    var containers = document.querySelectorAll('.obj-figure[data-src$=".obj"]');
    containers.forEach(function (c) {
      createViewer(c, c.getAttribute('data-src'));
    });
  }

  // Ensure scripts are loaded and DOM ready
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(initViewers, 0);
  } else {
    document.addEventListener('DOMContentLoaded', initViewers);
  }
})();
