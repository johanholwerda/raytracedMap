import WebGPUPreLoader from 'webGPU-framework/renderer/core/WebGPUPreLoader';
import UncappedFrames from 'webGPU-framework/utils/UncappedFrames';
import MapWorld from './mapWorld/MapWorld';
import './main.css';

// Always installed, so the profile checkbox next to the fps counter can untie the frames from vsync; ?uncapped starts with it on.
UncappedFrames.install(new URLSearchParams(window.location.search).has('uncapped'));

const containerDiv = document.getElementById('webgl-preloader-container');
const progressDiv = document.getElementById('webgl-preloader-progress');

function handlePreloadProgress(progress: number) {
  if (!containerDiv || !progressDiv) {
    throw new ReferenceError('Unable to find container divs in html');
  }

  containerDiv.style.display = progress < 1 ? 'block' : 'none';
  progressDiv.style.width = ((276 * progress) | 0) + 'px';
}

async function main() {
  handlePreloadProgress(0);

  const preloader = new WebGPUPreLoader();

  const canvasContainer = document.createElement('div');
  canvasContainer.classList.add('webgpu-container');
  document.body.appendChild(canvasContainer);

  const mapWorld = new MapWorld();

  await mapWorld.start(canvasContainer, preloader, true);

  preloader.load(
    () => {
      handlePreloadProgress(1);
      mapWorld.init();
    },
    (progress) => handlePreloadProgress(progress),
  );
}

main();
