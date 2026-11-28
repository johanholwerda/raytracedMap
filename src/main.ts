import WebGPUPreLoader from 'webGPU-framework/renderer/core/WebGPUPreLoader';
import UncappedFrames from 'webGPU-framework/utils/UncappedFrames';
import MapWorld from './mapWorld/MapWorld';
import './main.css';
import { installRemoteDebug } from './mapWorld/debug/RemoteDebug';

// ?remote on the dev server: console, errors and timings to tools/remote.mjs (before anything asks for the GPU)
installRemoteDebug();

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

  if (!(await mapWorld.start(canvasContainer, preloader, true))) {
    // no WebGPU: hide the loading bar so it doesn't cover the Renderer's message
    handlePreloadProgress(1);
    return;
  }

  preloader.load(
    () => {
      handlePreloadProgress(1);
      mapWorld.init();
    },
    (progress) => handlePreloadProgress(progress),
  );
}

main();
