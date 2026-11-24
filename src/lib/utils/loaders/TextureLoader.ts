import Renderer from '../../renderer/render/Renderer';
import WebGPUPreLoader from '../../renderer/core/WebGPUPreLoader';
import IPreloadable from '../../renderer/core/IPreloadable';
import Texture2D from "webGPU-framework/renderer/texture/Texture2D";
import {defaultTextureOptions, TextureOptions} from "webGPU-framework/renderer/texture/Texture";

/*
in webGPU, textures cannot be resized and when loading an image, the size of the texture is not known until the image is loaded.
so the 1x1 texture it starts with is replaced once the image is loaded. A material that got this texture through
setTexture follows that by itself (see ShaderBase.setTexture), one that got its view keeps the 1x1 texture.
 */
export default class TextureLoader extends Texture2D implements IPreloadable {
  private url: string;

  constructor(renderer: Renderer, preLoader: WebGPUPreLoader, url: string, options: Partial<TextureOptions>={}) {
    //added to the usage that is passed: setImage and the mipmaps need these
    const usage = (options.usage ?? defaultTextureOptions.usage) | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT;
    super(renderer, 1, 1, {...options, usage});
    this.url = url;
    preLoader.add(this);
  }

  public load(callback: () => void): void {
    this.loadURL(this.url).then(() => {
      callback();
    });
  }

  public async loadURL(url: string | { default: string }) {
    if (typeof url !== 'string') {
      url = <string>url.default;
    }

    const response = await fetch(url);
    const imageBitmap = await createImageBitmap(await response.blob(), {imageOrientation: 'flipY'});

    //setImage regenerates the mips when useMips is set
    this.setImage(imageBitmap);
    //copyExternalImageToTexture copied it, the decoded image is not needed anymore
    imageBitmap.close();
  }

}
