//a virtual joystick for touch screens, in the bottom left corner. value: x to the right, y forward (up on the screen),
//within the unit circle. Its own element over the canvas, so a drag on it doesn't rotate the camera
export default class Joystick {
  public readonly value = {x: 0, y: 0};
  private readonly base: HTMLDivElement;
  private readonly knob: HTMLDivElement;
  //in css pixels
  private readonly radius = 60;
  private pointerId: number | null = null;
  private centerX = 0;
  private centerY = 0;

  //className: placed by that class (position, left, bottom...) instead of in the bottom left corner
  constructor(parent: HTMLElement = document.body, className?: string) {
    const size = this.radius * 2;
    this.base = document.createElement('div');
    if (className) this.base.className = className;
    else Object.assign(this.base.style, { position: 'fixed', left: '24px', bottom: '24px', zIndex: '10' });
    Object.assign(this.base.style, {
      width: `${size}px`,
      height: `${size}px`,
      borderRadius: '50%',
      background: 'rgba(0, 0, 0, .2)',
      border: '2px solid rgba(255, 255, 255, .5)',
      boxSizing: 'border-box',
      touchAction: 'none',
      userSelect: 'none',
      webkitUserSelect: 'none',
    });
    this.knob = document.createElement('div');
    const knobSize = size * .45;
    Object.assign(this.knob.style, {
      position: 'absolute',
      left: `${(size - knobSize) / 2 - 2}px`,
      top: `${(size - knobSize) / 2 - 2}px`,
      width: `${knobSize}px`,
      height: `${knobSize}px`,
      borderRadius: '50%',
      background: 'rgba(255, 255, 255, .7)',
      pointerEvents: 'none',
    });
    this.base.appendChild(this.knob);
    parent.appendChild(this.base);

    this.base.addEventListener('pointerdown', this.onDown);
    this.base.addEventListener('pointermove', this.onMove);
    this.base.addEventListener('pointerup', this.onUp);
    this.base.addEventListener('pointercancel', this.onUp);
    this.base.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  public get element(): HTMLDivElement {
    return this.base;
  }

  //touch screens only, a mouse has WASD
  public static get wanted(): boolean {
    return window.matchMedia('(pointer: coarse)').matches || navigator.maxTouchPoints > 0;
  }

  private onDown = (e: PointerEvent) => {
    if (this.pointerId !== null) return;
    e.preventDefault();
    e.stopPropagation();
    this.pointerId = e.pointerId;
    this.base.setPointerCapture(e.pointerId);
    const bounds = this.base.getBoundingClientRect();
    this.centerX = bounds.left + bounds.width / 2;
    this.centerY = bounds.top + bounds.height / 2;
    this.setFromPointer(e);
  };

  private onMove = (e: PointerEvent) => {
    if (e.pointerId !== this.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    this.setFromPointer(e);
  };

  private onUp = (e: PointerEvent) => {
    if (e.pointerId !== this.pointerId) return;
    e.stopPropagation();
    this.pointerId = null;
    this.value.x = 0;
    this.value.y = 0;
    this.knob.style.transform = '';
  };

  private setFromPointer(e: PointerEvent) {
    let dx = (e.clientX - this.centerX) / this.radius;
    let dy = (e.clientY - this.centerY) / this.radius;
    const l = Math.hypot(dx, dy);
    if (l > 1) {
      dx /= l;
      dy /= l;
    }
    this.value.x = dx;
    this.value.y = -dy;
    this.knob.style.transform = `translate(${dx * this.radius}px, ${dy * this.radius}px)`;
  }

  public destruct() {
    this.base.remove();
  }
}
