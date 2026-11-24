export function isTouchDevice() {
  // @ts-ignore
  return typeof window !== "undefined" && ('ontouchstart' in window || navigator.maxTouchPoints > 1 || navigator.msMaxTouchPoints > 1);
}

export function isMobile() {
  return isTouchDevice() || /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

export function isIOS() {
  return typeof navigator !== "undefined" && /iPad|iPhone|iPod/.test(navigator.userAgent);
}

export function isAndroid() {
  return typeof navigator !== "undefined" && /Android/i.test(navigator.userAgent);
}

export function isChrome() {
  return typeof navigator !== "undefined" && /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor);
}

export function isIOSSafari() {
  const ua = window.navigator.userAgent;
  const iOS = !!ua.match(/iPad/i) || !!ua.match(/iPhone/i);
  const webkit = !!ua.match(/WebKit/i);
  return iOS && webkit && !ua.match(/CriOS/i);
}

function checkImageSupport(base64: string) {
  return new Promise<boolean>((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      resolve(img.width > 0 && img.height > 0);
    };
    img.onerror = () => {
      resolve(false);
    };
    try {
      img.src = base64;
    } catch (e) {
      resolve(false);
    }
  });
}

export function browserHasWebpSupport() {
  return checkImageSupport('data:image/webp;base64,UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEADsD+JaQAA3AAAAAA');
}

export function browserHasAvifSupport() {
  return checkImageSupport('data:image/avif;base64,AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUEAAADybWV0YQAAAAAAAAAoaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAGxpYmF2aWYAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABGgAAABMAAAAoaWluZgAAAAAAAQAAABppbmZlAgAAAAABAABhdjAxQ29sb3IAAAAAamlwcnAAAABLaXBjbwAAABRpc3BlAAAAAAAAAAEAAAABAAAAEHBpeGkAAAAAAwgICAAAAAxhdjFDgS0AAAAAABNjb2xybmNseAACAAIAAoAAAAAXaXBtYQAAAAAAAAABAAEEAQKDBAAAABttZGF0EgAKBDgABokyCRAAAAAP+I9ngw==');
}