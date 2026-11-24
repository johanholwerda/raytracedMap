import Vector4 from '../../renderer/math/Vector4';
import Vector3 from '../../renderer/math/Vector3';
import Matrix4x4 from '../../renderer/math/Matrix4x4';
import Camera from '../../renderer/camera/Camera';
import AABox from '../../renderer/math/AABox';
import {lerp} from '../../renderer/utils/MathUtils';

export default class Frustum {
  public points: [Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3];
  private planes: [Vector4, Vector4, Vector4, Vector4, Vector4, Vector4];

  constructor() {
    this.planes = Array.from({length: 6}, () => new Vector4()) as [Vector4, Vector4, Vector4, Vector4, Vector4, Vector4];
    this.points = Array.from({length: 8}, () => new Vector3()) as [Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3];
  }

  public setFromCamera(camera: Camera) {
    this.setFromViewProjectionMatrix(camera.viewProjection);

    //create 8 world space frustum corners. Needed for frustum/ frustum intersection
    const fc = camera.projection.frustumCorner;
    const v = camera.view.worldMatrix;
    let i = 0;
    for (let z = 0; z <= 1; z++) {
      for (let y = 0; y <= 1; y++) {
        for (let x = 0; x <= 1; x++) {
          const p = fc.clone();
          p.multiplyScalar(lerp(camera.projection.near, camera.projection.far, z));
          p.x *= x * 2 - 1;
          p.y *= y * 2 - 1;
          p.z *= -1;
          this.points[i++] = Vector3.transform(p, v);
        }
      }
    }
    return this;
  }

  public setFromViewProjectionMatrix(viewProjection: Matrix4x4) {
    const vp = viewProjection.m;

    // Right clipping plane.
    this.planes[0].setValues(vp[3] - vp[0], vp[7] - vp[4], vp[11] - vp[8], vp[15] - vp[12]);
    // Left clipping plane.
    this.planes[1].setValues(vp[3] + vp[0], vp[7] + vp[4], vp[11] + vp[8], vp[15] + vp[12]);
    // Bottom clipping plane.
    this.planes[2].setValues(vp[3] + vp[1], vp[7] + vp[5], vp[11] + vp[9], vp[15] + vp[13]);
    // Top clipping plane.
    this.planes[3].setValues(vp[3] - vp[1], vp[7] - vp[5], vp[11] - vp[9], vp[15] - vp[13]);
    // Far clipping plane.
    this.planes[4].setValues(vp[3] - vp[2], vp[7] - vp[6], vp[11] - vp[10], vp[15] - vp[14]);
    // Near clipping plane. WebGPU clip space has z in [0, w], so this is z >= 0 (OpenGL's [-w, w] would be row 3 + row 2)
    this.planes[5].setValues(vp[2], vp[6], vp[10], vp[14]);

    this.planes.forEach((plane) => {
      plane.multiplyScalar(1 / Math.sqrt(plane.x ** 2 + plane.y ** 2 + plane.z ** 2));
    });
    return this;
  }

  //https://www.yosoygames.com.ar/wp/2016/12/frustum-vs-pyramid-intersection-also-frustum-vs-frustum/
  public frustumInFrustum(frustum: Frustum): boolean {
    return this.planes.every(plane =>
      frustum.points.some(point => this.dot3(plane, point) > 0),
    ) && frustum.planes.every(plane =>
      this.points.some(point => this.dot3(plane, point) > 0),
    );
  }

  public pointInFrustum(p: Vector3): boolean {
    return this.sphereInFrustum(p, 0);
  }

  public sphereInFrustum(p: Vector3, radius: number): boolean {
    return !this.planes.some(plane => this.dot(plane, p.x, p.y, p.z) < -radius);
  }

  //outside when all corners are behind one of the planes. per plane that is the corner furthest along its normal
  //(the one with the largest dot product): when that one is behind, the others are too
  public boxInFrustum(min: Vector3, max: Vector3): boolean {
    for (let i = 0; i < 6; i++) {
      const plane = this.planes[i];
      const x = plane.x >= 0 ? max.x : min.x;
      const y = plane.y >= 0 ? max.y : min.y;
      const z = plane.z >= 0 ? max.z : min.z;
      if (this.dot(plane, x, y, z) < 0.0) return false;
    }
    return true;
  }

  //seems to have lots of false positives
  public transformedBoxInFrustum(min: Vector3, max: Vector3, worldMatrix: Matrix4x4): boolean {
    // Transform the 8 corners of the box
    const corners = [
      new Vector3(min.x, min.y, min.z),
      new Vector3(max.x, min.y, min.z),
      new Vector3(min.x, max.y, min.z),
      new Vector3(max.x, max.y, min.z),
      new Vector3(min.x, min.y, max.z),
      new Vector3(max.x, min.y, max.z),
      new Vector3(min.x, max.y, max.z),
      new Vector3(max.x, max.y, max.z)
    ].map(corner => Vector3.transform(corner, worldMatrix));

    // Check if any corner is inside the frustum
    return this.planes.every(plane =>
      corners.some(corner => this.dot3(plane, corner) >= 0)
    );
  }

  public aaBoxInFrustum(box: AABox): boolean {
    return this.boxInFrustum(box.min, box.max);
  }

  //only tests for the 4 corners so it does not cover all cases
  public quadInFrustum(worldMatrix: Matrix4x4): boolean {
    for (let x = -1; x <= 1; x += 2) {
      for (let y = -1; y <= 1; y += 2) {
        const p = Vector3.transform(new Vector3(x, y, 0), worldMatrix);
        if (this.pointInFrustum(p)) return true;
      }
    }
    return false;
  }

  private dot(plane: Vector4, x: number, y: number, z: number) {
    return plane.x * x + plane.y * y + plane.z * z + plane.w;
  }

  private dot3(plane: Vector4, p: Vector3) {
    return plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w;
  }
}
