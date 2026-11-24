import Vector3 from './Vector3';
import type Matrix4x4 from './Matrix4x4';

export default class AABox {
  public min: Vector3;
  public max: Vector3;

  constructor(min: Vector3 = new Vector3(1e38, 1e38, 1e38), max: Vector3 = new Vector3(-1e38, -1e38, -1e38)) {
    this.min = min;
    this.max = max;
  }

  public get area(): number {
    const dim = Vector3.subtract(this.max, this.min);
    return 2 * (dim.x * dim.y + dim.y * dim.z + dim.z * dim.x);
  }

  public get center(): Vector3 {
    return Vector3.add(this.max, this.min).multiplyScalar(0.5);
  }

  public get extents(): Vector3 {
    return Vector3.subtract(this.max, this.min).multiplyScalar(0.5);
  }

  public static merge(boxes: AABox[]): AABox {
    const ret = new AABox();
    for (const box of boxes) ret.expandWithAABox(box);
    return ret;
  }

  // writes into min and max, like copy, so references to them stay valid
  public setCenterExtents(c: Vector3, e: Vector3) {
    this.min.copy(c).subtract(e);
    this.max.copy(c).add(e);
  }

  public expandWithPoint(p: Vector3) {
    this.max.max(p);
    this.min.min(p);
  }

  public expandWithAABox(b: AABox) {
    this.max.max(b.max);
    this.min.min(b.min);
  }

  public copy(p: AABox) {
    this.min.copy(p.min);
    this.max.copy(p.max);
  }

  // the box around the 8 transformed corners: with a rotation, min and max alone don't stay the extremes
  public transform(worldMatrix: Matrix4x4) {
    const min = this.min.clone();
    const max = this.max.clone();
    this.min.setValues(Infinity, Infinity, Infinity);
    this.max.setValues(-Infinity, -Infinity, -Infinity);
    const corner = new Vector3();
    for (let i = 0; i < 8; i++) {
      corner.setValues(i & 1 ? max.x : min.x, i & 2 ? max.y : min.y, i & 4 ? max.z : min.z).transform(worldMatrix);
      this.expandWithPoint(corner);
    }
    return this;
  }

  public clone(): AABox {
    return new AABox(this.min.clone(), this.max.clone());
  }
}
