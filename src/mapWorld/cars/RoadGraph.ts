import { ROAD_HEADER_FLOATS } from '../map/worldGeometryCore';

/** A point where road segments meet, in world xz. */
export type RoadNode = {
  x: number;
  z: number;
  // the edges that start or end here
  edges: RoadEdge[];
  // the key of its cell in the snap grid
  snapKey: number;
};

/**
 * A straight road segment from a to b, in world units. A one way edge is only driven from a to b. Speed in world units
 * per second.
 */
export type RoadEdge = {
  a: RoadNode;
  b: RoadNode;
  length: number;
  // the unit direction from a to b
  dx: number;
  dz: number;
  width: number;
  speed: number;
  oneway: boolean;
  // false once its tile is removed: a car on it is taken away
  alive: boolean;
  // the key of its cell in the spawn grid
  cellKey: number;
};

type SpawnCell = { edges: RoadEdge[]; length: number };

// The node snap grid, in meters: a point snaps to a node within the snap radius, which is at most this, so the 3 x 3
// cells around it hold every candidate.
const SNAP_CELL_M = 2;
// Points of the same tile that are the same vertex of the vector tile are the same float: a tiny radius. The ends of a
// line also snap to the cut of the neighbouring tile (clipLineToTile), which is computed there in other coordinates.
const SNAP_INNER_M = 0.05;
const SNAP_END_M = 1.5;
// The spawn grid, in meters: a car spawns on an edge of a random cell, see sampleEdge.
const SPAWN_CELL_M = 100;
// 2^21 cells per axis around the origin: enough for any distance a projection covers.
const GRID_OFFSET = 1 << 20;
const GRID_STRIDE = 1 << 21;

function cellKey(ix: number, iz: number): number {
  return (ix + GRID_OFFSET) * GRID_STRIDE + (iz + GRID_OFFSET);
}

/**
 * The roads the cars drive on: the center lines of the terrain tiles (WorldTileResult.roads) as a graph of straight
 * edges between nodes, joined where lines share a point, also across tiles. Tiles come and go with the streaming; a
 * removed tile takes its edges along, the nodes they leave without edges go too.
 * The edges are also bucketed by area (SPAWN_CELL_M), for the road length around a point and the spawning.
 */
export default class RoadGraph {
  private metersToNorm = 1;
  private snapCell = SNAP_CELL_M;
  private spawnCell = SPAWN_CELL_M;
  private readonly snapGrid = new Map<number, RoadNode[]>();
  private readonly spawnGrid = new Map<number, SpawnCell>();
  private readonly tileEdges = new Map<string, RoadEdge[]>();
  // the most road length of a spawn cell so far, for the rejection sampling of sampleEdge
  private maxCellLength = 0;
  /** Changes whenever edges are added or removed. */
  public version = 0;

  /** The scale of the projection the coordinates of the next tiles are in. */
  public setMetersToNorm(metersToNorm: number): void {
    if (metersToNorm === this.metersToNorm) return;
    this.clear();
    this.metersToNorm = metersToNorm;
    this.snapCell = SNAP_CELL_M * metersToNorm;
    this.spawnCell = SPAWN_CELL_M * metersToNorm;
  }

  public clear(): void {
    for (const edges of this.tileEdges.values()) for (const e of edges) e.alive = false;
    this.snapGrid.clear();
    this.spawnGrid.clear();
    this.tileEdges.clear();
    this.maxCellLength = 0;
    this.version++;
  }

  /** The roads of a tile, see WorldTileResult.roads: map (x, north) coords, world (x, -north). */
  public addTile(key: string, roads: Float32Array): void {
    if (this.tileEdges.has(key)) this.removeTile(key);
    const edges: RoadEdge[] = [];
    const mtn = this.metersToNorm;
    let i = 0;
    while (i + ROAD_HEADER_FLOATS <= roads.length) {
      const count = roads[i];
      const width = roads[i + 1] * mtn;
      const speed = roads[i + 2] * mtn;
      const oneway = roads[i + 3] !== 0;
      const first = i + ROAD_HEADER_FLOATS;
      i = first + count * 2;
      let prev: RoadNode | null = null;
      for (let k = 0; k < count; k++) {
        const end = k === 0 || k === count - 1;
        const node = this.getNode(roads[first + k * 2], -roads[first + k * 2 + 1], (end ? SNAP_END_M : SNAP_INNER_M) * mtn);
        if (prev && node !== prev) {
          const edge = this.addEdge(prev, node, width, speed, oneway);
          if (edge) edges.push(edge);
        }
        prev = node;
      }
    }
    this.tileEdges.set(key, edges);
    this.version++;
  }

  public removeTile(key: string): void {
    const edges = this.tileEdges.get(key);
    if (!edges) return;
    this.tileEdges.delete(key);
    const cells = new Set<number>();
    const nodes = new Set<RoadNode>();
    for (const e of edges) {
      e.alive = false;
      nodes.add(e.a);
      nodes.add(e.b);
      const cell = this.spawnGrid.get(e.cellKey);
      if (cell) {
        cell.length -= e.length;
        cells.add(e.cellKey);
      }
    }
    for (const node of nodes) {
      node.edges = node.edges.filter(e => e.alive);
      if (node.edges.length > 0) continue;
      const list = this.snapGrid.get(node.snapKey);
      if (!list) continue;
      const index = list.indexOf(node);
      if (index >= 0) list.splice(index, 1);
      if (list.length === 0) this.snapGrid.delete(node.snapKey);
    }
    for (const k of cells) {
      const cell = this.spawnGrid.get(k)!;
      cell.edges = cell.edges.filter(e => e.alive);
      if (cell.edges.length === 0) this.spawnGrid.delete(k);
    }
    this.version++;
  }

  /** The road length (world units) of the spawn cells whose center is within radius of x, z. */
  public lengthWithin(x: number, z: number, radius: number): number {
    const c = this.spawnCell;
    const x0 = Math.floor((x - radius) / c), x1 = Math.floor((x + radius) / c);
    const z0 = Math.floor((z - radius) / c), z1 = Math.floor((z + radius) / c);
    const r2 = radius * radius;
    let length = 0;
    for (let ix = x0; ix <= x1; ix++) {
      for (let iz = z0; iz <= z1; iz++) {
        const cell = this.spawnGrid.get(cellKey(ix, iz));
        if (!cell) continue;
        const dx = (ix + .5) * c - x, dz = (iz + .5) * c - z;
        if (dx * dx + dz * dz <= r2) length += cell.length;
      }
    }
    return length;
  }

  /**
   * A random edge within about radius of x, z, picked in proportion to the road length: a random cell, accepted in
   * proportion to its road length, then an edge of it by length. Null when the tries found no road.
   */
  public sampleEdge(x: number, z: number, radius: number, tries = 24): RoadEdge | null {
    if (this.maxCellLength <= 0) return null;
    const c = this.spawnCell;
    for (let t = 0; t < tries; t++) {
      const r = radius * Math.sqrt(Math.random());
      const angle = Math.random() * Math.PI * 2;
      const cell = this.spawnGrid.get(cellKey(Math.floor((x + Math.cos(angle) * r) / c), Math.floor((z + Math.sin(angle) * r) / c)));
      if (!cell || cell.length <= 0 || Math.random() * this.maxCellLength > cell.length) continue;
      let pick = Math.random() * cell.length;
      for (const e of cell.edges) {
        pick -= e.length;
        if (pick <= 0) return e;
      }
      return cell.edges[cell.edges.length - 1] ?? null;
    }
    return null;
  }

  /**
   * The nearest point on a road within radius of x, z (world units) that accept takes, and the direction of its edge,
   * null for none. The edges are found by the spawn cell of their middle: an edge longer than a cell may be missed
   * near its ends.
   */
  public nearestPoint(x: number, z: number, radius: number, accept: (x: number, z: number) => boolean = () => true):
    { x: number; z: number; dx: number; dz: number } | null {
    const c = this.spawnCell;
    const reach = radius + c;
    const x0 = Math.floor((x - reach) / c), x1 = Math.floor((x + reach) / c);
    const z0 = Math.floor((z - reach) / c), z1 = Math.floor((z + reach) / c);
    const points: { x: number; z: number; dx: number; dz: number; d2: number }[] = [];
    for (let ix = x0; ix <= x1; ix++) {
      for (let iz = z0; iz <= z1; iz++) {
        const cell = this.spawnGrid.get(cellKey(ix, iz));
        if (!cell) continue;
        for (const e of cell.edges) {
          const t = Math.min(Math.max((x - e.a.x) * e.dx + (z - e.a.z) * e.dz, 0), e.length);
          const px = e.a.x + e.dx * t, pz = e.a.z + e.dz * t;
          const d2 = (px - x) ** 2 + (pz - z) ** 2;
          if (d2 <= radius * radius) points.push({ x: px, z: pz, dx: e.dx, dz: e.dz, d2 });
        }
      }
    }
    points.sort((a, b) => a.d2 - b.d2);
    return points.find(p => accept(p.x, p.z)) ?? null;
  }

  //the node within radius of x, z, or a new one there
  private getNode(x: number, z: number, radius: number): RoadNode {
    const s = this.snapCell;
    const ix = Math.floor(x / s), iz = Math.floor(z / s);
    let best: RoadNode | null = null;
    let bestD = radius * radius;
    // the cells the radius reaches (at most the 3 x 3 around it): mostly only its own
    const x0 = Math.floor((x - radius) / s), x1 = Math.floor((x + radius) / s);
    const z0 = Math.floor((z - radius) / s), z1 = Math.floor((z + radius) / s);
    for (let cx = x0; cx <= x1; cx++) {
      for (let cz = z0; cz <= z1; cz++) {
        const list = this.snapGrid.get(cellKey(cx, cz));
        if (!list) continue;
        for (const n of list) {
          const d = (n.x - x) * (n.x - x) + (n.z - z) * (n.z - z);
          if (d <= bestD) {
            bestD = d;
            best = n;
          }
        }
      }
    }
    if (best) return best;
    const key = cellKey(ix, iz);
    const node: RoadNode = { x, z, edges: [], snapKey: key };
    let list = this.snapGrid.get(key);
    if (!list) this.snapGrid.set(key, list = []);
    list.push(node);
    return node;
  }

  private addEdge(a: RoadNode, b: RoadNode, width: number, speed: number, oneway: boolean): RoadEdge | null {
    const dx = b.x - a.x, dz = b.z - a.z;
    const length = Math.hypot(dx, dz);
    if (length < 1e-9) return null;
    const c = this.spawnCell;
    const key = cellKey(Math.floor((a.x + b.x) * .5 / c), Math.floor((a.z + b.z) * .5 / c));
    const edge: RoadEdge = { a, b, length, dx: dx / length, dz: dz / length, width, speed, oneway, alive: true, cellKey: key };
    a.edges.push(edge);
    b.edges.push(edge);
    let cell = this.spawnGrid.get(key);
    if (!cell) this.spawnGrid.set(key, cell = { edges: [], length: 0 });
    cell.edges.push(edge);
    cell.length += length;
    this.maxCellLength = Math.max(this.maxCellLength, cell.length);
    return edge;
  }
}
