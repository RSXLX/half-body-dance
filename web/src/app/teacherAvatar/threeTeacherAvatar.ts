import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { getVisibility } from '../../core/geometry.js';
import { getPoseReference } from '../../core/reference.js';
import type { HandFrame, NormalizedLandmark } from '../../core/types.js';
import type { StageRect } from '../stageRenderer.js';
import { buildTeacherAvatarPalette } from './avatarPalette.js';
import {
  TEACHER_BODY_PARTS,
  type TeacherBodyPartId,
} from './poseParts.js';
import type { TeacherAvatarPalette, TeacherBasePalette } from './types.js';

export const DEFAULT_TEACHER_BJD_MODEL_URL = '/models/teacher-bjd.glb';
export const GLTF_BJD_PART_NAME_CANDIDATES = {
  upperArm: ['upperArmMesh004', 'upperArmMesh001', 'upperArmMesh003', 'upperArmMesh'],
  foreArm: ['foreArmMesh002', 'foreArmMesh001', 'foreArmMesh'],
  thigh: ['tightMesh002', 'tightMesh001', 'tightMesh'],
  calf: ['calfMesh002', 'calfMesh001', 'calfMesh'],
  shoulder: ['shoulderMesh003', 'shoulderMesh002', 'shoulderMesh005'],
  elbow: ['elbowMesh002', 'elbowMesh001', 'elbowMesh'],
  knee: ['kneeMesh002', 'kneeMesh001', 'kneeMesh'],
  wrist: ['forearmKnotMesh001', 'forearmKnotMesh'],
  foot: ['footMesh002', 'footMesh001', 'footMesh'],
  chest: ['chestMesh002', 'chestMesh001', 'chestMesh'],
  waist: ['waistMesh002', 'waistMesh001', 'waistMesh'],
  hip: ['hipMesh002', 'hipMesh001', 'hipMesh'],
  head: ['headMesh002', 'headMesh001', 'headMesh'],
} as const;

const WORLD_HEIGHT = 4.6;
const HAND_CONNECTIONS: ReadonlyArray<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const GLTF_LIMB_PROPORTIONS: Record<TeacherBodyPartId, {
  shoulderRatio: number;
  lengthRatio: number;
  scaleRatio: number;
  min: number;
  depthRatio: number;
}> = {
  leftUpperArm: { shoulderRatio: 0.2, lengthRatio: 0.38, scaleRatio: 0.12, min: 0.045, depthRatio: 0.84 },
  rightUpperArm: { shoulderRatio: 0.2, lengthRatio: 0.38, scaleRatio: 0.12, min: 0.045, depthRatio: 0.84 },
  leftForearm: { shoulderRatio: 0.16, lengthRatio: 0.4, scaleRatio: 0.095, min: 0.038, depthRatio: 0.78 },
  rightForearm: { shoulderRatio: 0.16, lengthRatio: 0.4, scaleRatio: 0.095, min: 0.038, depthRatio: 0.78 },
  leftThigh: { shoulderRatio: 0.24, lengthRatio: 0.42, scaleRatio: 0.14, min: 0.055, depthRatio: 0.9 },
  rightThigh: { shoulderRatio: 0.24, lengthRatio: 0.42, scaleRatio: 0.14, min: 0.055, depthRatio: 0.9 },
  leftCalf: { shoulderRatio: 0.2, lengthRatio: 0.34, scaleRatio: 0.11, min: 0.045, depthRatio: 0.84 },
  rightCalf: { shoulderRatio: 0.2, lengthRatio: 0.34, scaleRatio: 0.11, min: 0.045, depthRatio: 0.84 },
};

interface ThreeTeacherAvatarRenderInput {
  poseLandmarks: readonly (NormalizedLandmark | null | undefined)[];
  hands?: readonly HandFrame[];
  stageRect: StageRect;
  palette: TeacherBasePalette;
  mirrorX?: boolean;
  highDetail?: boolean;
  showHead?: boolean;
}

interface CapsuleNode {
  group: THREE.Group;
  shaft: THREE.Mesh<THREE.CylinderGeometry, THREE.MeshStandardMaterial>;
  startCap: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
  endCap: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
}

interface JointNode {
  group: THREE.Group;
  orb: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
  seam: THREE.Mesh<THREE.TorusGeometry, THREE.MeshStandardMaterial>;
}

interface GltfPartInstance {
  group: THREE.Group;
  size: THREE.Vector3;
}

interface GltfBjdRig {
  limbs: Partial<Record<TeacherBodyPartId, GltfPartInstance>>;
  joints: Partial<Record<number, GltfPartInstance>>;
  torso: {
    chest?: GltfPartInstance;
    waist?: GltfPartInstance;
    hip?: GltfPartInstance;
    head?: GltfPartInstance;
  };
}

interface ParsedCssColor {
  color: THREE.Color;
  alpha: number;
}

function parseCssColor(style: string | undefined, fallback: string): ParsedCssColor {
  const source = style || fallback;
  const rgb = source.match(/rgba?\(([^)]+)\)/i);
  if (rgb) {
    const parts = rgb[1]!
      .split(/[\s,\/]+/)
      .map((part) => part.trim())
      .filter(Boolean);
    const r = Number(parts[0]) || 0;
    const g = Number(parts[1]) || 0;
    const b = Number(parts[2]) || 0;
    const alpha = parts[3] == null ? 1 : Math.max(0, Math.min(1, Number(parts[3])));
    return {
      color: new THREE.Color(r / 255, g / 255, b / 255),
      alpha: Number.isFinite(alpha) ? alpha : 1,
    };
  }
  return { color: new THREE.Color(source), alpha: 1 };
}

function normalizedToWorld(
  point: NormalizedLandmark | null | undefined,
  aspect: number,
  mirrorX: boolean,
): THREE.Vector3 | null {
  if (!point) return null;
  const x = (point.x - 0.5) * WORLD_HEIGHT * aspect;
  const z = -(point.z ?? 0) * WORLD_HEIGHT * 0.65;
  return new THREE.Vector3(mirrorX ? -x : x, (0.5 - point.y) * WORLD_HEIGHT, z);
}

function setObjectVisible(object: THREE.Object3D, visible: boolean): void {
  if (object.visible !== visible) object.visible = visible;
}

function applySegmentTransform(
  object: THREE.Object3D,
  start: THREE.Vector3,
  end: THREE.Vector3,
): number {
  const delta = new THREE.Vector3().subVectors(end, start);
  const length = delta.length();
  if (!length) {
    object.visible = false;
    return 0;
  }
  object.visible = true;
  object.position.copy(start).add(end).multiplyScalar(0.5);
  object.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
  return length;
}

function setCapsule(
  node: CapsuleNode,
  start: THREE.Vector3,
  end: THREE.Vector3,
  startRadius: number,
  endRadius: number,
): void {
  const length = applySegmentTransform(node.group, start, end);
  if (!length) return;
  const radius = (startRadius + endRadius) * 0.5;
  node.shaft.scale.set(radius, length, radius);
  node.startCap.position.set(0, -length / 2, 0);
  node.endCap.position.set(0, length / 2, 0);
  node.startCap.scale.setScalar(startRadius);
  node.endCap.scale.setScalar(endRadius);
}

function setJoint(node: JointNode, position: THREE.Vector3, radius: number, showSeam: boolean): void {
  node.group.visible = true;
  node.group.position.copy(position);
  node.orb.scale.setScalar(radius);
  node.seam.visible = showSeam;
  node.seam.scale.setScalar(radius * 0.7);
}

function bodyScale(points: readonly (NormalizedLandmark | null | undefined)[]): number {
  const reference = getPoseReference(points);
  return (reference?.scale || 0.22) * WORLD_HEIGHT;
}

function hasFinitePosePoint(point: NormalizedLandmark | null | undefined): point is NormalizedLandmark {
  return !!point && Number.isFinite(point.x) && Number.isFinite(point.y);
}

function namedBone(name: string, patterns: readonly RegExp[]): boolean {
  return patterns.some((pattern) => pattern.test(name));
}

function disposeObjectTree(object: THREE.Object3D): void {
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.geometry.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) material.dispose();
  });
}

export class ThreeTeacherAvatarRenderer {
  readonly canvas: HTMLCanvasElement;
  readonly modelReady: Promise<void>;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
  private readonly root = new THREE.Group();
  private readonly proceduralRoot = new THREE.Group();
  private readonly gltfRigRoot = new THREE.Group();
  private readonly resinMaterial = new THREE.MeshStandardMaterial({ roughness: 0.64, metalness: 0.02, transparent: true });
  private readonly jointMaterial = new THREE.MeshStandardMaterial({ roughness: 0.72, metalness: 0.02, transparent: true });
  private readonly seamMaterial = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0.0, transparent: true });
  private readonly handMaterial = new THREE.LineBasicMaterial({ linewidth: 2, transparent: true });
  private readonly gltfPartMaterials: THREE.MeshStandardMaterial[] = [];
  private readonly limbNodes: Record<TeacherBodyPartId, CapsuleNode>;
  private readonly jointNodes: Record<number, JointNode>;
  private readonly torsoNodes: {
    chest: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
    abdomen: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
    pelvis: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
    head: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
  };
  private readonly handLines: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>[] = [];
  private readonly keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
  private gltfRoot: THREE.Object3D | null = null;
  private gltfBones: Partial<Record<TeacherBodyPartId, THREE.Object3D>> = {};
  private gltfRig: GltfBjdRig | null = null;
  private gltfCenter = new THREE.Vector3();
  private gltfSize = new THREE.Vector3(1, 1, 1);
  private gltfState: 'idle' | 'loading' | 'ready' | 'failed' = 'idle';
  private resolveModelReady: () => void = () => {};
  private disposed = false;

  constructor(args: { canvas: HTMLCanvasElement; modelUrl?: string }) {
    this.canvas = args.canvas;
    this.modelReady = new Promise((resolve) => {
      this.resolveModelReady = resolve;
    });
    this.renderer = new THREE.WebGLRenderer({
      canvas: args.canvas,
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.camera.position.set(0, 0, 8);
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.55));
    this.keyLight.position.set(2.5, 3.5, 5);
    this.scene.add(this.keyLight);
    this.scene.add(this.root);
    this.root.add(this.proceduralRoot);
    this.root.add(this.gltfRigRoot);
    this.gltfRigRoot.visible = false;

    this.limbNodes = this.createLimbs();
    this.jointNodes = this.createJoints();
    this.torsoNodes = this.createTorso();
    this.startModelLoad(args.modelUrl ?? DEFAULT_TEACHER_BJD_MODEL_URL);
  }

  clear(): void {
    this.renderer.clear();
  }

  dispose(): void {
    this.disposed = true;
    this.renderer.dispose();
    this.resinMaterial.dispose();
    this.jointMaterial.dispose();
    this.seamMaterial.dispose();
    this.handMaterial.dispose();
    for (const material of this.gltfPartMaterials) material.dispose();
    for (const node of Object.values(this.limbNodes)) {
      node.shaft.geometry.dispose();
      node.startCap.geometry.dispose();
      node.endCap.geometry.dispose();
    }
    for (const node of Object.values(this.jointNodes)) {
      node.orb.geometry.dispose();
      node.seam.geometry.dispose();
    }
    for (const node of Object.values(this.torsoNodes)) node.geometry.dispose();
    for (const line of this.handLines) line.geometry.dispose();
    for (const child of this.gltfRigRoot.children) disposeObjectTree(child);
  }

  render(input: ThreeTeacherAvatarRenderInput): boolean {
    if (this.disposed || input.poseLandmarks.length < 29) return false;
    const rectAspect = input.stageRect.width > 0 && input.stageRect.height > 0
      ? input.stageRect.width / input.stageRect.height
      : 9 / 16;
    this.resize(rectAspect);
    const palette = buildTeacherAvatarPalette(input.palette);
    this.applyPalette(palette);
    this.renderProcedural(input, palette, rectAspect);
    const renderedModel = this.renderLoadedModel(input, rectAspect);
    this.setProceduralBodyVisible(!renderedModel);
    this.canvas.dataset.avatar = renderedModel ? 'gltf-bjd-rig' : 'procedural-bjd';
    this.renderer.render(this.scene, this.camera);
    return true;
  }

  private resize(contentAspect: number): void {
    const width = Math.max(1, this.canvas.clientWidth || this.canvas.width || 1);
    const height = Math.max(1, this.canvas.clientHeight || this.canvas.height || 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    const aspect = width / height || contentAspect;
    this.camera.left = -WORLD_HEIGHT * aspect * 0.5;
    this.camera.right = WORLD_HEIGHT * aspect * 0.5;
    this.camera.top = WORLD_HEIGHT * 0.5;
    this.camera.bottom = -WORLD_HEIGHT * 0.5;
    this.camera.updateProjectionMatrix();
  }

  private createLimbs(): Record<TeacherBodyPartId, CapsuleNode> {
    const nodes = {} as Record<TeacherBodyPartId, CapsuleNode>;
    for (const part of TEACHER_BODY_PARTS) {
      const group = new THREE.Group();
      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 24, 1), this.resinMaterial);
      const startCap = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), this.resinMaterial);
      const endCap = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), this.resinMaterial);
      group.add(shaft, startCap, endCap);
      this.proceduralRoot.add(group);
      nodes[part.id] = { group, shaft, startCap, endCap };
    }
    return nodes;
  }

  private createJoints(): Record<number, JointNode> {
    const nodes = {} as Record<number, JointNode>;
    for (const index of [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
      const group = new THREE.Group();
      const orb = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 18), this.jointMaterial);
      const seam = new THREE.Mesh(new THREE.TorusGeometry(1, 0.035, 8, 36), this.seamMaterial);
      seam.rotation.x = Math.PI * 0.5;
      group.add(orb, seam);
      this.proceduralRoot.add(group);
      nodes[index] = { group, orb, seam };
    }
    return nodes;
  }

  private createTorso(): ThreeTeacherAvatarRenderer['torsoNodes'] {
    const make = () => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 18), this.resinMaterial);
      this.proceduralRoot.add(mesh);
      return mesh;
    };
    return {
      chest: make(),
      abdomen: make(),
      pelvis: make(),
      head: make(),
    };
  }

  private applyPalette(palette: TeacherAvatarPalette): void {
    const resin = parseCssColor(palette.resinFill, 'rgb(243, 234, 220)');
    const joint = parseCssColor(palette.resinShadow, 'rgb(216, 195, 168)');
    const seam = parseCssColor(palette.jointStroke, 'rgb(90, 58, 42)');
    const teacher = parseCssColor(palette.teacherStroke, 'rgb(84, 243, 168)');
    this.resinMaterial.color.copy(resin.color);
    this.resinMaterial.opacity = Math.max(0.58, resin.alpha);
    this.jointMaterial.color.copy(joint.color);
    this.jointMaterial.opacity = Math.max(0.74, joint.alpha);
    this.seamMaterial.color.copy(seam.color);
    this.seamMaterial.opacity = Math.max(0.42, seam.alpha);
    this.handMaterial.color.copy(teacher.color);
    this.handMaterial.opacity = 0.86;
    this.keyLight.color.copy(teacher.color).lerp(new THREE.Color(0xffffff), 0.68);
    for (const material of this.gltfPartMaterials) {
      if (material.userData.role === 'outline') {
        material.color.copy(teacher.color).lerp(new THREE.Color(0xffffff), 0.18);
        material.emissive.copy(teacher.color);
        material.emissiveIntensity = 0.18;
        material.opacity = 0.92;
      } else {
        material.color.copy(resin.color).lerp(new THREE.Color(0xffffff), 0.08);
        material.emissive.set(0x000000);
        material.emissiveIntensity = 0;
        material.opacity = 1;
      }
    }
  }

  private setProceduralBodyVisible(visible: boolean): void {
    for (const node of Object.values(this.limbNodes)) node.group.visible = visible && node.group.visible;
    for (const node of Object.values(this.jointNodes)) node.group.visible = visible && node.group.visible;
    for (const node of Object.values(this.torsoNodes)) node.visible = visible && node.visible;
  }

  private renderProcedural(
    input: ThreeTeacherAvatarRenderInput,
    palette: TeacherAvatarPalette,
    aspect: number,
  ): void {
    this.proceduralRoot.visible = true;
    const points = input.poseLandmarks;
    const mirrorX = Boolean(input.mirrorX);
    const scale = bodyScale(points);
    const radius = (ratio: number, min: number) => Math.max(min, scale * ratio);

    for (const part of TEACHER_BODY_PARTS) {
      const from = points[part.from];
      const to = points[part.to];
      const node = this.limbNodes[part.id];
      if (!node || !from || !to || getVisibility(from) < 0.25 || getVisibility(to) < 0.25) {
        if (node) setObjectVisible(node.group, false);
        continue;
      }
      const start = normalizedToWorld(from, aspect, mirrorX);
      const end = normalizedToWorld(to, aspect, mirrorX);
      if (!start || !end) {
        setObjectVisible(node.group, false);
        continue;
      }
      setCapsule(node, start, end, radius(part.startRatio, 0.035), radius(part.endRatio, 0.028));
    }

    this.renderTorso(points, aspect, mirrorX, scale, input.showHead ?? false);
    this.renderJoints(points, aspect, mirrorX, scale, input.highDetail ?? true);
    this.renderHands(input.hands ?? [], aspect, mirrorX, palette);
  }

  private renderTorso(
    points: readonly (NormalizedLandmark | null | undefined)[],
    aspect: number,
    mirrorX: boolean,
    scale: number,
    showHead: boolean,
  ): void {
    const ls = normalizedToWorld(points[11], aspect, mirrorX);
    const rs = normalizedToWorld(points[12], aspect, mirrorX);
    const lh = normalizedToWorld(points[23], aspect, mirrorX);
    const rh = normalizedToWorld(points[24], aspect, mirrorX);
    if (!ls || !rs || !lh || !rh || getVisibility(points[11]) < 0.25 || getVisibility(points[12]) < 0.25) {
      this.torsoNodes.chest.visible = false;
      this.torsoNodes.abdomen.visible = false;
      this.torsoNodes.pelvis.visible = false;
    } else {
      const shoulderMid = new THREE.Vector3().copy(ls).add(rs).multiplyScalar(0.5);
      const hipMid = new THREE.Vector3().copy(lh).add(rh).multiplyScalar(0.5);
      const torsoLen = Math.max(0.24, shoulderMid.distanceTo(hipMid));
      const shoulderWidth = Math.max(0.24, ls.distanceTo(rs));
      const setBlock = (mesh: THREE.Mesh, t: number, sx: number, sy: number, sz: number) => {
        mesh.visible = true;
        mesh.position.copy(shoulderMid).lerp(hipMid, t);
        mesh.scale.set(shoulderWidth * sx, torsoLen * sy, scale * sz);
      };
      setBlock(this.torsoNodes.chest, 0.22, 0.58, 0.22, 0.28);
      setBlock(this.torsoNodes.abdomen, 0.56, 0.43, 0.24, 0.23);
      setBlock(this.torsoNodes.pelvis, 0.86, 0.52, 0.19, 0.26);
    }

    const head = normalizedToWorld(points[0], aspect, mirrorX);
    if (showHead && head && getVisibility(points[0]) >= 0.25) {
      this.torsoNodes.head.visible = true;
      this.torsoNodes.head.position.copy(head);
      this.torsoNodes.head.scale.set(scale * 0.22, scale * 0.27, scale * 0.2);
    } else {
      this.torsoNodes.head.visible = false;
    }
  }

  private renderJoints(
    points: readonly (NormalizedLandmark | null | undefined)[],
    aspect: number,
    mirrorX: boolean,
    scale: number,
    highDetail: boolean,
  ): void {
    const base = Math.max(0.035, scale * 0.075);
    const jointRadii: Partial<Record<number, number>> = {
      11: base * 1.12,
      12: base * 1.12,
      13: base,
      14: base,
      15: base * 0.62,
      16: base * 0.62,
      23: base * 1.18,
      24: base * 1.18,
      25: base,
      26: base,
      27: base * 0.58,
      28: base * 0.58,
    };
    for (const [indexText, node] of Object.entries(this.jointNodes)) {
      const index = Number(indexText);
      const point = points[index];
      const mapped = normalizedToWorld(point, aspect, mirrorX);
      const radius = jointRadii[index];
      if (!point || !mapped || !radius || getVisibility(point) < 0.25) {
        node.group.visible = false;
        continue;
      }
      setJoint(node, mapped, radius, highDetail);
    }
  }

  private renderHands(
    hands: readonly HandFrame[],
    aspect: number,
    mirrorX: boolean,
    palette: TeacherAvatarPalette,
  ): void {
    const teacher = parseCssColor(palette.teacherStroke, 'rgb(84, 243, 168)');
    this.handMaterial.color.copy(teacher.color);
    while (this.handLines.length < hands.length) {
      const line = new THREE.LineSegments(new THREE.BufferGeometry(), this.handMaterial);
      this.handLines.push(line);
      this.proceduralRoot.add(line);
    }
    for (let i = 0; i < this.handLines.length; i += 1) {
      const line = this.handLines[i]!;
      const hand = hands[i];
      if (!hand?.landmarks?.length) {
        line.visible = false;
        continue;
      }
      const positions: number[] = [];
      for (const [a, b] of HAND_CONNECTIONS) {
        const p1 = normalizedToWorld(hand.landmarks[a], aspect, mirrorX);
        const p2 = normalizedToWorld(hand.landmarks[b], aspect, mirrorX);
        if (!p1 || !p2) continue;
        positions.push(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
      }
      line.geometry.dispose();
      line.geometry = new THREE.BufferGeometry();
      line.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      line.visible = positions.length > 0;
    }
  }

  private startModelLoad(url: string): void {
    if (this.gltfState !== 'idle') return;
    this.gltfState = 'loading';
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => this.onModelLoaded(gltf),
      undefined,
      () => {
        this.gltfState = 'failed';
        this.resolveModelReady();
      },
    );
  }

  private onModelLoaded(gltf: GLTF): void {
    if (this.disposed) return;
    this.gltfState = 'ready';
    this.gltfRoot = gltf.scene;
    this.gltfRoot.visible = false;
    this.gltfRoot.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.castShadow = false;
        object.frustumCulled = false;
        const hasMaterialArray = Array.isArray(object.material);
        const materials = (hasMaterialArray ? object.material : [object.material]) as THREE.Material[];
        const clonedMaterials = materials.map((material) => {
          const clone = material.clone();
          clone.transparent = true;
          clone.opacity = 0.92;
          clone.depthWrite = true;
          return clone;
        });
        object.material = (hasMaterialArray ? clonedMaterials : clonedMaterials[0]!) as typeof object.material;
      }
    });
    const box = new THREE.Box3().setFromObject(this.gltfRoot);
    box.getCenter(this.gltfCenter);
    box.getSize(this.gltfSize);
    this.gltfBones = this.collectGltfBones(this.gltfRoot);
    this.gltfRig = this.createGltfBjdRig(this.gltfRoot);
    this.root.add(this.gltfRoot);
    this.resolveModelReady();
  }

  private collectGltfBones(root: THREE.Object3D): Partial<Record<TeacherBodyPartId, THREE.Object3D>> {
    const bones: Partial<Record<TeacherBodyPartId, THREE.Object3D>> = {};
    const patterns: Record<TeacherBodyPartId, readonly RegExp[]> = {
      leftUpperArm: [/left.*upper.*arm/i, /leftarm/i, /upper_arm\.l/i, /mixamorigleftarm/i],
      leftForearm: [/left.*fore.*arm/i, /leftforearm/i, /lower_arm\.l/i, /mixamorigleftforearm/i],
      rightUpperArm: [/right.*upper.*arm/i, /rightarm/i, /upper_arm\.r/i, /mixamorigrightarm/i],
      rightForearm: [/right.*fore.*arm/i, /rightforearm/i, /lower_arm\.r/i, /mixamorigrightforearm/i],
      leftThigh: [/left.*thigh/i, /leftupleg/i, /thigh\.l/i, /mixamorigleftupleg/i],
      leftCalf: [/left.*calf/i, /leftleg/i, /shin\.l/i, /mixamorigleftleg/i],
      rightThigh: [/right.*thigh/i, /rightupleg/i, /thigh\.r/i, /mixamorigrightupleg/i],
      rightCalf: [/right.*calf/i, /rightleg/i, /shin\.r/i, /mixamorigrightleg/i],
    };
    root.traverse((object) => {
      for (const part of TEACHER_BODY_PARTS) {
        if (!bones[part.id] && namedBone(object.name, patterns[part.id])) bones[part.id] = object;
      }
    });
    return bones;
  }

  private renderLoadedModel(input: ThreeTeacherAvatarRenderInput, aspect: number): boolean {
    if (!this.gltfRoot || this.gltfState !== 'ready') {
      this.gltfRigRoot.visible = false;
      return false;
    }
    if (this.gltfRig && this.renderGltfBjdRig(input, aspect)) {
      this.gltfRoot.visible = false;
      return true;
    }
    const mappedParts = Object.keys(this.gltfBones).length;
    this.fitLoadedModelToPose(input, aspect);
    if (mappedParts < 4) {
      this.gltfRoot.visible = true;
      this.proceduralRoot.visible = true;
      this.gltfRigRoot.visible = false;
      return false;
    }
    this.gltfRoot.visible = true;
    this.proceduralRoot.visible = false;
    this.gltfRigRoot.visible = false;
    for (const part of TEACHER_BODY_PARTS) {
      const bone = this.gltfBones[part.id];
      const from = normalizedToWorld(input.poseLandmarks[part.from], aspect, Boolean(input.mirrorX));
      const to = normalizedToWorld(input.poseLandmarks[part.to], aspect, Boolean(input.mirrorX));
      if (!bone || !from || !to) continue;
      const direction = new THREE.Vector3().subVectors(to, from).normalize();
      bone.quaternion.slerp(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, -1, 0), direction), 0.85);
    }
    return true;
  }

  private fitLoadedModelToPose(input: ThreeTeacherAvatarRenderInput, aspect: number): void {
    if (!this.gltfRoot) return;
    const worldPoints = [0, 11, 12, 15, 16, 23, 24, 27, 28]
      .map((index) => input.poseLandmarks[index])
      .filter((point): point is NormalizedLandmark => !!point && getVisibility(point) >= 0.25)
      .map((point) => normalizedToWorld(point, aspect, Boolean(input.mirrorX)))
      .filter((point): point is THREE.Vector3 => !!point);
    const poseBox = new THREE.Box3();
    for (const point of worldPoints) poseBox.expandByPoint(point);
    const poseSize = new THREE.Vector3();
    const poseCenter = new THREE.Vector3();
    poseBox.getSize(poseSize);
    poseBox.getCenter(poseCenter);
    const fallbackCenter = normalizedToWorld(
      input.poseLandmarks[23] ?? input.poseLandmarks[0],
      aspect,
      Boolean(input.mirrorX),
    ) ?? new THREE.Vector3();
    const targetCenter = poseBox.isEmpty() ? fallbackCenter : poseCenter;
    const targetHeight = Math.max(poseSize.y * 1.06, bodyScale(input.poseLandmarks) * 3.6);
    const modelHeight = Math.max(this.gltfSize.y, 0.001);
    const scale = targetHeight / modelHeight;
    const sx = input.mirrorX ? -scale : scale;
    this.gltfRoot.scale.set(sx, scale, scale);
    this.gltfRoot.position.set(
      targetCenter.x - this.gltfCenter.x * sx,
      targetCenter.y - this.gltfCenter.y * scale,
      targetCenter.z - this.gltfCenter.z * scale - 0.12,
    );
  }

  private createGltfBjdRig(root: THREE.Object3D): GltfBjdRig | null {
    const upperArm = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.upperArm);
    const foreArm = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.foreArm);
    const thigh = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.thigh);
    const calf = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.calf);
    const shoulder = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.shoulder);
    const elbow = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.elbow);
    const knee = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.knee);
    const wrist = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.wrist);
    const foot = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.foot);
    const chest = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.chest);
    const waist = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.waist);
    const hip = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.hip);
    const head = this.findGltfPart(root, GLTF_BJD_PART_NAME_CANDIDATES.head);
    if (!upperArm || !foreArm || !thigh || !calf || !chest || !hip) return null;

    const make = (source: THREE.Object3D | null): GltfPartInstance | undefined => {
      if (!source) return undefined;
      const instance = this.createGltfPartInstance(source);
      this.gltfRigRoot.add(instance.group);
      return instance;
    };
    const limbSource: Record<TeacherBodyPartId, THREE.Object3D> = {
      leftUpperArm: upperArm,
      rightUpperArm: upperArm,
      leftForearm: foreArm,
      rightForearm: foreArm,
      leftThigh: thigh,
      rightThigh: thigh,
      leftCalf: calf,
      rightCalf: calf,
    };
    const limbs = {} as Partial<Record<TeacherBodyPartId, GltfPartInstance>>;
    for (const part of TEACHER_BODY_PARTS) limbs[part.id] = make(limbSource[part.id]);
    return {
      limbs,
      joints: {
        11: make(shoulder),
        12: make(shoulder),
        13: make(elbow),
        14: make(elbow),
        15: make(wrist),
        16: make(wrist),
        23: make(hip),
        24: make(hip),
        25: make(knee),
        26: make(knee),
        27: make(foot),
        28: make(foot),
      },
      torso: {
        chest: make(chest),
        waist: make(waist),
        hip: make(hip),
        head: make(head),
      },
    };
  }

  private findGltfPart(root: THREE.Object3D, names: readonly string[]): THREE.Object3D | null {
    for (const name of names) {
      const found = root.getObjectByName(name);
      if (found) return found;
    }
    const normalizedNames = names.map((name) => name.toLowerCase());
    let fallback: THREE.Object3D | null = null;
    root.traverse((object) => {
      if (fallback) return;
      const normalized = object.name.toLowerCase().replace(/\./g, '');
      if (normalizedNames.includes(normalized)) fallback = object;
    });
    return fallback;
  }

  private createGltfPartInstance(source: THREE.Object3D): GltfPartInstance {
    source.updateWorldMatrix(true, true);
    const clone = source.clone(true);
    source.matrixWorld.decompose(clone.position, clone.quaternion, clone.scale);
    clone.updateMatrix();
    clone.updateMatrixWorld(true);
    clone.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.frustumCulled = false;
      const hasMaterialArray = Array.isArray(object.material);
      const materials = (hasMaterialArray ? object.material : [object.material]) as THREE.Material[];
      const clonedMaterials = materials.map((material) => {
        const role = /outline/i.test(`${object.name} ${material.name}`) ? 'outline' : 'resin';
        const cloned = new THREE.MeshStandardMaterial({
          color: role === 'outline' ? 0x7defff : 0xf3eadc,
          roughness: 0.62,
          metalness: 0.02,
          transparent: true,
          opacity: role === 'outline' ? 0.92 : 1,
          depthWrite: true,
        });
        cloned.userData.role = role;
        cloned.transparent = true;
        this.gltfPartMaterials.push(cloned);
        return cloned;
      });
      object.material = (hasMaterialArray ? clonedMaterials : clonedMaterials[0]!) as typeof object.material;
    });
    const box = new THREE.Box3().setFromObject(source);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);
    clone.position.sub(center);
    const group = new THREE.Group();
    group.add(clone);
    return {
      group,
      size: new THREE.Vector3(
        Math.max(size.x, 0.001),
        Math.max(size.y, 0.001),
        Math.max(size.z, 0.001),
      ),
    };
  }

  private renderGltfBjdRig(input: ThreeTeacherAvatarRenderInput, aspect: number): boolean {
    if (!this.gltfRig) return false;
    this.gltfRigRoot.visible = true;
    const points = input.poseLandmarks;
    const mirrorX = Boolean(input.mirrorX);
    const scale = bodyScale(points);
    const shoulderWidth = this.getShoulderWidth(points, aspect, mirrorX, scale);
    let visibleParts = 0;

    for (const part of TEACHER_BODY_PARTS) {
      const instance = this.gltfRig.limbs[part.id];
      const from = points[part.from];
      const to = points[part.to];
      const start = normalizedToWorld(from, aspect, mirrorX);
      const end = normalizedToWorld(to, aspect, mirrorX);
      if (!instance || !hasFinitePosePoint(from) || !hasFinitePosePoint(to) || !start || !end) {
        if (instance) instance.group.visible = false;
        continue;
      }
      const segmentLength = start.distanceTo(end);
      const width = this.getGltfLimbWidth(part.id, segmentLength, shoulderWidth, scale);
      const depth = width * GLTF_LIMB_PROPORTIONS[part.id].depthRatio;
      this.setGltfSegment(instance, start, end, width, depth);
      visibleParts += 1;
    }

    this.renderGltfTorso(points, aspect, mirrorX, scale);
    this.renderGltfJoints(points, aspect, mirrorX, scale, Boolean(input.highDetail));
    return visibleParts >= 4;
  }

  private getShoulderWidth(
    points: readonly (NormalizedLandmark | null | undefined)[],
    aspect: number,
    mirrorX: boolean,
    scale: number,
  ): number {
    const ls = normalizedToWorld(points[11], aspect, mirrorX);
    const rs = normalizedToWorld(points[12], aspect, mirrorX);
    if (ls && rs) return Math.max(0.08, ls.distanceTo(rs));
    return Math.max(0.18, scale * 0.52);
  }

  private getGltfLimbWidth(
    partId: TeacherBodyPartId,
    length: number,
    shoulderWidth: number,
    scale: number,
  ): number {
    const proportions = GLTF_LIMB_PROPORTIONS[partId];
    const target = Math.min(
      shoulderWidth * proportions.shoulderRatio,
      length * proportions.lengthRatio,
      scale * proportions.scaleRatio,
    );
    return Math.max(proportions.min, target);
  }

  private setGltfSegment(
    instance: GltfPartInstance,
    start: THREE.Vector3,
    end: THREE.Vector3,
    width: number,
    depth: number,
  ): void {
    const length = applySegmentTransform(instance.group, start, end);
    if (!length) return;
    instance.group.scale.set(
      width / instance.size.x,
      length / instance.size.y,
      depth / instance.size.z,
    );
  }

  private setGltfPoint(
    instance: GltfPartInstance | undefined,
    position: THREE.Vector3 | null,
    width: number,
    height: number,
    depth = width,
  ): void {
    if (!instance || !position) {
      if (instance) instance.group.visible = false;
      return;
    }
    instance.group.visible = true;
    instance.group.position.copy(position);
    instance.group.quaternion.identity();
    instance.group.scale.set(
      width / instance.size.x,
      height / instance.size.y,
      depth / instance.size.z,
    );
  }

  private renderGltfTorso(
    points: readonly (NormalizedLandmark | null | undefined)[],
    aspect: number,
    mirrorX: boolean,
    scale: number,
  ): void {
    const ls = normalizedToWorld(points[11], aspect, mirrorX);
    const rs = normalizedToWorld(points[12], aspect, mirrorX);
    const lh = normalizedToWorld(points[23], aspect, mirrorX);
    const rh = normalizedToWorld(points[24], aspect, mirrorX);
    if (
      !ls || !rs || !lh || !rh ||
      !hasFinitePosePoint(points[11]) ||
      !hasFinitePosePoint(points[12]) ||
      !hasFinitePosePoint(points[23]) ||
      !hasFinitePosePoint(points[24])
    ) {
      this.setGltfPoint(this.gltfRig?.torso.chest, null, 0, 0);
      this.setGltfPoint(this.gltfRig?.torso.waist, null, 0, 0);
      this.setGltfPoint(this.gltfRig?.torso.hip, null, 0, 0);
      return;
    }
    const shoulderMid = new THREE.Vector3().copy(ls).add(rs).multiplyScalar(0.5);
    const hipMid = new THREE.Vector3().copy(lh).add(rh).multiplyScalar(0.5);
    const torsoLength = Math.max(0.24, shoulderMid.distanceTo(hipMid));
    const shoulderWidth = Math.max(0.2, ls.distanceTo(rs));
    const pointAt = (t: number) => new THREE.Vector3().copy(shoulderMid).lerp(hipMid, t);
    this.setGltfPoint(this.gltfRig?.torso.chest, pointAt(0.22), shoulderWidth * 1.06, torsoLength * 0.43, scale * 0.46);
    this.setGltfPoint(this.gltfRig?.torso.waist, pointAt(0.55), shoulderWidth * 0.78, torsoLength * 0.25, scale * 0.34);
    this.setGltfPoint(this.gltfRig?.torso.hip, pointAt(0.86), shoulderWidth * 0.94, torsoLength * 0.28, scale * 0.42);

    const head = normalizedToWorld(points[0], aspect, mirrorX);
    if (head && hasFinitePosePoint(points[0])) {
      const headWidth = Math.max(0.16, Math.min(scale * 0.34, shoulderWidth * 0.52));
      this.setGltfPoint(this.gltfRig?.torso.head, head, headWidth, headWidth * 1.34, headWidth * 1.12);
    } else {
      const fallbackHead = new THREE.Vector3()
        .copy(shoulderMid)
        .add(new THREE.Vector3().subVectors(shoulderMid, hipMid).normalize().multiplyScalar(torsoLength * 0.56));
      const headWidth = Math.max(0.16, Math.min(scale * 0.34, shoulderWidth * 0.52));
      this.setGltfPoint(this.gltfRig?.torso.head, fallbackHead, headWidth, headWidth * 1.34, headWidth * 1.12);
    }
  }

  private renderGltfJoints(
    points: readonly (NormalizedLandmark | null | undefined)[],
    aspect: number,
    mirrorX: boolean,
    scale: number,
    highDetail: boolean,
  ): void {
    const shoulderWidth = this.getShoulderWidth(points, aspect, mirrorX, scale);
    const joint = Math.max(0.03, Math.min(scale * 0.052, shoulderWidth * 0.085));
    const sizes: Partial<Record<number, [number, number, number]>> = {
      11: [joint * 2.5, joint * 2.2, joint * 2.2],
      12: [joint * 2.5, joint * 2.2, joint * 2.2],
      13: [joint * 2.1, joint * 1.9, joint * 1.9],
      14: [joint * 2.1, joint * 1.9, joint * 1.9],
      15: [joint * 1.7, joint * 1.1, joint * 1.1],
      16: [joint * 1.7, joint * 1.1, joint * 1.1],
      23: [joint * 2.8, joint * 2.1, joint * 2.2],
      24: [joint * 2.8, joint * 2.1, joint * 2.2],
      25: [joint * 2.1, joint * 1.7, joint * 1.8],
      26: [joint * 2.1, joint * 1.7, joint * 1.8],
      27: [joint * 2.4, joint * 1.4, joint * 2.2],
      28: [joint * 2.4, joint * 1.4, joint * 2.2],
    };
    for (const [indexText, instance] of Object.entries(this.gltfRig?.joints ?? {})) {
      const index = Number(indexText);
      const point = points[index];
      const mapped = normalizedToWorld(point, aspect, mirrorX);
      const size = sizes[index];
      if (!instance || !hasFinitePosePoint(point) || !mapped || !size || (!highDetail && (index === 15 || index === 16))) {
        if (instance) instance.group.visible = false;
        continue;
      }
      this.setGltfPoint(instance, mapped, size[0], size[1], size[2]);
    }
  }
}
