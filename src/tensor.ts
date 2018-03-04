/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {doc} from './doc';
import {ENV} from './environment';
import * as ops from './ops/ops';
import {RandNormalDataTypes} from './ops/rand';
import * as tensor_util from './tensor_util';
import {DataType, DataTypeMap, Rank, ShapeMap, TypedArray} from './types';
import * as util from './util';

/** @hidden */
export interface TensorData {
  dataId?: DataId;
  values?: TypedArray;
}

/**
 * A mutable object, similar to `Tensor`, that allows users to set values
 * at locations before converting to an immutable `Tensor`.
 *
 * See `buffer` for creating a tensor buffer.
 */
@doc({heading: 'Tensors', subheading: 'Classes'})
export class TensorBuffer<R extends Rank> {
  size: number;

  private strides: number[];

  constructor(
      public shape: ShapeMap[R], public dtype: DataType,
      public values: TypedArray) {
    if (values != null) {
      const n = values.length;
      const size = util.sizeFromShape(shape);
      util.assert(
          n === size,
          `Length of values '${n}' does not match the size ` +
              `inferred by the shape '${size}'`);
    }
    this.values =
        values || util.getTypedArrayFromDType(dtype, util.sizeFromShape(shape));
    this.strides = computeStrides(shape);
    this.size = util.sizeFromShape(shape);
  }

  /**
   * Sets a value in the buffer at a given location.
   *
   * @param value The value to set.
   * @param locs  The location indices.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  set(value: number, ...locs: number[]) {
    if (locs.length === 0) {
      locs = [0];
    }
    util.assert(
        locs.length === this.rank,
        `The number of provided coordinates (${locs.length}) must ` +
            `match the rank (${this.rank})`);
    const index = this.locToIndex(locs);
    this.values[index] = value;
  }

  /**
   * Returns the value in the buffer at the provided location.
   *
   * @param locs The location indices.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  get(...locs: number[]): number {
    if (locs.length === 0) {
      locs = [0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return this.values[index];
  }

  locToIndex(locs: number[]): number {
    if (this.rank === 0) {
      return 0;
    } else if (this.rank === 1) {
      return locs[0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return index;
  }

  indexToLoc(index: number): number[] {
    if (this.rank === 0) {
      return [];
    } else if (this.rank === 1) {
      return [index];
    }
    const locs: number[] = new Array(this.shape.length);
    for (let i = 0; i < locs.length - 1; ++i) {
      locs[i] = Math.floor(index / this.strides[i]);
      index -= locs[i] * this.strides[i];
    }
    locs[locs.length - 1] = index;
    return locs;
  }

  get rank() {
    return this.shape.length;
  }

  /**
   * Creates an immutable `Tensor` object from the buffer.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  toTensor(): Tensor<R> {
    return Tensor.make(this.shape, {values: this.values}, this.dtype);
  }
}

/**
 * We wrap data id since we use weak map to avoid memory leaks.
 * Since we have our own memory management, we have a reference counter
 * mapping a tensor to its data, so there is always a pointer (even if that
 * data is otherwise garbage collectable).
 * See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/
 * Global_Objects/WeakMap
 */
export type DataId = object;  // object instead of {} to force non-primitive.

/**
 * A `Tensor` object represents an immutable, multidimensional array of numbers
 * that has a shape and a data type.
 *
 * See `tensor` for details on how to create a `Tensor`.
 */
@doc({heading: 'Tensors', subheading: 'Classes'})
export class Tensor<R extends Rank = Rank> {
  private static nextId = 0;

  /** Unique id of this tensor. */
  readonly id: number;
  /**
   * Id of the bucket holding the data for this tensor. Multiple arrays can
   * point to the same bucket (e.g. when calling array.reshape()).
   */
  dataId: DataId;
  /** The shape of the tensor. */
  readonly shape: ShapeMap[R];
  /** Number of elements in the tensor. */
  readonly size: number;
  /** The data type for the array. */
  readonly dtype: DataType;
  /** The rank type for the array (see `Rank` enum). */
  readonly rankType: R;

  /**
   * Number of elements to skip in each dimension when indexing. See
   * https://docs.scipy.org/doc/numpy/reference/generated/\
   * numpy.ndarray.strides.html
   */
  readonly strides: number[];

  protected constructor(
      shape: ShapeMap[R], dtype: DataType, values?: TypedArray,
      dataId?: DataId) {
    this.size = util.sizeFromShape(shape);
    if (values != null) {
      util.assert(
          this.size === values.length,
          `Constructing tensor of shape (${this.size}) should match the ` +
              `length of values (${values.length})`);
    }
    this.shape = shape;
    this.dtype = dtype || 'float32';
    this.strides = computeStrides(shape);
    this.dataId = dataId != null ? dataId : {};
    this.id = Tensor.nextId++;
    this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher') as R;
    ENV.engine.registerTensor(this);
    if (values != null) {
      ENV.engine.write(this.dataId, values);
    }
  }

  /** @deprecated Please use dl.ones() */
  static ones<R extends Rank>(shape: ShapeMap[R], dtype?: DataType): Tensor<R> {
    return ops.ones(shape, dtype);
  }

  /** @deprecated Please use dl.zeros() */
  static zeros<R extends Rank>(shape: ShapeMap[R], dtype?: DataType):
      Tensor<R> {
    return ops.zeros(shape, dtype);
  }

  /** @deprecated Please use dl.onesLike() */
  static onesLike<T extends Tensor>(x: T): T {
    return ops.onesLike(x);
  }

  /** @deprecated Please use dl.zerosLike() */
  static zerosLike<T extends Tensor>(x: T): T {
    return ops.zerosLike(x);
  }

  /** @deprecated Please use dl.clone() */
  static like<T extends Tensor>(x: T): T {
    return ops.clone(x);
  }

  /**
   * Makes a new tensor with the provided shape and values. Values should be in
   * a flat array.
   */
  static make<T extends Tensor<R>, D extends DataType = 'float32',
                                             R extends Rank = Rank>(
      shape: ShapeMap[R], data: TensorData, dtype?: D): T {
    return new Tensor(shape, dtype, data.values, data.dataId) as T;
  }

  /** @deprecated Please use dl.fromPixels() */
  static fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels = 3): Tensor3D {
    return ops.fromPixels(pixels, numChannels);
  }

  /** @deprecated Please use dl.rand() */
  static rand<R extends Rank>(
      shape: ShapeMap[R], randFunction: () => number,
      dtype?: DataType): Tensor<R> {
    return ops.rand(shape, randFunction, dtype);
  }

  /** @deprecated Please use dl.randomNormal() */
  static randNormal<R extends Rank>(
      shape: ShapeMap[R], mean = 0, stdDev = 1,
      dtype?: keyof RandNormalDataTypes, seed?: number): Tensor<R> {
    return ops.randomNormal(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.truncatedNormal() */
  static randTruncatedNormal<R extends Rank>(
      shape: ShapeMap[R], mean = 0, stdDev = 1,
      dtype?: keyof RandNormalDataTypes, seed?: number): Tensor<R> {
    return ops.truncatedNormal(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randomUniform() */
  static randUniform<R extends Rank>(
      shape: ShapeMap[R], a: number, b: number, dtype?: DataType): Tensor<R> {
    return ops.randomUniform(shape, a, b, dtype);
  }

  /** Flatten a Tensor to a 1D array. */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  flatten(): Tensor1D {
    this.throwIfDisposed();
    return this.as1D();
  }

  /** Converts a size-1 `Tensor` to a `Scalar`. */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  asScalar(): Scalar {
    this.throwIfDisposed();
    util.assert(this.size === 1, 'The array must have only 1 element.');
    return this.reshape<Rank.R0>([]);
  }

  /** Converts a `Tensor` to a `Tensor1D`. */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  as1D(): Tensor1D {
    this.throwIfDisposed();
    return this.reshape<Rank.R1>([this.size]);
  }

  /**
   * Converts a `Tensor` to a `Tensor2D`.
   *
   * @param rows Number of rows in `Tensor2D`.
   * @param columns Number of columns in `Tensor2D`.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  as2D(rows: number, columns: number): Tensor2D {
    this.throwIfDisposed();
    return this.reshape<Rank.R2>([rows, columns]);
  }

  /**
   * Converts a `Tensor` to a `Tensor3D`.
   *
   * @param rows Number of rows in `Tensor3D`.
   * @param columns Number of columns in `Tensor3D`.
   * @param depth Depth of `Tensor3D`.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  as3D(rows: number, columns: number, depth: number): Tensor3D {
    this.throwIfDisposed();
    return this.reshape<Rank.R3>([rows, columns, depth]);
  }

  /**
   * Converts a `Tensor` to a `Tensor4D`.
   *
   * @param rows Number of rows in `Tensor4D`.
   * @param columns Number of columns in `Tensor4D`.
   * @param depth Depth of `Tensor4D`.
   * @param depth2 4th dimension of `Tensor4D`.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  as4D(rows: number, columns: number, depth: number, depth2: number): Tensor4D {
    this.throwIfDisposed();
    return this.reshape<Rank.R4>([rows, columns, depth, depth2]);
  }

  /**
   * Casts a `Tensor` to a specified dtype.
   *
   * @param dtype Data-type to cast the tensor to.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  asType<T extends this>(this: T, dtype: DataType): T {
    this.throwIfDisposed();
    return ops.cast(this, dtype);
  }

  get rank(): number {
    return this.shape.length;
  }

  /** @deprecated. Use `tensor.buffer().get(...locs)` */
  get(...locs: number[]) {
    this.throwIfDisposed();
    if (locs.length === 0) {
      locs = [0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return this.dataSync()[index];
  }

  /** @deprecated. Use `tensor.buffer().get(...locs)` */
  async val(...locs: number[]): Promise<number> {
    if (locs.length === 0) {
      locs = [0];
    }
    this.throwIfDisposed();
    await this.data();
    return this.get(...locs);
  }

  /** @deprecated. Use `tensor.buffer().locToIndex(locs)` */
  locToIndex(locs: number[]): number {
    this.throwIfDisposed();
    if (this.rank === 0) {
      return 0;
    } else if (this.rank === 1) {
      return locs[0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return index;
  }

  /** @deprecated. Use `tensor.buffer().indexToLoc(index)` */
  indexToLoc(index: number): number[] {
    this.throwIfDisposed();
    if (this.rank === 0) {
      return [];
    } else if (this.rank === 1) {
      return [index];
    }
    const locs: number[] = new Array(this.shape.length);
    for (let i = 0; i < locs.length - 1; ++i) {
      locs[i] = Math.floor(index / this.strides[i]);
      index -= locs[i] * this.strides[i];
    }
    locs[locs.length - 1] = index;
    return locs;
  }

  /** @deprecated Use dataSync() instead. */
  getValues(): TypedArray {
    return this.dataSync();
  }

  /** @deprecated Use data() instead. */
  getValuesAsync(): Promise<TypedArray> {
    return this.data();
  }

  /** Returns a `TensorBuffer` that holds the underlying data. */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  buffer(): TensorBuffer<R> {
    return ops.buffer(this.shape, this.dtype, this.dataSync());
  }

  /**
   * Asynchronously downloads the values from the `Tensor`. Returns a promise of
   * `TypedArray` that resolves when the computation has finished.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  async data(): Promise<TypedArray> {
    this.throwIfDisposed();
    return ENV.engine.read(this.dataId);
  }

  /**
   * Synchronously downloads the values from the `Tensor`. This blocks the UI
   * thread until the values are ready, which can cause performance issues.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  dataSync(): TypedArray {
    this.throwIfDisposed();
    return ENV.engine.readSync(this.dataId);
  }

  /**
   * Disposes `Tensor` from memory.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this.isDisposed = true;
    ENV.engine.disposeTensor(this);
  }

  private isDisposed = false;
  protected throwIfDisposed() {
    if (this.isDisposed) {
      throw new Error(`Tensor is disposed.`);
    }
  }

  /** Casts the array to type `float32` */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  toFloat<T extends this>(this: T): T {
    return this.asType('float32');
  }

  /** Casts the array to type `int32` */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  toInt() {
    return this.asType('int32');
  }

  /** Casts the array to type `bool` */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  toBool() {
    return this.asType('bool');
  }

  /**
   * Prints the `Tensor`. See `print` for details.
   *
   * @param verbose Whether to print verbose information about the tensor,
   *    including dtype and size.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  print(verbose = false): void {
    return ops.print(this, verbose);
  }

  /**
   * Reshapes the tensor into the provided shape.
   * See `reshape` for more details.
   *
   * @param newShape An array of integers defining the output tensor shape.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  reshape<R2 extends Rank>(newShape: ShapeMap[R2]): Tensor<R2> {
    this.throwIfDisposed();
    return ops.reshape(this, newShape);
  }

  /**
   * Reshapes the tensor into the shape of the provided tensor.
   *
   * @param x The tensor of required shape.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  reshapeAs<T extends Tensor>(x: T): T {
    this.throwIfDisposed();
    return this.reshape(x.shape) as T;
  }

  /**
   * Returns a `Tensor` that has expanded rank, by inserting a dimension
   * into the tensor's shape. See `expandDims` for details.
   *
   * @param axis The dimension index at which to insert shape of 1. Defaults to
   *    0 (the first dimension).
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  expandDims<R2 extends Rank>(axis = 0): Tensor<R2> {
    return ops.expandDims(this, axis);
  }

  /**
   * Returns a `Tensor` with dimensions of size 1 removed from the shape.
   * See `squeeze` for more details.
   *
   * @param axis A list of numbers. If specified, only squeezes the
   *    dimensions listed. The dimension index starts at 0. It is an error to
   *    squeeze a dimension that is not 1.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  squeeze<T extends Tensor>(axis?: number[]): T {
    this.throwIfDisposed();
    return ops.squeeze(this, axis);
  }

  /** Returns a copy of the tensor. See `clone` for details. */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  clone<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.clone(this);
  }

  /** Returns a human-readable description of the tensor. Useful for logging. */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  toString(): string {
    return tensor_util.tensorToString(this, true /* verbose */);
  }

  // Below is chain API that is not exposed to docs to avoid repetition. To
  // expose a method, move it above this comment and add @doc and jsdoc.

  tile<T extends this>(this: T, reps: number[]): T {
    this.throwIfDisposed();
    return ops.tile(this, reps);
  }

  gather<T extends this>(this: T, indices: Tensor1D, axis = 0): T {
    this.throwIfDisposed();
    return ops.gather(this, indices);
  }

  matMul(b: Tensor2D, transposeA = false, transposeB = false): Tensor2D {
    this.throwIfDisposed();
    return ops.matMul(this as Tensor2D, b, transposeA, transposeB);
  }
  norm(
      ord: number|'euclidean'|'fro' = 'euclidean', axis: number|number[] = null,
      keepDims = false): Tensor {
    this.throwIfDisposed();
    return ops.norm(this, ord, axis, keepDims);
  }
  slice<T extends Tensor<R>>(this: T, begin: ShapeMap[R], size: ShapeMap[R]):
      T {
    this.throwIfDisposed();
    return ops.slice(this, begin, size);
  }
  reverse<T extends Tensor>(this: T, axis?: number|number[]): T {
    this.throwIfDisposed();
    return ops.reverse(this, axis);
  }
  concat<T extends Tensor>(this: T, x: T, axis = 0): T {
    this.throwIfDisposed();
    return ops.concat([this, x], axis);
  }
  stack(x: Tensor, axis = 0): Tensor {
    return ops.stack([this, x], axis);
  }
  pad<T extends Tensor>(
      this: T, paddings: Array<[number, number]>, constantValue = 0): T {
    return ops.pad(this, paddings, constantValue);
  }
  batchNormalization(
      mean: Tensor<R>|Tensor1D, variance: Tensor<R>|Tensor1D,
      varianceEpsilon = .001, scale?: Tensor<R>|Tensor1D,
      offset?: Tensor<R>|Tensor1D): Tensor<R> {
    this.throwIfDisposed();
    return ops.batchNormalization(
        this, mean, variance, varianceEpsilon, scale, offset);
  }

  // Reduction ops.

  logSumExp<T extends Tensor>(axis: number|number[] = null, keepDims = false):
      T {
    this.throwIfDisposed();
    return ops.logSumExp(this, axis, keepDims);
  }
  sum<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.sum(this, axis, keepDims);
  }
  mean<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.mean(this, axis, keepDims);
  }
  min<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.min(this, axis, keepDims);
  }
  max<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.max(this, axis, keepDims);
  }
  argMin<T extends Tensor>(axis: number = null): T {
    this.throwIfDisposed();
    return ops.argMin(this, axis);
  }
  argMax<T extends Tensor>(axis: number = null): T {
    this.throwIfDisposed();
    return ops.argMax(this, axis);
  }

  // Binary ops.

  add<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.add(this, x);
  }
  addStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.addStrict(this, x);
  }
  sub<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.sub(this, x);
  }
  subStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.subStrict(this, x);
  }
  pow<T extends Tensor>(this: T, exp: Tensor): T {
    this.throwIfDisposed();
    return ops.pow(this, exp);
  }
  powStrict(exp: Tensor): Tensor<R> {
    this.throwIfDisposed();
    return ops.powStrict(this, exp);
  }
  mul<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.mul(this, x);
  }
  mulStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.mulStrict(this, x);
  }
  div<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.div(this, x);
  }
  divStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.divStrict(this, x);
  }
  minimum<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.minimum(this, x);
  }
  minimumStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.minimumStrict(this, x);
  }
  maximum<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.maximum(this, x);
  }
  maximumStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.maximumStrict(this, x);
  }
  transpose<T extends Tensor>(this: T, perm?: number[]): T {
    this.throwIfDisposed();
    return ops.transpose(this, perm);
  }

  // Compare ops.

  notEqual<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.notEqual(this, x);
  }
  notEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.notEqualStrict(this, x);
  }
  less<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.less(this, x);
  }
  lessStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.lessStrict(this, x);
  }
  equal<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.equal(this, x);
  }
  equalStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.equalStrict(this, x);
  }
  lessEqual<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.lessEqual(this, x);
  }
  lessEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.lessEqualStrict(this, x);
  }
  greater<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.greater(this, x);
  }
  greaterStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.greaterStrict(this, x);
  }
  greaterEqual<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.greaterEqual(this, x);
  }
  greaterEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.greaterEqualStrict(this, x);
  }

  // Compare ops.
  logicalAnd(x: Tensor): Tensor {
    this.throwIfDisposed();
    return ops.logicalAnd(this, x);
  }
  logicalOr(x: Tensor): Tensor {
    this.throwIfDisposed();
    return ops.logicalOr(this, x);
  }
  logicalXor(x: Tensor): Tensor {
    this.throwIfDisposed();
    return ops.logicalXor(this, x);
  }
  where(condition: Tensor, x: Tensor): Tensor {
    this.throwIfDisposed();
    return ops.where(condition, this, x);
  }

  // Unary ops.
  neg<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.neg(this);
  }
  ceil<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.ceil(this);
  }
  floor<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.floor(this);
  }
  exp<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.exp(this);
  }
  log<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.log(this);
  }
  sqrt<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.sqrt(this);
  }
  square<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.square(this);
  }
  abs<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.abs(this);
  }
  clipByValue(min: number, max: number): Tensor<R> {
    this.throwIfDisposed();
    return ops.clipByValue(this, min, max);
  }
  relu<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.relu(this);
  }
  elu<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.elu(this);
  }
  selu<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.selu(this);
  }
  leakyRelu(alpha = 0.2): Tensor<R> {
    this.throwIfDisposed();
    return ops.leakyRelu(this, alpha);
  }
  prelu(alpha: Tensor<R>): Tensor<R> {
    this.throwIfDisposed();
    return ops.prelu(this, alpha);
  }
  sigmoid<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.sigmoid(this);
  }
  sin<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.sin(this);
  }
  cos<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.cos(this);
  }
  tan<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.tan(this);
  }
  asin<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.asin(this);
  }
  acos<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.acos(this);
  }
  atan<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.atan(this);
  }
  sinh<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.sinh(this);
  }
  cosh<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.cosh(this);
  }
  tanh<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return ops.tanh(this);
  }
  step<T extends Tensor>(this: T, alpha = 0.0): T {
    this.throwIfDisposed();
    return ops.step(this, alpha);
  }
  softmax<T extends this>(this: T, dim = -1): T {
    this.throwIfDisposed();
    return ops.softmax(this, dim);
  }

  // Image ops.
  resizeBilinear<T extends Tensor3D|Tensor4D>(
      this: T, newShape2D: [number, number], alignCorners = false): T {
    (this as Tensor).throwIfDisposed();
    return ops.image.resizeBilinear(this, newShape2D, alignCorners);
  }

  // Convolutions.
  conv1d<T extends Tensor2D|Tensor3D>(
      this: T, filter: Tensor3D, stride: number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.conv1d(this, filter, stride, pad, dimRoundingMode);
  }
  conv2d<T extends Tensor3D|Tensor4D>(
      this: T, filter: Tensor4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.conv2d(this, filter, strides, pad, dimRoundingMode);
  }
  conv2dTranspose<T extends Tensor3D|Tensor4D>(
      this: T, filter: Tensor4D,
      outputShape: [number, number, number, number]|[number, number, number],
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.conv2dTranspose(
        this, filter, outputShape, strides, pad, dimRoundingMode);
  }
  depthwiseConv2D<T extends Tensor3D|Tensor4D>(
      this: T, filter: Tensor4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, dilations: [number, number]|number = [1, 1],
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.depthwiseConv2d(
        this, filter, strides, pad, dilations, dimRoundingMode);
  }

  // Pooling.
  avgPool<T extends Tensor3D|Tensor4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.avgPool(this, filterSize, strides, pad, dimRoundingMode);
  }
  maxPool<T extends Tensor3D|Tensor4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.maxPool(this, filterSize, strides, pad, dimRoundingMode);
  }
  minPool<T extends Tensor3D|Tensor4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.minPool(this, filterSize, strides, pad, dimRoundingMode);
  }
  localResponseNormalization<T extends Tensor3D|Tensor4D>(
      this: T, radius = 5, bias = 1, alpha = 1, beta = 0.5,
      normRegion: 'acrossChannels'|'withinChannel' = 'acrossChannels'): T {
    return ops.localResponseNormalization(
        this, radius, bias, alpha, beta, normRegion);
  }

  variable(trainable = true, name?: string, dtype?: DataType): Variable<R> {
    this.throwIfDisposed();
    return Variable.variable(this, trainable, name, dtype);
  }
}

/** @doclink Tensor */
export class Scalar extends Tensor<Rank.R0> {
  static new(value: number|boolean, dtype?: DataType): Scalar {
    return ops.scalar(value, dtype);
  }
}

/** @doclink Tensor */
export class Tensor1D extends Tensor<Rank.R1> {
  static new<D extends DataType = 'float32'>(
      values: DataTypeMap[D]|number[]|boolean[], dtype?: D): Tensor1D {
    return ops.tensor1d(values, dtype);
  }
}

/** @doclink Tensor */
export class Tensor2D extends Tensor<Rank.R2> {
  static new<D extends DataType = 'float32'>(
      shape: [number, number],
      values: DataTypeMap[D]|number[]|number[][]|boolean[]|boolean[][],
      dtype?: D): Tensor2D {
    return ops.tensor2d(values, shape, dtype);
  }
}

/** @doclink Tensor */
export class Tensor3D extends Tensor<Rank.R3> {
  static new<D extends DataType = 'float32'>(
      shape: [number, number, number],
      values: DataTypeMap[D]|number[]|number[][][]|boolean[]|boolean[][][],
      dtype?: D): Tensor3D {
    return ops.tensor3d(values, shape, dtype);
  }
}

/** @doclink Tensor */
export class Tensor4D extends Tensor<Rank.R4> {
  static new<D extends DataType = 'float32'>(
      shape: [number, number, number, number],
      values: DataTypeMap[D]|number[]|number[][][][]|boolean[]|boolean[][][][],
      dtype?: D): Tensor4D {
    return ops.tensor4d(values, shape, dtype);
  }
}

/**
 * A mutable `Tensor`, useful for persisting state, e.g. for training.
 */
@doc({heading: 'Tensors', subheading: 'Classes'})
export class Variable<R extends Rank = Rank> extends Tensor<R> {
  private static nextVarId = 0;
  name: string;

  /**
   * Private constructor since we can not add logic before calling `super()`.
   * Instead, we expose static `Variable.variable` method below, which will be
   * added to global namespace.
   */
  private constructor(
      initialValue: Tensor<R>, public trainable = true, name?: string) {
    super(
        initialValue.shape, initialValue.dtype, null /* values */,
        initialValue.dataId);
    this.name = name;
    if (this.name == null) {
      this.name = Variable.nextVarId.toString();
      Variable.nextVarId++;
    }
    ENV.engine.registerVariable(this);
  }

  /**
   * Creates a new variable with the provided initial value.
   * ```js
   * const x = dl.variable(dl.tensor([1, 2, 3]));
   * x.assign(dl.tensor([4, 5, 6]));
   *
   * x.print();
   * ```
   *
   * @param initialValue Initial value for the tensor.
   * @param trainable If true, optimizers are allowed to update it.
   * @param name Name of the variable. Defaults to a unique id.
   * @param dtype If set, initialValue will be converted to the given type.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static variable<R extends Rank>(
      initialValue: Tensor<R>, trainable = true, name?: string,
      dtype?: DataType): Variable<R> {
    if (dtype != null && dtype !== initialValue.dtype) {
      initialValue = initialValue.asType(dtype) as Tensor<R>;
    }
    return new Variable(initialValue, trainable, name);
  }

  /**
   * Assign a new `Tensor` to this variable. The new `Tensor` must have the
   * same shape and dtype as the old `Tensor`.
   *
   * @param newValue New tensor to be assigned to this variable.
   */
  @doc({heading: 'Tensors', subheading: 'Classes'})
  assign(newValue: Tensor<R>): void {
    if (newValue.dtype !== this.dtype) {
      throw new Error(
          `dtype of the new value (${newValue.dtype}) and ` +
          `previous value (${this.dtype}) must match`);
    }
    if (!util.arraysEqual(newValue.shape, this.shape)) {
      throw new Error(
          `shape of the new value (${newValue.shape}) and ` +
          `previous value (${this.shape}) must match`);
    }
    ENV.engine.disposeTensor(this);
    this.dataId = newValue.dataId;
    ENV.engine.registerTensor(this);
  }
}

const variable = Variable.variable;
export {variable};

function computeStrides(shape: number[]): number[] {
  const rank = shape.length;
  if (rank < 2) {
    return [];
  }

  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  const strides = new Array(rank - 1);
  strides[rank - 2] = shape[rank - 1];
  for (let i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Aliases for backwards compatibility.
export {
  Tensor as NDArray,
  Tensor1D as Array1D,
  Tensor2D as Array2D,
  Tensor3D as Array3D,
  Tensor4D as Array4D
};
