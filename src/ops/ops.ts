/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ArrayOps} from './array_ops';
import {BatchNormOps} from './batchnorm';
import {BinaryOps} from './binary_ops';
import {CompareOps} from './compare';
import {ConcatOps} from './concat';
import {ConvOps} from './conv';
import {ImageOps} from './image_ops';
import {LogicalOps} from './logical_ops';
import {LRNOps} from './lrn';
import {LSTMOps} from './lstm';
import {MatmulOps} from './matmul';
import {NormOps} from './norm';
import {PoolOps} from './pool';
import {ReductionOps} from './reduction_ops';
import {ReverseOps} from './reverse';
import {SliceOps} from './slice';
import {SoftmaxOps} from './softmax';
import {TransposeOps} from './transpose';
import {UnaryOps} from './unary_ops';

export const batchNormalization = BatchNormOps.batchNormalization;
export const batchNormalization2d = BatchNormOps.batchNormalization2d;
export const batchNormalization3d = BatchNormOps.batchNormalization3d;
export const batchNormalization4d = BatchNormOps.batchNormalization4d;

export const concat = ConcatOps.concat;
export const concat1d = ConcatOps.concat1d;
export const concat2d = ConcatOps.concat2d;
export const concat3d = ConcatOps.concat3d;
export const concat4d = ConcatOps.concat4d;

export const conv1d = ConvOps.conv1d;
export const conv2d = ConvOps.conv2d;
export const conv2dTranspose = ConvOps.conv2dTranspose;
export const depthwiseConv2d = ConvOps.depthwiseConv2d;

export const matMul = MatmulOps.matMul;
export const matrixTimesVector = MatmulOps.matrixTimesVector;
export const outerProduct = MatmulOps.outerProduct;
export const vectorTimesMatrix = MatmulOps.vectorTimesMatrix;

export const avgPool = PoolOps.avgPool;
export const maxPool = PoolOps.maxPool;
export const minPool = PoolOps.minPool;

export const transpose = TransposeOps.transpose;

export const reverse = ReverseOps.reverse;
export const reverse1d = ReverseOps.reverse1d;
export const reverse2d = ReverseOps.reverse2d;
export const reverse3d = ReverseOps.reverse3d;
export const reverse4d = ReverseOps.reverse4d;

export const slice = SliceOps.slice;
export const slice1d = SliceOps.slice1d;
export const slice2d = SliceOps.slice2d;
export const slice3d = SliceOps.slice3d;
export const slice4d = SliceOps.slice4d;

export const argMax = ReductionOps.argMax;
export const argMin = ReductionOps.argMin;
export const logSumExp = ReductionOps.logSumExp;
export const max = ReductionOps.max;
export const mean = ReductionOps.mean;
export const min = ReductionOps.min;
export const moments = ReductionOps.moments;
export const sum = ReductionOps.sum;

export const equal = CompareOps.equal;
export const equalStrict = CompareOps.equalStrict;
export const greater = CompareOps.greater;
export const greaterStrict = CompareOps.greaterStrict;
export const greaterEqual = CompareOps.greaterEqual;
export const greaterEqualStrict = CompareOps.greaterEqualStrict;
export const less = CompareOps.less;
export const lessStrict = CompareOps.lessStrict;
export const lessEqual = CompareOps.lessEqual;
export const lessEqualStrict = CompareOps.lessEqualStrict;
export const notEqual = CompareOps.notEqual;
export const notEqualStrict = CompareOps.notEqualStrict;

export const logicalNot = LogicalOps.logicalNot;
export const logicalAnd = LogicalOps.logicalAnd;
export const logicalOr = LogicalOps.logicalOr;
export const logicalXor = LogicalOps.logicalXor;
export const where = LogicalOps.where;

export const abs = UnaryOps.abs;
export const acos = UnaryOps.acos;
export const asin = UnaryOps.asin;
export const atan = UnaryOps.atan;
export const ceil = UnaryOps.ceil;
export const clipByValue = UnaryOps.clipByValue;
export const cos = UnaryOps.cos;
export const cosh = UnaryOps.cosh;
export const elu = UnaryOps.elu;
export const exp = UnaryOps.exp;
export const floor = UnaryOps.floor;
export const leakyRelu = UnaryOps.leakyRelu;
export const log = UnaryOps.log;
export const neg = UnaryOps.neg;
export const prelu = UnaryOps.prelu;
export const relu = UnaryOps.relu;
export const selu = UnaryOps.selu;
export const sigmoid = UnaryOps.sigmoid;
export const sin = UnaryOps.sin;
export const sinh = UnaryOps.sinh;
export const sqrt = UnaryOps.sqrt;
export const square = UnaryOps.square;
export const step = UnaryOps.step;
export const tan = UnaryOps.tan;
export const tanh = UnaryOps.tanh;

export const add = BinaryOps.add;
export const addStrict = BinaryOps.addStrict;
export const div = BinaryOps.div;
export const divStrict = BinaryOps.divStrict;
export const maximum = BinaryOps.maximum;
export const maximumStrict = BinaryOps.maximumStrict;
export const minimum = BinaryOps.minimum;
export const minimumStrict = BinaryOps.minimumStrict;
export const mul = BinaryOps.mul;
export const mulStrict = BinaryOps.mulStrict;
export const pow = BinaryOps.pow;
export const powStrict = BinaryOps.powStrict;
export const sub = BinaryOps.sub;
export const subStrict = BinaryOps.subStrict;

export const norm = NormOps.norm;

export const cast = ArrayOps.cast;
export const clone = ArrayOps.clone;
export const fromPixels = ArrayOps.fromPixels;
export const ones = ArrayOps.ones;
export const onesLike = ArrayOps.onesLike;
export const zeros = ArrayOps.zeros;
export const zerosLike = ArrayOps.zerosLike;
export const rand = ArrayOps.rand;
export const randomNormal = ArrayOps.randomNormal;
export const truncatedNormal = ArrayOps.truncatedNormal;
export const randomUniform = ArrayOps.randomUniform;
export const reshape = ArrayOps.reshape;
export const squeeze = ArrayOps.squeeze;
export const tile = ArrayOps.tile;
export const gather = ArrayOps.gather;
export const oneHot = ArrayOps.oneHot;
export const linspace = ArrayOps.linspace;
export const range = ArrayOps.range;
export const buffer = ArrayOps.buffer;
export const fill = ArrayOps.fill;
export const tensor = ArrayOps.tensor;
export const scalar = ArrayOps.scalar;
export const tensor1d = ArrayOps.tensor1d;
export const tensor2d = ArrayOps.tensor2d;
export const tensor3d = ArrayOps.tensor3d;
export const tensor4d = ArrayOps.tensor4d;
export const print = ArrayOps.print;
export const expandDims = ArrayOps.expandDims;
export const stack = ArrayOps.stack;

export const pad = ArrayOps.pad;
export const pad1d = ArrayOps.pad1d;
export const pad2d = ArrayOps.pad2d;
export const pad3d = ArrayOps.pad3d;
export const pad4d = ArrayOps.pad4d;

export const basicLSTMCell = LSTMOps.basicLSTMCell;
export const multiRNNCell = LSTMOps.multiRNNCell;

export const softmax = SoftmaxOps.softmax;

export const localResponseNormalization = LRNOps.localResponseNormalization;

// So typings can propagate.
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {Rank} from '../types';
// tslint:disable-next-line:no-unused-expression
[Tensor, Rank, Tensor3D, Tensor4D];

export const losses = {
  softmaxCrossEntropy: SoftmaxOps.softmaxCrossEntropy
};

export const image = {
  resizeBilinear: ImageOps.resizeBilinear
};
