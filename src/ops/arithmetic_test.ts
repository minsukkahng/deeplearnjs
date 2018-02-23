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

import * as dl from '../index';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('div', ALL_ENVS, () => {
  it('same shape', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const c = dl.tensor2d([1, 2, 3, 4, 2, 5], [2, 3]);

    const r = dl.div(a, c);

    expectArraysClose(r, [1, 1, 1, 1, 2.5, 6 / 5]);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor2d([1, 2], [2, 1]);
    const c = dl.tensor2d([3, NaN], [2, 1]);

    const r = dl.div(a, c);

    expectArraysClose(r, [1 / 3, NaN]);
  });

  it('broadcasting same rank Tensors different shape', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor2d([2, 3], [2, 1]);

    const result = dl.div(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [1 / 2, 1, -1, -4 / 3];

    expectArraysClose(result, expected);
  });

  it('broadcast 2D + 1D', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor1d([1, 2]);

    const result = dl.div(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [1, 1, -3, -2];

    expectArraysClose(result, expected);
  });

  it('throws when passed tensors of different shapes', () => {
    const a = dl.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = dl.tensor2d([5, 3, 4, -7], [2, 2]);

    expect(() => dl.div(a, b)).toThrowError();
    expect(() => dl.div(b, a)).toThrowError();
  });

  it('scalar divided by array', () => {
    const c = dl.scalar(2);
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);

    const r = dl.div(c, a);

    expectArraysClose(r, [2 / 1, 2 / 2, 2 / 3, 2 / 4, 2 / 5, 2 / 6]);
  });

  it('scalar divided by array propagates NaNs', () => {
    const c = dl.scalar(NaN);
    const a = dl.tensor2d([1, 2, 3], [1, 3]);

    const r = dl.div(c, a);

    expectArraysEqual(r, [NaN, NaN, NaN]);
  });

  it('array divided by scalar', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const c = dl.scalar(2);

    const r = dl.div(a, c);

    expectArraysClose(r, [1 / 2, 2 / 2, 3 / 2, 4 / 2, 5 / 2, 6 / 2]);
  });

  it('array divided by scalar propagates NaNs', () => {
    const a = dl.tensor2d([1, 2, NaN], [1, 3]);
    const c = dl.scalar(2);

    const r = dl.div(a, c);
    expectArraysClose(r, [1 / 2, 2 / 2, NaN]);
  });

  it('gradient: Scalar', () => {
    const a = dl.scalar(5);
    const b = dl.scalar(2);
    const dy = dl.scalar(4);

    const grads = dl.grads((a, b) => dl.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [4 / 2]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-4 * 5 / (2 * 2)]);
  });

  it('gradient: Tensor1D', () => {
    const a = dl.tensor1d([1, 2, 3]);
    const b = dl.tensor1d([3, 4, 5]);
    const dy = dl.tensor1d([1, 10, 20]);

    const grads = dl.grads((a, b) => dl.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(da, [1 / 3, 10 / 4, 20 / 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-1 * 1 / 9, -10 * 2 / 16, -20 * 3 / 25]);
  });

  it('gradient: Tensor1D with int32', () => {
    const a = dl.tensor1d([1, 2, 3], 'int32');
    const b = dl.tensor1d([3, 4, 5], 'int32');
    const dy = dl.tensor1d([1, 10, 20]);

    const grads = dl.grads((a, b) => dl.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(da, [1 / 3, 10 / 4, 20 / 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-1 * 1 / 9, -10 * 2 / 16, -20 * 3 / 25]);
  });

  it('gradient: 1d<int32> with 1d<bool> ', () => {
    const a = dl.tensor1d([true, false, true], 'bool');
    const b = dl.tensor1d([1, 2, 3], 'int32');
    const dy = dl.tensor1d([1, 19, 20]);

    const grads = dl.grads((a, b) => dl.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(da, [1, 19 / 2, 20 / 3]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-1 / 1, 0, -20 / 9]);
  });

  it('gradient: Tensor2D', () => {
    const a = dl.tensor2d([3, 1, 2, 3], [2, 2]);
    const b = dl.tensor2d([1, 3, 4, 5], [2, 2]);
    const dy = dl.tensor2d([1, 10, 15, 20], [2, 2]);

    const grads = dl.grads((a, b) => dl.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 / 1, 10 / 3, 15 / 4, 20 / 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        db, [-1 * 3 / 1, -10 * 1 / 9, -15 * 2 / 16, -20 * 3 / 25]);
  });

  it('gradient: scalar / Tensor1D', () => {
    const a = dl.scalar(2);
    const b = dl.tensor1d([3, 4, 5]);
    const dy = dl.tensor1d([6, 7, 8]);

    const grads = dl.grads((a, b) => dl.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [6 / 3 + 7 / 4 + 8 / 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-6 * 2 / 9, -7 * 2 / 16, -8 * 2 / 25]);
  });

  it('gradient: Tensor2D / scalar', () => {
    const a = dl.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const b = dl.scalar(2);
    const dy = dl.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = dl.grads((a, b) => dl.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [6 / 2, 7 / 2, 8 / 2, 9 / 2]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-6 * 2 / 4 + -7 * 3 / 4 + -8 * 4 / 4 + -9 * 5 / 4]);
  });

  it('gradient: Tensor2D / Tensor2D w/ broadcast', () => {
    const a = dl.tensor2d([3, 4], [2, 1]);
    const b = dl.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const dy = dl.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = dl.grads((a, b) => dl.div(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [6 / 2 + 7 / 3, 8 / 4 + 9 / 5]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-6 * 3 / 4, -7 * 3 / 9, -8 * 4 / 16, -9 * 4 / 25]);
  });
});

describeWithFlags('mul', ALL_ENVS, () => {
  it('strict same-shaped tensors', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor2d([5, 3, 4, -7], [2, 2]);
    const expected = [5, 6, -12, 28];
    const result = dl.mulStrict(a, b);

    expect(result.shape).toEqual([2, 2]);
    expect(result.dtype).toBe('float32');
    expectArraysClose(result, expected);
  });

  it('strict propagates NaNs', () => {
    const a = dl.tensor2d([1, 3, 4, 0], [2, 2]);
    const b = dl.tensor2d([NaN, 3, NaN, 3], [2, 2]);

    const result = dl.mulStrict(a, b);

    expect(result.dtype).toBe('float32');
    expectArraysClose(result, [NaN, 9, NaN, 0]);
  });

  it('strict throws when passed tensors of different shapes', () => {
    const a = dl.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = dl.tensor2d([5, 3, 4, -7], [2, 2]);

    expect(() => dl.mulStrict(a, b)).toThrowError();
    expect(() => dl.mulStrict(b, a)).toThrowError();
  });

  it('strict throws when dtypes do not match', () => {
    const a = dl.tensor2d([1, 2, -3, -4, 5, 6], [2, 3], 'float32');
    const b = dl.tensor2d([5, 3, 4, -7], [2, 2], 'int32');

    expect(() => dl.mulStrict(a, b)).toThrowError();
    expect(() => dl.mulStrict(b, a)).toThrowError();
  });

  it('strict int32 * int32', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2], 'int32');
    const b = dl.tensor2d([2, 1, 3, -4], [2, 2], 'int32');
    const res = dl.mulStrict(a, b);

    expect(res.dtype).toBe('int32');
    expectArraysClose(res, [2, 2, -9, 16]);
  });

  it('same-shaped tensors', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor2d([5, 3, 4, -7], [2, 2]);
    const expected = [5, 6, -12, 28];
    const result = dl.mul(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, expected);
  });

  it('broadcasting tensors', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.scalar(2);
    const expected = [2, 4, -6, -8];
    const result = dl.mul(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, expected);
  });

  it('broadcasting same rank Tensors different shape', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor2d([2, 3], [2, 1]);

    const result = dl.mul(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [2, 4, -9, -12];

    expectArraysClose(result, expected);
  });

  it('broadcast 2D + 1D', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor1d([1, 2]);

    const result = dl.mul(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [1, 4, -3, -8];

    expectArraysClose(result, expected);
  });

  it('gradient: Scalar', () => {
    const a = dl.scalar(5);
    const b = dl.scalar(2);
    const dy = dl.scalar(4);

    const grads = dl.grads((a, b) => dl.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [b.get() * dy.get()]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [a.get() * dy.get()]);
  });

  it('gradient: Tensor1D', () => {
    const a = dl.tensor1d([1, 2, 3]);
    const b = dl.tensor1d([3, 4, 5]);
    const dy = dl.tensor1d([1, 10, 20]);

    const grads = dl.grads((a, b) => dl.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [3 * 1, 4 * 10, 5 * 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [1 * 1, 2 * 10, 3 * 20]);
  });

  it('gradient: Tensor1D with dtype int32', () => {
    const a = dl.tensor1d([1, 2, 3], 'int32');
    const b = dl.tensor1d([3, 4, 5], 'int32');
    const dy = dl.tensor1d([1, 10, 20]);

    const grads = dl.grads((a, b) => dl.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(da, [3 * 1, 4 * 10, 5 * 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [1 * 1, 2 * 10, 3 * 20]);
  });

  it('gradient: Tensor2D', () => {
    const a = dl.tensor2d([3, 1, 2, 3], [2, 2]);
    const b = dl.tensor2d([1, 3, 4, 5], [2, 2]);
    const dy = dl.tensor2d([1, 10, 15, 20], [2, 2]);

    const grads = dl.grads((a, b) => dl.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 * 1, 3 * 10, 4 * 15, 5 * 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [3 * 1, 1 * 10, 2 * 15, 3 * 20]);
  });

  it('gradient: scalar * Tensor1D', () => {
    const a = dl.scalar(2);
    const b = dl.tensor1d([3, 4, 5]);
    const dy = dl.tensor1d([6, 7, 8]);

    const grads = dl.grads((a, b) => dl.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [3 * 6 + 4 * 7 + 5 * 8]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [2 * 6, 2 * 7, 2 * 8]);
  });

  it('gradient: Tensor2D * scalar', () => {
    const a = dl.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const b = dl.scalar(2);
    const dy = dl.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = dl.grads((a, b) => dl.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [2 * 6, 2 * 7, 2 * 8, 2 * 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [2 * 6 + 3 * 7 + 4 * 8 + 5 * 9]);
  });

  it('gradient: Tensor2D * Tensor2D w/ broadcast', () => {
    const a = dl.tensor2d([3, 4], [2, 1]);
    const b = dl.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const dy = dl.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = dl.grads((a, b) => dl.mul(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [2 * 6 + 3 * 7, 4 * 8 + 5 * 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [6 * 3, 7 * 3, 8 * 4, 9 * 4]);
  });
});

describeWithFlags('pow', ALL_ENVS, () => {
  it('same-shaped tensors', () => {
    const a = dl.tensor2d([1, -2, -3, 0, 7, 1], [2, 3]);
    const b = dl.tensor2d([5, 3, 4, 5, 2, -3], [2, 3], 'int32');
    const expected = [1, -8, 81, 0, 49, 1];
    const result = dl.pow(a, b);

    expect(result.shape).toEqual([2, 3]);
    expectArraysClose(result, expected, 0.01);
  });

  it('int32^int32 returns int32', () => {
    const a = dl.tensor1d([1, 2, 3], 'int32');
    const exp = dl.scalar(2, 'int32');

    const result = dl.pow(a, exp);

    expect(result.shape).toEqual([3]);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [1, 4, 9]);
  });

  it('different-shaped tensors', () => {
    const a = dl.tensor2d([1, -2, -3, 0, 7, 1], [2, 3]);
    const b = dl.scalar(2, 'int32');
    const expected = [1, 4, 9, 0, 49, 1];
    const result = dl.pow(a, b);

    expect(result.shape).toEqual([2, 3]);
    expectArraysClose(result, expected, 0.05);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor2d([NaN, 3, NaN, 0], [2, 2]);
    const b = dl.tensor2d([1, 3, 2, 3], [2, 2], 'int32');

    const result = dl.pow(a, b);
    expectArraysClose(result, [NaN, 27, NaN, 0], 0.05);
  });

  it('throws when passed non int32 exponent param', () => {
    const a = dl.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = dl.tensor2d([5, 3, 4, -7], [2, 2], 'float32');

    // tslint:disable-next-line
    expect(() => dl.pow(a, b as any)).toThrowError();
  });

  it('broadcasting same rank Tensors different shape', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor2d([2, 1], [2, 1], 'int32');

    const result = dl.pow(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [1, 4, -3, -4];

    expectArraysClose(result, expected);
  });

  it('broadcast 2D + 1D', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor1d([1, 2], 'int32');

    const result = dl.pow(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [1, 4, -3, 16];

    expectArraysClose(result, expected);
  });

  it('powStrict same-shaped tensors', () => {
    const a = dl.tensor2d([1, -2, -3, 0, 7, 1], [2, 3]);
    const b = dl.tensor2d([5, 3, 4, 5, 2, -3], [2, 3], 'int32');
    const expected = [1, -8, 81, 0, 49, 1];
    const result = dl.powStrict(a, b);

    expect(result.shape).toEqual([2, 3]);
    expectArraysClose(result, expected, 0.01);
  });

  it('powStrict throws when passed tensors of different shapes', () => {
    const a = dl.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = dl.tensor2d([5, 3, 4, -7], [2, 2], 'int32');

    expect(() => dl.powStrict(a, b)).toThrowError();
  });

  it('powStrict throws when passed non int32 exponent param', () => {
    const a = dl.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = dl.tensor2d([5, 3, 4, -7], [2, 2], 'float32');

    // tslint:disable-next-line
    expect(() => dl.powStrict(a, b as any)).toThrowError();
  });

  it('gradients: Scalar ^ Scalar', () => {
    const a = dl.scalar(5);
    const b = dl.scalar(2, 'int32');
    const dy = dl.scalar(3);

    const grad = dl.grad(a => dl.pow(a, b));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [2 * 5 * 3]);
  });

  it('gradients: Tensor ^ Tensor', () => {
    const a = dl.tensor1d([-1, .5, 2]);
    const b = dl.tensor1d([3, 2, -1], 'int32');
    const dy = dl.tensor1d([1, 5, 10]);

    const grad = dl.grad(a => dl.pow(a, b));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(
        da,
        [
          3 * Math.pow(-1, 2) * 1, 2 * Math.pow(.5, 1) * 5,
          -1 * Math.pow(2, -2) * 10
        ],
        1e-1);
  });
});

describeWithFlags('add', ALL_ENVS, () => {
  it('c + A', () => {
    const c = dl.scalar(5);
    const a = dl.tensor1d([1, 2, 3]);

    const result = dl.add(c, a);

    expectArraysClose(result, [6, 7, 8]);
  });

  it('c + A propagates NaNs', () => {
    const c = dl.scalar(NaN);
    const a = dl.tensor1d([1, 2, 3]);

    const res = dl.add(c, a);

    expectArraysEqual(res, [NaN, NaN, NaN]);
  });

  it('A + B broadcasting same rank Tensors different shape', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor2d([2, 3], [2, 1]);

    const result = dl.add(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [3, 4, 0, -1];

    expectArraysClose(result, expected);
  });

  it('A + B broadcast 2D + 1D', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor1d([1, 2]);

    const result = dl.add(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [2, 4, -2, -2];

    expectArraysClose(result, expected);
  });

  it('A + B', () => {
    const a = dl.tensor1d([2, 5, 1]);
    const b = dl.tensor1d([4, 2, -1]);

    const result = dl.add(a, b);

    const expected = [6, 7, 0];
    expectArraysClose(result, expected);
  });

  it('A + B propagates NaNs', () => {
    const a = dl.tensor1d([2, 5, NaN]);
    const b = dl.tensor1d([4, 2, -1]);

    const res = dl.add(a, b);
    expectArraysClose(res, [6, 7, NaN]);
  });

  it('A + B throws when passed tensors with different shape', () => {
    const a = dl.tensor1d([2, 5, 1, 5]);
    const b = dl.tensor1d([4, 2, -1]);

    expect(() => dl.add(a, b)).toThrowError();
    expect(() => dl.add(b, a)).toThrowError();
  });

  it('2D+scalar broadcast', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = dl.scalar(2);
    const res = dl.add(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(res, [3, 4, 5, 6, 7, 8]);
  });

  it('scalar+1D broadcast', () => {
    const a = dl.scalar(2);
    const b = dl.tensor1d([1, 2, 3, 4, 5, 6]);
    const res = dl.add(a, b);
    expect(res.shape).toEqual([6]);
    expectArraysClose(res, [3, 4, 5, 6, 7, 8]);
  });

  it('2D+2D broadcast each with 1 dim', () => {
    const a = dl.tensor2d([1, 2, 5], [1, 3]);
    const b = dl.tensor2d([7, 3], [2, 1]);
    const res = dl.add(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(res, [8, 9, 12, 4, 5, 8]);
  });

  it('2D+2D broadcast inner dim of b', () => {
    const a = dl.tensor2d([1, 2, 5, 4, 5, 6], [2, 3]);
    const b = dl.tensor2d([7, 3], [2, 1]);
    const res = dl.add(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(res, [8, 9, 12, 7, 8, 9]);
  });

  it('3D+scalar', () => {
    const a = dl.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
    const b = dl.scalar(-1);
    const res = dl.add(a, b);
    expect(res.shape).toEqual([2, 3, 1]);
    expectArraysClose(res, [0, 1, 2, 3, 4, 5]);
  });

  it('gradient: scalar + 1D broadcast', () => {
    const a = dl.scalar(2);
    const b = dl.tensor1d([3, 4, 5]);
    const dy = dl.tensor1d([7, 8, 9]);

    const grads = dl.grads((a, b) => dl.add(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [7 + 8 + 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [7, 8, 9]);
  });

  it('gradient: 2D + 2D broadcast', () => {
    const a = dl.tensor2d([2, 3], [2, 1]);
    const b = dl.tensor2d([4, 5, 6, 7], [2, 2]);
    const dy = dl.tensor2d([5, 4, 3, 2], [2, 2]);

    const grads = dl.grads((a, b) => dl.add(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [5 + 4, 3 + 2]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [5, 4, 3, 2]);
  });
});

describeWithFlags('sub', ALL_ENVS, () => {
  it('c - A', () => {
    const c = dl.scalar(5);
    const a = dl.tensor1d([7, 2, 3]);

    const result = dl.sub(c, a);

    expectArraysClose(result, [-2, 3, 2]);
  });

  it('A - c', () => {
    const a = dl.tensor1d([1, 2, -3]);
    const c = dl.scalar(5);

    const result = dl.sub(a, c);

    expectArraysClose(result, [-4, -3, -8]);
  });

  it('A - c propagates NaNs', () => {
    const a = dl.tensor1d([1, NaN, 3]);
    const c = dl.scalar(5);

    const res = dl.sub(a, c);

    expectArraysClose(res, [-4, NaN, -2]);
  });

  it('A - B', () => {
    const a = dl.tensor1d([2, 5, 1]);
    const b = dl.tensor1d([4, 2, -1]);

    const result = dl.sub(a, b);

    const expected = [-2, 3, 2];
    expectArraysClose(result, expected);
  });

  it('A - B propagates NaNs', () => {
    const a = dl.tensor1d([2, 5, 1]);
    const b = dl.tensor1d([4, NaN, -1]);

    const res = dl.sub(a, b);

    expectArraysClose(res, [-2, NaN, 2]);
  });

  it('A - B throws when passed tensors with different shape', () => {
    const a = dl.tensor1d([2, 5, 1, 5]);
    const b = dl.tensor1d([4, 2, -1]);

    expect(() => dl.sub(a, b)).toThrowError();
    expect(() => dl.sub(b, a)).toThrowError();
  });

  it('A - B broadcasting same rank Tensors different shape', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor2d([2, 3], [2, 1]);

    const result = dl.sub(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [-1, 0, -6, -7];

    expectArraysClose(result, expected);
  });

  it('A - B broadcast 2D + 1D', () => {
    const a = dl.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = dl.tensor1d([1, 2]);

    const result = dl.sub(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [0, 0, -4, -6];

    expectArraysClose(result, expected);
  });

  it('2D-scalar broadcast', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = dl.scalar(2);
    const res = dl.sub(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(res, [-1, 0, 1, 2, 3, 4]);
  });

  it('scalar-1D broadcast', () => {
    const a = dl.scalar(2);
    const b = dl.tensor1d([1, 2, 3, 4, 5, 6]);
    const res = dl.sub(a, b);
    expect(res.shape).toEqual([6]);
    expectArraysClose(res, [1, 0, -1, -2, -3, -4]);
  });

  it('2D-2D broadcast each with 1 dim', () => {
    const a = dl.tensor2d([1, 2, 5], [1, 3]);
    const b = dl.tensor2d([7, 3], [2, 1]);
    const res = dl.sub(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(res, [-6, -5, -2, -2, -1, 2]);
  });

  it('2D-2D broadcast inner dim of b', () => {
    const a = dl.tensor2d([1, 2, 5, 4, 5, 6], [2, 3]);
    const b = dl.tensor2d([7, 3], [2, 1]);
    const res = dl.sub(a, b);
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(res, [-6, -5, -2, 1, 2, 3]);
  });

  it('3D-scalar', () => {
    const a = dl.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
    const b = dl.scalar(-1);
    const res = dl.sub(a, b);
    expect(res.shape).toEqual([2, 3, 1]);
    expectArraysClose(res, [2, 3, 4, 5, 6, 7]);
  });

  it('gradients: basic 1D arrays', () => {
    const a = dl.tensor1d([1, 2, 3]);
    const b = dl.tensor1d([3, 2, 1]);
    const dy = dl.tensor1d([1, 10, 20]);

    const grads = dl.grads((a, b) => dl.sub(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1, 10, 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-1, -10, -20]);
  });

  it('gradients: basic 2D arrays', () => {
    const a = dl.tensor2d([0, 1, 2, 3], [2, 2]);
    const b = dl.tensor2d([3, 2, 1, 0], [2, 2]);
    const dy = dl.tensor2d([1, 10, 15, 20], [2, 2]);

    const grads = dl.grads((a, b) => dl.sub(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1, 10, 15, 20]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-1, -10, -15, -20]);
  });

  it('gradient: 1D - scalar broadcast', () => {
    const a = dl.tensor1d([3, 4, 5]);
    const b = dl.scalar(2);
    const dy = dl.tensor1d([7, 8, 9]);

    const grads = dl.grads((a, b) => dl.sub(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [7, 8, 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-7 - 8 - 9]);
  });

  it('gradient: scalar - 1D broadcast', () => {
    const a = dl.scalar(2);
    const b = dl.tensor1d([3, 4, 5]);
    const dy = dl.tensor1d([7, 8, 9]);

    const grads = dl.grads((a, b) => dl.sub(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [7 + 8 + 9]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-7, -8, -9]);
  });

  it('gradient: 2D - 2D broadcast', () => {
    const a = dl.tensor2d([4, 5, 6, 7], [2, 2]);
    const b = dl.tensor2d([2, 3], [2, 1]);
    const dy = dl.tensor2d([5, 4, 3, 2], [2, 2]);

    const grads = dl.grads((a, b) => dl.sub(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [5, 4, 3, 2]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-5 - 4, -3 - 2]);
  });
});
