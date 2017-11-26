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

import * as test_util from '../test_util';
import {MPRandGauss} from './rand';

function isFloat(n: number): boolean {
  return Number(n) === n && n % 1 !== 0;
}

test_util.describeCustom('MPRandGauss', () => {
  const EPSILON_FLOAT32 = 0.05;
  const EPSILON_NONFLOAT = 0.10;

  it('should default to float32 numbers', () => {
    const rand = new MPRandGauss(0, 1.5);
    expect(isFloat(rand.nextValue())).toBe(true);
  });

  it('should handle create a mean/stdv of float32 numbers', () => {
    const rand = new MPRandGauss(0, 1.5, 'float32');
    const values = [];
    const size = 10000;
    for (let i = 0; i < size; i++) {
      values.push(rand.nextValue());
    }
    test_util.expectArrayInMeanStdRange(values, 0, 1.5, EPSILON_FLOAT32);
    test_util.jarqueBeraNormalityTest(values);
  });

  it('should handle int32 numbers', () => {
    const rand = new MPRandGauss(0, 1, 'int32');
    expect(isFloat(rand.nextValue())).toBe(false);
  });

  it('should handle create a mean/stdv of float32 numbers', () => {
    const rand = new MPRandGauss(0, 1, 'int32');
    const values = [];
    const size = 1000;
    for (let i = 0; i < size; i++) {
      values.push(rand.nextValue());
    }
    test_util.expectArrayInMeanStdRange(values, 0, 1, EPSILON_NONFLOAT);
    test_util.jarqueBeraNormalityTest(values);
  });
});
