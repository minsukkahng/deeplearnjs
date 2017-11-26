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

export interface RandGauss { nextValue(): number; }

export interface RandNormalDataTypes {
  float32: Float32Array;
  int32: Int32Array;
}

// https://en.wikipedia.org/wiki/Marsaglia_polar_method
export class MPRandGauss implements RandGauss {
  private mean: number;
  private stdDev: number;
  private nextVal: number;
  private dtype?: keyof RandNormalDataTypes;

  constructor(
      mean: number, stdDeviation: number, dtype?: keyof RandNormalDataTypes) {
    this.mean = mean;
    this.stdDev = stdDeviation;
    this.dtype = dtype;
    this.nextVal = NaN;
  }

  /** Returns next sample from a gaussian distribution. */
  public nextValue(): number {
    if (!isNaN(this.nextVal)) {
      const value = this.nextVal;
      this.nextVal = NaN;
      return value;
    }

    let v1: number, v2: number, s: number;
    do {
      v1 = 2 * Math.random() - 1;
      v2 = 2 * Math.random() - 1;
      s = v1 * v1 + v2 * v2;
    } while (s > 1);

    const resultX = Math.sqrt(-2 * Math.log(s) / s) * v1;
    const resultY = Math.sqrt(-2 * Math.log(s) / s) * v2;

    // TODO(kreeger): Handle truncated random generation.
    this.nextVal = this.convertValue(this.mean + this.stdDev * resultY);
    return this.convertValue(this.mean + this.stdDev * resultX);
  }

  /** Handles proper rounding for non floating point numbers. */
  private convertValue(value: number): number {
    if (this.dtype == null || this.dtype === 'float32') {
      return value;
    }
    return Math.round(value);
  }
}
