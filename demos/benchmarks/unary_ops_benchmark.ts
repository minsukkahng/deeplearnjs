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

import * as dl from 'deeplearn';

import {BenchmarkTest} from './benchmark';
import * as benchmark_util from './benchmark_util';

function getUnaryOp(option: string) {
  switch (option) {
    case 'log':
      return (x: dl.Tensor) => x.log();
    case 'exp':
      return (x: dl.Tensor) => x.exp();
    case 'neg':
      return (x: dl.Tensor) => x.neg();
    case 'sqrt':
      return (x: dl.Tensor) => x.sqrt();
    case 'abs':
      return (x: dl.Tensor) => x.abs();
    case 'relu':
      return (x: dl.Tensor) => x.relu();
    case 'sigmoid':
      return (x: dl.Tensor) => x.sigmoid();
    case 'sin':
      return (x: dl.Tensor) => x.sin();
    case 'cos':
      return (x: dl.Tensor) => x.cos();
    case 'tan':
      return (x: dl.Tensor) => x.tan();
    case 'asin':
      return (x: dl.Tensor) => x.asin();
    case 'acos':
      return (x: dl.Tensor) => x.acos();
    case 'atan':
      return (x: dl.Tensor) => x.atan();
    case 'sinh':
      return (x: dl.Tensor) => x.sinh();
    case 'cosh':
      return (x: dl.Tensor) => x.cosh();
    case 'tanh':
      return (x: dl.Tensor) => x.tanh();
    default:
      throw new Error(`Not found such ops: ${option}`);
  }
}

export class UnaryOpsCPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string): Promise<number> {
    const safeMode = false;
    const math = new dl.NDArrayMath('cpu', safeMode);
    dl.ENV.setMath(math);

    const input: dl.Tensor2D = dl.randomUniform([size, size], -1, 1);
    const op = getUnaryOp(option);
    const start = performance.now();

    dl.tidy(() => {
      op(input).get();
    });

    math.dispose();

    const end = performance.now();
    return end - start;
  }
}

export class UnaryOpsGPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string) {
    const safeMode = false;
    const math = new dl.NDArrayMath('webgl', safeMode);
    dl.ENV.setMath(math);

    const input: dl.Tensor2D = dl.randomUniform([size, size], -1, 1);
    const op = getUnaryOp(option);

    const benchmark = () => op(input);

    const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

    input.dispose();
    math.dispose();

    return time;
  }
}
