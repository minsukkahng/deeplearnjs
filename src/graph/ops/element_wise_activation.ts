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

import {keep, tidy} from '../../globals';
import {NDArrayMath} from '../../math';
// tslint:disable-next-line:max-line-length
import {ActivationFunction, EluFunc, LeakyReluFunc, ReLUFunc, SigmoidFunc, SquareFunc, TanHFunc} from '../activation_functions';
import {SymbolicTensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';
import {Operation} from './op';

/**
 * @hidden
 */
export class ElementWiseActivation extends Operation {
  constructor(
      protected xTensor: SymbolicTensor, protected yTensor: SymbolicTensor,
      private func: ActivationFunction) {
    super();
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x = inferenceArrays.get(this.xTensor);

    tidy(() => {
      inferenceArrays.set(this.yTensor, keep(this.func.output(math, x)));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    // dE/dx_i = sum_j dE/dy_j * dy_j/dx_i
    //         = dE/dy_i * dy_i/dx_i
    const x = inferenceArrays.get(this.xTensor);
    const y = inferenceArrays.get(this.yTensor);
    const dy = gradientArrays.get(this.yTensor);

    tidy(() => {
      const dydx = this.func.der(math, x, y);
      gradientArrays.add(this.xTensor, math.elementWiseMul(dy, dydx));
      dydx.dispose();
    });
  }

  dispose() {
    this.func.dispose();
  }
}

/**
 * @hidden
 */
export class ReLU extends ElementWiseActivation {
  constructor(xTensor: SymbolicTensor, yTensor: SymbolicTensor) {
    super(xTensor, yTensor, new ReLUFunc());
  }
}

/**
 * @hidden
 */
export class LeakyReLU extends ElementWiseActivation {
  constructor(xTensor: SymbolicTensor, yTensor: SymbolicTensor, alpha: number) {
    super(xTensor, yTensor, new LeakyReluFunc(alpha));
  }
}

/**
 * @hidden
 */
export class TanH extends ElementWiseActivation {
  constructor(xTensor: SymbolicTensor, yTensor: SymbolicTensor) {
    super(xTensor, yTensor, new TanHFunc());
  }
}

/**
 * @hidden
 */
export class Sigmoid extends ElementWiseActivation {
  constructor(xTensor: SymbolicTensor, yTensor: SymbolicTensor) {
    super(xTensor, yTensor, new SigmoidFunc());
  }
}

/**
 * @hidden
 */
export class Square extends ElementWiseActivation {
  constructor(xTensor: SymbolicTensor, yTensor: SymbolicTensor) {
    super(xTensor, yTensor, new SquareFunc());
  }
}

/**
 * @hidden
 */
export class Elu extends ElementWiseActivation {
  constructor(xTensor: SymbolicTensor, yTensor: SymbolicTensor) {
    super(xTensor, yTensor, new EluFunc());
  }
}

/**
 * @hidden
 */
export class PReLU extends Operation {
  constructor(
      private xTensor: SymbolicTensor, private alphaTensor: SymbolicTensor,
      private yTensor: SymbolicTensor) {
    super();
  }
  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x = inferenceArrays.get(this.xTensor);
    const alpha = inferenceArrays.get(this.alphaTensor);
      tidy(() => {
      inferenceArrays.set(this.yTensor, keep(math.prelu(x, alpha)));
    });
  }
  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    throw new Error('Not implemented');
  }
}
