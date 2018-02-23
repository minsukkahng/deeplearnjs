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
import {Tensor} from '../tensor';
import {expectArraysClose} from '../test_util';
import {InMemoryDataset} from './dataset';

class StubDataset extends InMemoryDataset {
  constructor(data: Tensor[][]) {
    super(data.map(value => value[0].shape));
    this.dataset = data;
  }

  fetchData(): Promise<void> {
    return new Promise<void>((resolve, reject) => {});
  }
}

describe('Dataset', () => {
  it('normalize', () => {
    const data = [
      [
        dl.tensor2d([1, 2, 10, -1, -2, .75], [2, 3]),
        dl.tensor2d([2, 3, 20, -2, 2, .5], [2, 3]),
        dl.tensor2d([3, 4, 30, -3, -4, 0], [2, 3]),
        dl.tensor2d([4, 5, 40, -4, 4, 1], [2, 3])
      ],
      [
        dl.randomNormal([1]), dl.randomNormal([1]), dl.randomNormal([1]),
        dl.randomNormal([1])
      ]
    ];
    const dataset = new StubDataset(data);

    // Normalize only the first data index.
    const dataIndex = 0;
    dataset.normalizeWithinBounds(dataIndex, 0, 1);

    let normalizedInputs = dataset.getData()[0];

    expectArraysClose(normalizedInputs[0], [0, 0, 0, 1, .25, .75]);
    expectArraysClose(
        normalizedInputs[1], [1 / 3, 1 / 3, 1 / 3, 2 / 3, .75, .5]);
    expectArraysClose(normalizedInputs[2], [2 / 3, 2 / 3, 2 / 3, 1 / 3, 0, 0]);
    expectArraysClose(normalizedInputs[3], [1, 1, 1, 0, 1, 1]);

    dataset.normalizeWithinBounds(dataIndex, -1, 1);

    normalizedInputs = dataset.getData()[0];

    expectArraysClose(normalizedInputs[0], [-1, -1, -1, 1, -.5, .5]);
    expectArraysClose(
        normalizedInputs[1], [-1 / 3, -1 / 3, -1 / 3, 1 / 3, .5, .0]);
    expectArraysClose(
        normalizedInputs[2], [1 / 3, 1 / 3, 1 / 3, -1 / 3, -1, -1]);
    expectArraysClose(normalizedInputs[3], [1, 1, 1, -1, 1, 1]);

    dataset.removeNormalization(dataIndex);

    normalizedInputs = dataset.getData()[0];

    expectArraysClose(normalizedInputs[0], [1, 2, 10, -1, -2, .75]);
    expectArraysClose(normalizedInputs[1], [2, 3, 20, -2, 2, .5]);
    expectArraysClose(normalizedInputs[2], [3, 4, 30, -3, -4, 0]);
    expectArraysClose(normalizedInputs[3], [4, 5, 40, -4, 4, 1]);
  });
});
