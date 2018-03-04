/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as dl from '../../index';
import {CPU_ENVS, describeWithFlags, expectArraysClose} from '../../test_util';
import {TestDataset} from './dataset_test';

describeWithFlags('Dataset.batch()', CPU_ENVS, () => {
  it('batches entries into column-oriented DatasetBatches', done => {
    const ds = new TestDataset();
    const bds = ds.batch(8);
    const batchStreamPromise = bds.getStream();
    batchStreamPromise
        .then(batchStream => batchStream.collectRemaining().then(result => {
          expect(result.length).toEqual(13);
          result.slice(0, 12).forEach(batch => {
            expect((batch['number'] as dl.Tensor).shape).toEqual([8]);
            expect((batch['numberArray'] as dl.Tensor).shape).toEqual([8, 3]);
            expect((batch['Tensor'] as dl.Tensor).shape).toEqual([8, 3]);
            expect((batch['string'] as string[]).length).toEqual(8);
          });
          return result;
        }))
        .then((result) => {
          result.forEach(dl.dispose);
        })
        .then(() => expect(dl.ENV.engine.memory().numTensors).toBe(0))
        .then(done)
        .catch(done.fail);
  });
  it('creates a small last batch', done => {
    const ds = new TestDataset();
    const bds = ds.batch(8);
    const batchStreamPromise = bds.getStream();
    batchStreamPromise
        .then(batchStream => batchStream.collectRemaining().then(result => {
          const lastBatch = result[12];
          expect((lastBatch['number'] as dl.Tensor).shape).toEqual([4]);
          expect((lastBatch['numberArray'] as dl.Tensor).shape).toEqual([4, 3]);
          expect((lastBatch['Tensor'] as dl.Tensor).shape).toEqual([4, 3]);
          expect((lastBatch['string'] as string[]).length).toEqual(4);

          expectArraysClose(
              lastBatch['number'] as dl.Tensor,
              dl.Tensor1D.new([96, 97, 98, 99]));
          expectArraysClose(
              lastBatch['numberArray'] as dl.Tensor, dl.Tensor2D.new([4, 3], [
                [96, 96 ** 2, 96 ** 3], [97, 97 ** 2, 97 ** 3],
                [98, 98 ** 2, 98 ** 3], [99, 99 ** 2, 99 ** 3]
              ]));
          expectArraysClose(
              lastBatch['Tensor'] as dl.Tensor, dl.Tensor2D.new([4, 3], [
                [96, 96 ** 2, 96 ** 3], [97, 97 ** 2, 97 ** 3],
                [98, 98 ** 2, 98 ** 3], [99, 99 ** 2, 99 ** 3]
              ]));
          expect(lastBatch['string'] as string[]).toEqual([
            'Item 96', 'Item 97', 'Item 98', 'Item 99'
          ]);

          expect(lastBatch['string'] as string[]).toEqual([
            'Item 96', 'Item 97', 'Item 98', 'Item 99'
          ]);
          return result;
        }))
        .then((result) => {
          result.forEach(dl.dispose);
        })
        // these three tensors are just the expected results above
        .then(() => expect(dl.ENV.engine.memory().numTensors).toBe(3))
        .then(done)
        .catch(done.fail);
  });
});
