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
 *
 * =============================================================================
 */

import * as seedrandom from 'seedrandom';

import {tidy} from '../../globals';

import {BatchDataset} from './batch_dataset';
import {computeDatasetStatistics, DatasetStatistics} from './statistics';
import {DataStream, streamFromFunction} from './streams/data_stream';
import {streamFromConcatenated} from './streams/data_stream';
import {streamFromItems} from './streams/data_stream';
import {DatasetElement} from './types';

// TODO(soergel): consider vectorized operations within the pipeline.

/**
 * Represents a potentially large set of elements.
 *
 * A `Dataset` can be used to represent an input pipeline as a
 * collection of elements (maps from string keys to values) and a "logical
 * plan" of transformations that act on those elements.
 *
 * A `Dataset` provides a stream of unbatched examples, and its transformations
 * are applied one example at a time.  Batching produces a BatchDataset, and so
 * must come last in the pipeline because there are (so far) no batch-enabled
 * transformations.
 */
export abstract class Dataset {
  /*
   * Provide a new stream of elements.  Note this will also start new streams
   * from any underlying `Dataset`s.
   *
   * CAUTION: Any Tensors contained within the DatasetElements returned from
   * this stream *must* be manually disposed to avoid a GPU memory leak.
   * The dl.tidy() approach cannot be used in a asynchronous context.
   */
  abstract getStream(): DataStream<DatasetElement>;

  // TODO(soergel): Make Datasets report whether repeated getStream() calls
  // produce the same result (e.g., reading from a file) or different results
  // (e.g., from the webcam).  Currently we don't make this distinction but it
  // could be important for the user to know.
  // abstract isDeterministic(): boolean;

  // TODO(soergel): memoize computeStatistics()

  /**
   * Gathers statistics from a Dataset (or optionally from a sample).
   *
   * This obtains a stream from the Dataset and, by default, does a full pass
   * to gather the statistics.
   *
   * Statistics may be computed over a sample.  However: simply taking the first
   * n items from the stream may produce a poor estimate if the stream is
   * ordered in some way.
   *
   * A truly random shuffle of the stream would of course solve this
   * problem, but many streams do not allow for this, instead providing only a
   * sliding-window shuffle.  A partially-randomized sample could be obtained by
   * shuffling over a window followed by taking the first n samples (where n is
   * smaller than the shuffle window size).  However there is little point in
   * using that approach here, because the cost is likely dominated by obtaining
   * the data.  Thus, once we have filled our shuffle buffer, we may as well use
   * all of that data instead of sampling from it.
   *
   * @param sampleSize The number of examples to take from the (possibly
   *   shuffled) stream.
   * @param shuffleWindowSize The size of the shuffle window to use, if any.
   *   (Not recommended, as described above).
   */
  async computeStatistics(sampleSize?: number, shuffleWindowSize?: number):
      Promise<DatasetStatistics> {
    return computeDatasetStatistics(this, sampleSize, shuffleWindowSize);
  }

  /**
   * Filters this dataset according to `predicate`.
   *
   * @param predicate A function mapping a `DatasetElement` to a boolean or a
   * `Promise` for one.
   *
   * @returns A `Dataset` of elements for which the predicate was true.
   */
  filter(filterer: (value: DatasetElement) => boolean): Dataset {
    const base = this;
    return datasetFromStreamFn(() => {
      return base.getStream().filter(x => tidy(() => filterer(x)));
    });
  }

  /**
   * Maps this dataset through a 1-to-1 transform.
   *
   * @param transform A function mapping a `DatasetElement` to a transformed
   *   `DatasetElement`.
   *
   * @returns A `Dataset` of transformed elements.
   */
  map(transform: (value: DatasetElement) => DatasetElement): Dataset {
    const base = this;
    return datasetFromStreamFn(() => {
      return base.getStream().map(x => tidy(() => transform(x)));
    });
  }

  /**
   * Groups elements into batches and arranges their values in columnar form.
   *
   * It is assumed that each of the incoming DatasetElements has the same set of
   * keys.  For each key, the resulting BatchDataset provides a BatchElement
   * collecting all of the incoming values for that key.  Incoming strings are
   * grouped into a string[].  Incoming Tensors are grouped into a new Tensor
   * where the 0'th axis is the batch dimension.  These columnar representations
   * for each key can be zipped together to reconstruct the original
   * DatasetElements.
   *
   * @param batchSize The number of elements desired per batch.
   * @param smallLastBatch Whether to emit the final batch when it has fewer
   *   than batchSize elements. Default true.
   * @returns A `BatchDataset`, from which a stream of batches can be obtained.
   */
  batch(batchSize: number, smallLastBatch = true): BatchDataset {
    return new BatchDataset(this, batchSize, smallLastBatch);
  }

  /**
   * Concatenates this `Dataset` with another.
   *
   * @param dataset A `Dataset` to be concatenated onto this one.
   * @returns A `Dataset`.
   */
  concatenate(dataset: Dataset): Dataset {
    const base = this;
    return datasetFromStreamFn(
        () => base.getStream().concatenate(dataset.getStream()));
  }

  /**
   * Repeats this dataset `count` times.
   *
   * NOTE: If this dataset is a function of global state (e.g. a random number
   * generator), then different repetitions may produce different elements.
   *
   * @param count: (Optional.) An integer, representing the number of times
   *   the dataset should be repeated. The default behavior (if `count` is
   *   `undefined` or negative) is for the dataset be repeated indefinitely.
   * @returns A `Dataset`.
   */
  repeat(count?: number): Dataset {
    const base = this;
    return datasetFromStreamFn(() => {
      const streamStream = streamFromFunction(() => base.getStream());
      return streamFromConcatenated(streamStream.take(count));
    });
  }

  /**
   * Creates a `Dataset` with at most `count` elements from this dataset.
   *
   * @param count: The number of elements of this dataset that should be taken
   *   to form the new dataset.  If `count` is `undefined` or negative, or if
   *   `count` is greater than the size of this dataset, the new dataset will
   *   contain all elements of this dataset.
   * @returns A `Dataset`.
   */
  take(count: number): Dataset {
    const base = this;
    return datasetFromStreamFn(() => base.getStream().take(count));
  }

  /**
   * Creates a `Dataset` that skips `count` elements from this dataset.
   *
   * @param count: The number of elements of this dataset that should be skipped
   *   to form the new dataset.  If `count` is greater than the size of this
   *   dataset, the new dataset will contain no elements.  If `count`
   *   is `undefined` or negative, skips the entire dataset.
   *
   * @returns A `Dataset`.
   */
  skip(count: number): Dataset {
    const base = this;
    return datasetFromStreamFn(() => base.getStream().skip(count));
  }

  // TODO(soergel): deep sharded shuffle, where supported

  /**
   * Randomly shuffles the elements of this dataset.
   *
   * @param bufferSize: An integer specifying the number of elements from this
   *   dataset from which the new dataset will sample.
   * @param seed: (Optional.) An integer specifying the random seed that will
   *   be used to create the distribution.
   * @param reshuffleEachIteration: (Optional.) A boolean, which if true
   *   indicates that the dataset should be pseudorandomly reshuffled each time
   *   it is iterated over. (Defaults to `true`.)
   * @returns A `Dataset`.
   */
  shuffle(bufferSize: number, seed?: string, reshuffleEachIteration = true):
      Dataset {
    const base = this;
    const random = seedrandom(seed);
    return datasetFromStreamFn(() => {
      let seed2 = random.int32();
      if (reshuffleEachIteration) {
        seed2 += random.int32();
      }
      return base.getStream().shuffle(bufferSize, seed2.toString());
    });
  }

  /**
   *  Creates a `Dataset` that prefetches elements from this Dataset.
   *
   * @param bufferSize: An integer specifying the number of elements to be
   *   prefetched.
   * @returns A `Dataset`.
   */
  prefetch(bufferSize: number): Dataset {
    const base = this;
    return datasetFromStreamFn(() => base.getStream().prefetch(bufferSize));
  }

  /**
   * Collect all elements of this dataset into an array.
   * Obviously this will succeed only for small datasets that fit in memory.
   * Useful for testing.
   *
   * @returns A Promise for an array of elements, which will resolve
   *   when a new stream has been obtained and fully consumed.
   */
  async collectAll() {
    return this.getStream().collectRemaining();
  }

  /**
   * Apply a function to every element of the dataset.
   *
   * After the function is applied to a `DatasetElement`, any Tensors contained
   * within that element are disposed.
   *
   * @param f A function to apply to each dataset element.
   */
  async forEach(f: (input: DatasetElement) => {}): Promise<void> {
    return this.getStream().forEach(f);
  }

  /* TODO(soergel): for parity with tf.data:
  Dataset.flat_map()
  Dataset.zip()
  Dataset.dense_to_sparse_batch()
  Dataset.group_by_window()
  Dataset.padded_batch()
  */
}

/**
 * Create a `Dataset` defined by a provided getStream() function.
 */
export function datasetFromStreamFn(
    getStreamFn: () => DataStream<DatasetElement>): Dataset {
  return new class extends Dataset {
    /*
     * Provide a new stream of elements.  Note this will also start new streams
     * from any underlying `Dataset`s.
     */
    getStream(): DataStream<DatasetElement> {
      return getStreamFn();
    }
  }
  ();
}

/**
 * Create a `Dataset` from an array of elements.
 */
export function datasetFromElements(items: DatasetElement[]): Dataset {
  return datasetFromStreamFn(() => streamFromItems(items));
}
