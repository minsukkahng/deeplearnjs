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

import {DataStream, QueueStream} from './data_stream';

export abstract class StringStream extends DataStream<string> {
  /**
   * Splits a string stream on a given separator.
   *
   * It is assumed that the incoming chunk boundaries have no semantic meaning,
   * so conceptually the incoming stream is treated simply as the concatenation
   * of its elements.
   *
   * The outgoing stream provides chunks corresponding to the results of the
   * standard string split() operation (even if such a chunk spanned incoming
   * chunks).  The separators are not included.
   *
   * A typical usage is to split a text file (represented as a stream with
   * arbitrary chunk boundaries) into lines.
   *
   * @param upstream A readable stream of strings that can be treated as
   *   concatenated.
   * @param separator A character to split on.
   */
  split(separator: string): StringStream {
    return new SplitStream(this, separator);
  }
}

// ============================================================================
// The following private classes serve to implement the chainable methods
// on StringStream.  Unfortunately they can't be placed in separate files, due
// to resulting trouble with circular imports.
// ============================================================================

// We wanted multiple inheritance, e.g.
//   class SplitStream extends QueueStream<string>, StringStream
// but the TypeScript mixin approach is a bit hacky, so we take this adapter
// approach instead.

class SplitStream extends StringStream {
  private impl: SplitStreamImpl;

  constructor(upstream: DataStream<string>, separator: string) {
    super();
    this.impl = new SplitStreamImpl(upstream, separator);
  }

  async next() {
    return this.impl.next();
  }
}

class SplitStreamImpl extends QueueStream<string> {
  // A partial string at the end of an upstream chunk
  carryover = '';

  constructor(
      protected upstream: DataStream<string>, protected separator: string) {
    super();
  }

  async pump(): Promise<boolean> {
    const chunk = await this.upstream.next();
    if (chunk == null) {
      if (this.carryover === '') {
        return false;
      }

      // Pretend that the pump succeeded in order to emit the small last batch.
      // The next pump() call will actually fail.
      this.outputQueue.push(this.carryover);
      this.carryover = '';
      return true;
    }
    const lines = chunk.split(this.separator);
    // Note the behavior: " ab ".split(' ') === ['', 'ab', '']
    // Thus the carryover may be '' if the separator falls on a chunk
    // boundary; this produces the correct result.

    lines[0] = this.carryover + lines[0];
    for (const line of lines.slice(0, -1)) {
      this.outputQueue.push(line);
    }
    this.carryover = lines[lines.length - 1];

    return true;
  }
}