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

export interface HeadingMap {
  'Tensors': 'Creation'|'Classes'|'Transformations'|'Slicing and Joining';
  'Operations': 'Arithmetic'|'Basic math'|'Matrices'|'Convolution'|
      'Normalization'|'Images'|'Logical'|'RNN'|'Reduction'|'Classification';
  'Training': 'Gradients'|'Optimizers'|'Losses'|'Classes';
  'Performance': 'Memory'|'Timing';
  // TODO(nsthorat): Make subheading optional.
  'Environment': '';
}
export type Heading = keyof HeadingMap;
export type Namespace = 'losses'|'image'|'train';

export interface DocInfo<H extends Heading> {
  heading: H;
  subheading?: HeadingMap[H];
  namespace?: Namespace;
  subclasses?: string[];
}

// Pass through function that does nothing. Only used for documentation.
export function doc<H extends Heading>(info: DocInfo<H>) {
  // tslint:disable-next-line:no-any
  return (...args: any[]) => {};
}
