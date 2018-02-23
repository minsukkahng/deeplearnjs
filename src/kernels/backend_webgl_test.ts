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

// tslint:disable-next-line:max-line-length
import {describeWithFlags, expectArraysClose, WEBGL_ENVS} from '../test_util';
import {MathBackendWebGL} from './backend_webgl';

describeWithFlags('backendWebGL', WEBGL_ENVS, () => {
  it('delayed storage, reading', () => {
    const delayedStorage = true;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    const dataId = {};
    backend.register(dataId, [3], 'float32');
    backend.write(dataId, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    backend.getTexture(dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(backend.readSync(dataId), new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    backend.getTexture(dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(dataId);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('delayed storage, overwriting', () => {
    const delayedStorage = true;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    const dataId = {};
    backend.register(dataId, [3], 'float32');
    backend.write(dataId, new Float32Array([1, 2, 3]));
    backend.getTexture(dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    // overwrite.
    backend.write(dataId, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    expectArraysClose(backend.readSync(dataId), new Float32Array([4, 5, 6]));
    backend.getTexture(dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(backend.readSync(dataId), new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('immediate storage reading', () => {
    const delayedStorage = false;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    const dataId = {};
    backend.register(dataId, [3], 'float32');
    backend.write(dataId, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(backend.readSync(dataId), new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(dataId);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('immediate storage overwriting', () => {
    const delayedStorage = false;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    const dataId = {};
    backend.register(dataId, [3], 'float32');
    backend.write(dataId, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.write(dataId, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(backend.readSync(dataId), new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(dataId);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('disposal of backend disposes all textures', () => {
    const delayedStorage = false;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    const dataId = {};
    backend.register(dataId, [3], 'float32');
    backend.write(dataId, new Float32Array([1, 2, 3]));
    const dataId2 = {};
    backend.register(dataId2, [3], 'float32');
    backend.write(dataId2, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(2);
    backend.dispose();
    expect(texManager.getNumUsedTextures()).toBe(0);
  });
});
