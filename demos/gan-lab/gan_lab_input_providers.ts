import { Array2D, InputProvider, NDArray, NDArrayMath } from 'deeplearn';

export abstract class GANLabInputProviderBuilder {
  protected atlas: Array2D;
  protected providerCounter: number;

  constructor(protected batchSize: number) {
    this.providerCounter = -1;
  }

  protected abstract generateAtlas(): void;

  abstract getInputProvider(fixStarting?: boolean): InputProvider;
}

export class GANLabNoiseProviderBuilder extends
  GANLabInputProviderBuilder {

  constructor(
    private math: NDArrayMath, private noiseSize: number,
    private numSamplesVisualized: number, batchSize: number) {
    super(batchSize);
  }

  generateAtlas() {
    this.atlas = Array2D.randUniform(
      [this.numSamplesVisualized * this.batchSize, this.noiseSize], 0.0, 1.0);
  }

  getInputProvider(fixStarting?: boolean): InputProvider {
    const provider = this;
    return {
      getNextCopy(): NDArray {
        provider.providerCounter++;
        return provider.math.scope(() => {
          return provider.math.slice2D(
            provider.atlas,
            [fixStarting ? 0 :
              (provider.providerCounter * provider.batchSize) %
              provider.numSamplesVisualized, 0],
            [provider.batchSize, provider.noiseSize]);
        });
      },
      disposeCopy(math: NDArrayMath, copy: NDArray) {
        copy.dispose();
      }
    };
  }

  getNoiseSample(): Float32Array {
    return this.math.slice2D(this.atlas,
      [0, 0], [this.batchSize, this.noiseSize]).dataSync() as Float32Array;
  }
}

export class GANLabTrueSampleProviderBuilder extends
  GANLabInputProviderBuilder {

  private inputAtlasList: number[];

  constructor(
    private math: NDArrayMath, private atlasSize: number,
    private selectedShapeName: string,
    private drawingPositions: Array<[number, number]>,
    private sampleFromTrueDistribution: Function, batchSize: number) {
    super(batchSize);
    this.inputAtlasList = [];
  }

  generateAtlas() {
    for (let i = 0; i < this.atlasSize; ++i) {
      const distribution = this.sampleFromTrueDistribution(
        this.selectedShapeName, this.drawingPositions);
      this.inputAtlasList.push(distribution[0]);
      this.inputAtlasList.push(distribution[1]);
    }
    this.atlas = Array2D.new([this.atlasSize, 2], this.inputAtlasList);
  }

  getInputProvider(fixStarting?: boolean): InputProvider {
    const provider = this;
    return {
      getNextCopy(): NDArray {
        provider.providerCounter++;
        return provider.math.scope(() => {
          return provider.math.slice2D(
            provider.atlas,
            [fixStarting ? 0 :
              (provider.providerCounter * provider.batchSize) %
              provider.atlasSize, 0],
            [provider.batchSize, 2]);
        });
      },
      disposeCopy(math: NDArrayMath, copy: NDArray) {
        copy.dispose();
      }
    };
  }

  getInputAtlas(): number[] {
    return this.inputAtlasList;
  }
}

export class GANLabUniformNoiseProviderBuilder extends
  GANLabInputProviderBuilder {

  constructor(
    private math: NDArrayMath, private noiseSize: number,
    private numManifoldCells: number, batchSize: number) {
    super(batchSize);
  }

  generateAtlas() {
    const inputAtlasList = [];
    if (this.noiseSize === 1) {
      for (let i = 0; i < this.numManifoldCells + 1; ++i) {
        inputAtlasList.push(i / this.numManifoldCells);
      }
    } else if (this.noiseSize === 2) {
      for (let i = 0; i < this.numManifoldCells + 1; ++i) {
        for (let j = 0; j < this.numManifoldCells + 1; ++j) {
          inputAtlasList.push(i / this.numManifoldCells);
          inputAtlasList.push(j / this.numManifoldCells);
        }
      }
    }
    while (inputAtlasList.length % this.batchSize * this.noiseSize > 0) {
      inputAtlasList.push(0.5);
    }
    this.atlas = Array2D.new(
      [inputAtlasList.length / this.noiseSize, this.noiseSize],
      inputAtlasList);
  }

  getInputProvider(): InputProvider {
    const provider = this;
    return {
      getNextCopy(): NDArray {
        provider.providerCounter++;
        if (provider.providerCounter * provider.batchSize >
          Math.pow(provider.numManifoldCells + 1, provider.noiseSize)) {
          provider.providerCounter = 0;
        }
        return provider.math.scope(() => {
          const begin: [number, number] = [
            (provider.providerCounter * provider.batchSize) %
            Math.pow(provider.numManifoldCells + 1, provider.noiseSize),
            0
          ];
          return provider.math.slice2D(provider.atlas, begin,
            [provider.batchSize, 2]);
        });
      },
      disposeCopy(math: NDArrayMath, copy: NDArray) {
        copy.dispose();
      }
    };
  }
}

export class GANLabUniformSampleProviderBuilder extends
  GANLabInputProviderBuilder {

  constructor(
    private math: NDArrayMath, private numGridCells: number,
    batchSize: number) {
    super(batchSize);
  }

  generateAtlas() {
    const inputAtlasList = [];
    for (let j = 0; j < this.numGridCells; ++j) {
      for (let i = 0; i < this.numGridCells; ++i) {
        inputAtlasList.push((i + 0.5) / this.numGridCells);
        inputAtlasList.push((j + 0.5) / this.numGridCells);
      }
    }
    this.atlas = Array2D.new(
      [this.numGridCells * this.numGridCells, 2], inputAtlasList);
  }

  getInputProvider(): InputProvider {
    const provider = this;
    return {
      getNextCopy(): NDArray {
        provider.providerCounter++;
        return provider.math.scope(() => {
          const begin: [number, number] = [
            (provider.providerCounter * provider.batchSize) %
            (provider.numGridCells * provider.numGridCells),
            0
          ];
          return provider.math.slice2D(provider.atlas, begin,
            [provider.batchSize, 2]);
        });
      },
      disposeCopy(math: NDArrayMath, copy: NDArray) {
        copy.dispose();
      }
    };
  }
}

export function randNormal() {
  const u = 1 - Math.random();
  const v = 1 - Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
