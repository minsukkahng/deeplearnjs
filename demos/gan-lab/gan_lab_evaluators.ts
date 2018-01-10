export abstract class GANLabEvaluator {
  constructor() { }

  abstract getScore(): number;
}

export class GANLabEvaluatorAvgTrueGrid extends
  GANLabEvaluator {

  private gridSampleCount: number[];
  private gridDensities: number[];
  private currentScore: number;

  constructor(private numGrid: number) {
    super();

    this.gridSampleCount = new Array(numGrid * numGrid).fill(0);
    this.gridDensities = new Array(numGrid * numGrid).fill(0.0);
  }

  private mapPointToGridIndex(point: [number, number]) {
    return Math.trunc(point[0] * this.numGrid) +
      this.numGrid * Math.trunc(point[1] * this.numGrid);
  }

  createGridsForTrue(trueAtlas: number[], numTrueSamples: number) {
    for (let i = 0; i < numTrueSamples; ++i) {
      const values = trueAtlas.splice(i * 2, i * 2 + 2);
      this.gridSampleCount[this.mapPointToGridIndex(
        [values[0], values[1]])]++;
    }
    this.gridDensities = this.gridSampleCount.map(c => {
      return c / numTrueSamples;
    });
  }

  testGeneratedSamples(generatedSamples: Array<[number, number]>) {
    this.currentScore = 0.0;
    const numGeneratedSamples = generatedSamples.length;
    for (let i = 0; i < numGeneratedSamples; ++i) {
      this.currentScore += this.gridDensities[this.mapPointToGridIndex(
        generatedSamples[i])] / numGeneratedSamples;
    }
  }

  getScore(): number {
    return this.currentScore;
  }
}
