export class GANLabEvaluatorGridDensities {

  private gridTrueSampleCount: number[];
  private gridTrueDensities: number[];
  private gridGeneratedDensities: number[];

  constructor(private numGrid: number) {
    this.gridTrueSampleCount = new Array(numGrid * numGrid).fill(0);
    this.gridTrueDensities = new Array(numGrid * numGrid).fill(0.0);
    this.gridGeneratedDensities = new Array(numGrid * numGrid);
  }

  private mapPointToGridIndex(point: [number, number]) {
    return Math.trunc(point[0] * this.numGrid) +
      this.numGrid * Math.trunc(point[1] * this.numGrid);
  }

  createGridsForTrue(trueAtlas: number[], numTrueSamples: number) {
    for (let i = 0; i < numTrueSamples; ++i) {
      const values = trueAtlas.splice(i * 2, i * 2 + 2);
      this.gridTrueSampleCount[this.mapPointToGridIndex(
        [values[0], values[1]])]++;
      this.gridTrueDensities[this.mapPointToGridIndex(
        [values[0], values[1]])] += 1.0 / numTrueSamples;
    }
  }

  testGeneratedOnTrue(generatedSamples: Array<[number, number]>): number {
    let score = 0.0;
    const numGeneratedSamples = generatedSamples.length;
    for (let i = 0; i < numGeneratedSamples; ++i) {
      score += this.gridTrueDensities[this.mapPointToGridIndex(
        generatedSamples[i])];
    }
    return score;
  }

  updateGridsForGenerated(generatedSamples: Array<[number, number]>) {
    const numGeneratedSamples = generatedSamples.length;
    this.gridGeneratedDensities.fill(0.0);
    for (let i = 0; i < numGeneratedSamples; ++i) {
      this.gridGeneratedDensities[this.mapPointToGridIndex(
        generatedSamples[i])] += 1.0 / numGeneratedSamples;
    }
  }

  testTrueOnGenerated(): number {
    let score = 0.0;
    for (let j = 0; j < this.gridTrueSampleCount.length; ++j) {
      score += this.gridTrueSampleCount[j] * this.gridGeneratedDensities[j];
    }
    return score;
  }

  getKLDivergenceScore(): number {
    let score = 0.0;
    for (let j = 0; j < this.gridTrueDensities.length; ++j) {
      score += (this.gridTrueDensities[j] + 0.0001) * Math.log2(
        (this.gridTrueDensities[j] + 0.0001) /
        (this.gridGeneratedDensities[j] + 0.0001));
    }
    return score;
  }

  getJSDivergenceScore(): number {
    let leftJS = 0.0;
    let rightJS = 0.0;
    for (let j = 0; j < this.gridTrueDensities.length; ++j) {
      const averageDensity = 0.5 *
        (this.gridTrueDensities[j] + this.gridGeneratedDensities[j]);
      leftJS += (this.gridTrueDensities[j] + 0.0001) * Math.log2(
        (this.gridTrueDensities[j] + 0.0001) /
        (averageDensity + 0.0001));
      rightJS += (this.gridGeneratedDensities[j] + 0.0001) * Math.log2(
        (this.gridGeneratedDensities[j] + 0.0001) /
        (averageDensity + 0.0001));
    }
    return 0.5 * (leftJS + rightJS);
  }
}
