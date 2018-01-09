import * as d3 from 'd3-selection';
import { contourDensity } from 'd3-contour';
import { geoPath } from 'd3-geo';
import { scaleLinear, scaleSequential } from 'd3-scale';
import { interpolateYlGnBu } from 'd3-scale-chromatic';
import { line } from 'd3-shape';

import { PolymerElement, PolymerHTMLElement } from '../polymer-spec';
import {
  Array1D, CostReduction, Graph, InputProvider, NDArray, NDArrayMath,
  NDArrayMathGPU, Scalar, Session, SGDOptimizer, Tensor
} from 'deeplearn';
import { TypedArray } from '../../src/util';

import * as gan_lab_input_providers from './gan_lab_input_providers';
import * as gan_lab_drawing from './gan_lab_drawing';
import * as gan_lab_evaluators from './gan_lab_evaluators';

const BATCH_SIZE = 150;
const ATLAS_SIZE = 12000;
const NUM_GRID_CELLS = 30;
const NUM_MANIFOLD_CELLS = 20;
const GENERATED_SAMPLES_VISUALIZATION_INTERVAL = 10;
const NUM_SAMPLES_VISUALIZED = 300;
const NUM_TRUE_SAMPLES_VISUALIZED = 1200;

// tslint:disable-next-line:variable-name
const GANLabPolymer: new () => PolymerHTMLElement = PolymerElement({
  is: 'gan-lab',
  properties: {
    learningRate: Number,
    learningRateOptions: Array,
    selectedShapeName: String,
    shapeNames: Array
  }
});

class GANLab extends GANLabPolymer {
  private math: NDArrayMath;
  private mathGPU: NDArrayMathGPU;

  private graph: Graph;
  private session: Session;
  private iterationCount: number;

  private gOptimizer: SGDOptimizer;
  private dOptimizer: SGDOptimizer;
  private predictionTensor1: Tensor;
  private predictionTensor2: Tensor;
  private gCostTensor: Tensor;
  private dCostTensor: Tensor;

  private noiseTensor: Tensor;
  private inputTensor: Tensor;
  private generatedTensor: Tensor;

  private noiseProvider: InputProvider;
  private trueSampleProvider: InputProvider;
  private uniformNoiseProvider: InputProvider;
  private uniformInputProvider: InputProvider;

  private noiseSize: number;
  private numGeneratorLayers: number;
  private numDiscriminatorLayers: number;
  private numGeneratorNeurons: number;
  private numDiscriminatorNeurons: number;

  private learningRate: number;
  private kDSteps: number;
  private kGSteps: number;

  private plotSizePx: number;

  private evaluator: gan_lab_evaluators.GANLabEvaluatorGridDensities;

  private canvas: HTMLCanvasElement;
  private drawing: gan_lab_drawing.GANLabDrawing;

  ready() {
    // HTML elements.
    const noiseSlider = this.querySelector('#noise-slider') as HTMLInputElement;
    const noiseSizeElement = this.querySelector('#noise-size') as HTMLElement;
    this.noiseSize = +noiseSlider.value;
    noiseSlider.addEventListener('value-change', (event) => {
      this.noiseSize = +noiseSlider.value;
      noiseSizeElement.innerText = this.noiseSize.toString();
      this.createExperiment();
    });

    const gLayersSlider =
      this.querySelector('#g-layers-slider') as HTMLInputElement;
    const numGeneratorLayersElement =
      this.querySelector('#num-g-layers') as HTMLElement;
    this.numGeneratorLayers = +gLayersSlider.value;
    gLayersSlider.addEventListener('value-change', (event) => {
      this.numGeneratorLayers = +gLayersSlider.value;
      numGeneratorLayersElement.innerText = this.numGeneratorLayers.toString();
      this.createExperiment();
    });

    const dLayersSlider =
      this.querySelector('#d-layers-slider') as HTMLInputElement;
    const numDiscriminatorLayersElement =
      this.querySelector('#num-d-layers') as HTMLElement;
    this.numDiscriminatorLayers = +dLayersSlider.value;
    dLayersSlider.addEventListener('value-change', (event) => {
      this.numDiscriminatorLayers = +dLayersSlider.value;
      numDiscriminatorLayersElement.innerText =
        this.numDiscriminatorLayers.toString();
      this.createExperiment();
    });

    const gNeuronsSlider =
      this.querySelector('#g-neurons-slider') as HTMLInputElement;
    const numGeneratorNeuronsElement =
      this.querySelector('#num-g-neurons') as HTMLElement;
    this.numGeneratorNeurons = +gNeuronsSlider.value;
    gNeuronsSlider.addEventListener('value-change', (event) => {
      this.numGeneratorNeurons = +gNeuronsSlider.value;
      numGeneratorNeuronsElement.innerText =
        this.numGeneratorNeurons.toString();
      this.createExperiment();
    });

    const dNeuronsSlider =
      this.querySelector('#d-neurons-slider') as HTMLInputElement;
    const numDiscriminatorNeuronsElement =
      this.querySelector('#num-d-neurons') as HTMLElement;
    this.numDiscriminatorNeurons = +dNeuronsSlider.value;
    dNeuronsSlider.addEventListener('value-change', (event) => {
      this.numDiscriminatorNeurons = +dNeuronsSlider.value;
      numDiscriminatorNeuronsElement.innerText =
        this.numDiscriminatorNeurons.toString();
      this.createExperiment();
    });

    const kDStepsSlider =
      this.querySelector('#k-d-steps-slider') as HTMLInputElement;
    const kDStepsElement = this.querySelector('#k-d-steps') as HTMLElement;
    this.kDSteps = +kDStepsSlider.value;
    kDStepsSlider.addEventListener('value-change', (event) => {
      kDStepsElement.innerText = kDStepsSlider.value;
      this.kDSteps = +kDStepsSlider.value;
    });

    const kGStepsSlider =
      this.querySelector('#k-g-steps-slider') as HTMLInputElement;
    const kGStepsElement = this.querySelector('#k-g-steps') as HTMLElement;
    this.kGSteps = +kGStepsSlider.value;
    kGStepsSlider.addEventListener('value-change', (event) => {
      kGStepsElement.innerText = kGStepsSlider.value;
      this.kGSteps = +kGStepsSlider.value;
    });

    this.learningRateOptions = [0.001, 0.01, 0.05, 0.1, 0.5];
    this.learningRate = 0.1;
    this.querySelector('#learning-rate-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.learningRate = +event.detail.selected;
        this.createExperiment();
      });

    this.shapeNames = ['Line', 'Two Gaussian Hills', 'Drawing'];
    this.selectedShapeName = 'Two Gaussian Hills';
    this.querySelector('#shape-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.selectedShapeName = event.detail.selected;
        if (this.selectedShapeName === 'Drawing') {
          this.pause();
          this.drawing.prepareDrawing();
        } else {
          this.createExperiment();
        }
      });

    this.querySelector('#overlap-plots')!.addEventListener(
      'change', (event: Event) => {
        const container =
          this.querySelector('#vis-discriminator-output') as SVGGElement;
        // tslint:disable-next-line:no-any
        container.style.visibility =
          (event.target as any).active ? 'visible' : 'hidden';
      });
    this.querySelector('#enable-manifold')!.addEventListener(
      'change', (event: Event) => {
        const container = this.querySelector('#vis-manifold') as SVGGElement;
        // tslint:disable-next-line:no-any
        container.style.visibility =
          (event.target as any).active ? 'visible' : 'hidden';
      });
    this.querySelector('#show-g-samples')!.addEventListener(
      'change', (event: Event) => {
        const container =
          this.querySelector('#vis-generated-samples') as SVGGElement;
        // tslint:disable-next-line:no-any
        container.style.visibility =
          (event.target as any).active ? 'visible' : 'hidden';
      });
    this.querySelector('#show-t-samples')!.addEventListener(
      'change', (event: Event) => {
        const container =
          this.querySelector('#vis-true-samples') as SVGGElement;
        // tslint:disable-next-line:no-any
        container.style.visibility =
          (event.target as any).active ? 'visible' : 'hidden';
      });
    this.querySelector('#show-t-contour')!.addEventListener(
      'change', (event: Event) => {
        const container =
          this.querySelector('#vis-true-samples-contour') as SVGGElement;
        // tslint:disable-next-line:no-any
        container.style.visibility =
          (event.target as any).active ? 'visible' : 'hidden';
      });

    const playButton =
      document.getElementById('play-pause-button') as HTMLInputElement;
    playButton.addEventListener(
      'click', () => this.onClickPlayPauseButton());
    const nextStepButton =
      document.getElementById('next-step-button') as HTMLInputElement;
    nextStepButton.addEventListener(
      'click', () => this.onClickNextStepButton());
    const resetButton =
      document.getElementById('reset-button') as HTMLInputElement;
    resetButton.addEventListener(
      'click', () => this.onClickResetButton());
    const nextStepDButton =
      document.getElementById('next-step-d-button') as HTMLInputElement;
    nextStepDButton.addEventListener(
      'click', () => this.onClickNextStepButton("D"));
    const nextStepGButton =
      document.getElementById('next-step-g-button') as HTMLInputElement;
    nextStepGButton.addEventListener(
      'click', () => this.onClickNextStepButton("G"));

    this.iterCountElement =
      document.getElementById('iteration-count') as HTMLElement;

    // Visualization.
    this.plotSizePx = 500;

    this.visTrueSamples = d3.select('#vis-true-samples');
    this.visTrueSamplesContour = d3.select('#vis-true-samples-contour');
    this.visGeneratedSamples = d3.select('#vis-generated-samples');
    this.visDiscriminator = d3.select('#vis-discriminator-output');
    this.visManifold = d3.select('#vis-manifold');

    this.colorScale = scaleLinear<string>().domain([0.0, 0.5, 1.0]).range([
      '#af8dc3', '#f5f5f5', '#7fbf7b'
    ]);

    // Drawing-related.
    this.canvas =
      document.getElementById('input-drawing-canvas') as HTMLCanvasElement;
    this.drawing = new gan_lab_drawing.GANLabDrawing(
      this.canvas, this.plotSizePx);

    this.finishDrawingButton =
      document.getElementById('finish-drawing') as HTMLInputElement;
    this.finishDrawingButton.addEventListener(
      'click', () => this.onClickFinishDrawingButton());

    // Math.
    this.mathGPU = new NDArrayMathGPU();
    this.math = this.mathGPU;

    this.createExperiment();
  }

  private createExperiment() {
    // Reset.
    this.pause();
    this.iterationCount = 0;
    this.iterCountElement.innerText = this.iterationCount;

    this.recreateCharts();

    this.visTrueSamples.selectAll('.true-dot').data([]).exit().remove();
    this.visTrueSamplesContour.selectAll('path').data([]).exit().remove();
    this.visGeneratedSamples.selectAll('.generated-dot')
      .data([])
      .exit()
      .remove();
    this.visDiscriminator.selectAll('.uniform-dot').data([]).exit().remove();
    this.visManifold.selectAll('.uniform-generated-dot')
      .data([])
      .exit()
      .remove();
    this.visManifold.selectAll('.manifold-cells').data([]).exit().remove();
    this.visManifold.selectAll('.grids').data([]).exit().remove();

    // Create a new graph.
    this.buildNetwork();

    if (this.session != null) {
      this.session.dispose();
    }
    this.session = new Session(this.graph, this.math);

    // Input providers.
    const noiseProviderBuilder =
      new gan_lab_input_providers.GANLabNoiseProviderBuilder(
        this.math, this.noiseSize, NUM_SAMPLES_VISUALIZED, BATCH_SIZE);
    noiseProviderBuilder.generateAtlas();
    this.noiseProvider = noiseProviderBuilder.getInputProvider();
    this.noiseProviderFixed = noiseProviderBuilder.getInputProvider(true);

    const drawingPositions = this.drawing.drawingPositions;
    const trueSampleProviderBuilder =
      new gan_lab_input_providers.GANLabTrueSampleProviderBuilder(
        this.math, ATLAS_SIZE, this.selectedShapeName,
        drawingPositions, this.sampleFromTrueDistribution, BATCH_SIZE);
    trueSampleProviderBuilder.generateAtlas();
    this.trueSampleProvider = trueSampleProviderBuilder.getInputProvider();
    this.trueSampleProviderFixed =
      trueSampleProviderBuilder.getInputProvider(true);

    if (this.noiseSize <= 2) {
      const uniformNoiseProviderBuilder =
        new gan_lab_input_providers.GANLabUniformNoiseProviderBuilder(
          this.math, this.noiseSize, NUM_MANIFOLD_CELLS, BATCH_SIZE);
      uniformNoiseProviderBuilder.generateAtlas();
      this.uniformNoiseProvider =
        uniformNoiseProviderBuilder.getInputProvider();
    }

    const uniformSampleProviderBuilder =
      new gan_lab_input_providers.GANLabUniformSampleProviderBuilder(
        this.math, NUM_GRID_CELLS, BATCH_SIZE);
    uniformSampleProviderBuilder.generateAtlas();
    this.uniformInputProvider = uniformSampleProviderBuilder.getInputProvider();

    // Visualize true samples.
    this.visualizeTrueDistribution(trueSampleProviderBuilder.getInputAtlas());

    // Initialize evaluator.
    this.evaluator =
      new gan_lab_evaluators.GANLabEvaluatorGridDensities(NUM_GRID_CELLS);
    this.evaluator.createGridsForTrue(
      trueSampleProviderBuilder.getInputAtlas(), NUM_TRUE_SAMPLES_VISUALIZED);
  }

  private sampleFromTrueDistribution(
    selectedShapeName: string, drawingPositions: Array<[number, number]>) {
    const rand = Math.random();
    switch (selectedShapeName) {
      case 'Drawing': {
        const index = Math.floor(drawingPositions.length * rand);
        return [
          drawingPositions[index][0] +
          0.02 * gan_lab_input_providers.randNormal(),
          drawingPositions[index][1] +
          0.02 * gan_lab_input_providers.randNormal()
        ];
      }
      case 'Line': {
        return [
          0.8 - 0.75 * rand + 0.01 * gan_lab_input_providers.randNormal(),
          0.6 + 0.3 * rand + 0.01 * gan_lab_input_providers.randNormal()
        ];
      }
      case 'Two Gaussian Hills': {
        if (rand < 0.5)
          return [
            0.3 + 0.1 * gan_lab_input_providers.randNormal(),
            0.7 + 0.1 * gan_lab_input_providers.randNormal()
          ];
        else
          return [
            0.7 + 0.05 * gan_lab_input_providers.randNormal(),
            0.4 + 0.2 * gan_lab_input_providers.randNormal()
          ];
      }
      default: {
        throw new Error('Invalid true distribution');
      }
    }
  }

  private visualizeTrueDistribution(inputAtlasList: number[]) {
    const color = scaleSequential(interpolateYlGnBu)
      .domain([0, 0.05]);

    const trueDistribution: Array<[number, number]> = [];
    while (trueDistribution.length < NUM_TRUE_SAMPLES_VISUALIZED) {
      const values = inputAtlasList.splice(0, 2);
      trueDistribution.push([values[0], values[1]]);
    }

    const contour = contourDensity()
      .x((d: number[]) => d[0] * this.plotSizePx)
      .y((d: number[]) => (1.0 - d[1]) * this.plotSizePx)
      .size([this.plotSizePx, this.plotSizePx])
      .bandwidth(15)
      .thresholds(5);
    this.visTrueSamplesContour
      .selectAll('path')
      .data(contour(trueDistribution))
      .enter()
      .append('path')
      .attr('fill', (d: any) => color(d.value))
      .attr('data-value', (d: any) => d.value)
      .attr('d', geoPath());

    this.visTrueSamples.selectAll('.true-dot')
      .data(trueDistribution)
      .enter()
      .append('circle')
      .attr('class', 'true-dot gan-lab')
      .attr('r', 2)
      .attr('cx', (d: number[]) => d[0] * this.plotSizePx)
      .attr('cy', (d: number[]) => (1.0 - d[1]) * this.plotSizePx)
      .append('title')
      .text((d: number[]) => `${d[0].toFixed(2)}, ${d[1].toFixed(2)}`);
  }

  private onClickFinishDrawingButton() {
    const drawingElement =
      this.querySelector('#drawing-container') as HTMLElement;
    drawingElement.style.display = 'none';
    const drawingBackgroundElement =
      this.querySelector('#drawing-disable-background') as HTMLDivElement;
    drawingBackgroundElement.style.display = 'none';
    this.createExperiment();
  }

  private play() {
    this.isPlaying = true;
    document.getElementById('play-pause-button')!.classList.add('playing');
    this.iterateTraining(true);
  }

  private pause() {
    this.isPlaying = false;
    const button = document.getElementById('play-pause-button');
    if (button.classList.contains('playing')) {
      button.classList.remove('playing');
    }
  }

  private onClickPlayPauseButton() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  private onClickNextStepButton(type?: string) {
    if (this.isPlaying) {
      this.pause();
    }
    this.isPlaying = true;
    this.iterateTraining(false, type);
    this.isPlaying = false;
  }

  private onClickResetButton() {
    if (this.isPlaying) {
      this.pause();
    }
    this.createExperiment();
  }

  private async iterateTraining(keepIterating: boolean, type?: string) {
    if (!this.isPlaying) {
      return;
    }

    this.iterationCount++;

    await this.math.scope(async () => {
      const kDSteps = type === "D" ? 1 : (type === "G" ? 0 : this.kDSteps);
      const kGSteps = type === "G" ? 1 : (type === "D" ? 0 : this.kGSteps);

      let dCostVal = null;
      for (let j = 0; j < kDSteps; j++) {
        const dCost = this.session.train(
          this.dCostTensor,
          [
            { tensor: this.inputTensor, data: this.trueSampleProvider },
            { tensor: this.noiseTensor, data: this.noiseProvider }
          ],
          1, this.dOptimizer, CostReduction.MEAN);
        if (j + 1 === this.kDSteps) {
          dCostVal = dCost.get();
        }
      }

      let gCostVal = null;
      for (let j = 0; j < kGSteps; j++) {
        const gCost = this.session.train(
          this.gCostTensor,
          [{ tensor: this.noiseTensor, data: this.noiseProvider }],
          1, this.gOptimizer, CostReduction.MEAN);
        if (j + 1 === this.kGSteps) {
          gCostVal = gCost.get();
        }
      }

      this.iterCountElement.innerText = this.iterationCount;

      if (!keepIterating || this.iterationCount === 1 ||
        this.iterationCount % GENERATED_SAMPLES_VISUALIZATION_INTERVAL === 0) {

        // Update charts.
        if (this.iterationCount === 1) {
          const chartContainer =
            document.getElementById('chart-container') as HTMLElement;
          chartContainer.style.visibility = 'visible';
          const evalChartContainer =
            document.getElementById('eval-chart-container') as HTMLElement;
          evalChartContainer.style.visibility = 'visible';
        }

        this.dCostChartData.push({ x: this.iterationCount, y: dCostVal });
        this.gCostChartData.push({ x: this.iterationCount, y: gCostVal });
        this.costChart.update();

        // Visualize discriminator's output.
        const dData = [];
        for (let i = 0; i < NUM_GRID_CELLS * NUM_GRID_CELLS / BATCH_SIZE; ++i) {
          const result = this.session.eval(
            this.predictionTensor1,
            [{ tensor: this.inputTensor, data: this.uniformInputProvider }]);
          const resultData = await result.data();
          for (let j = 0; j < resultData.length; ++j) {
            dData.push(resultData[j]);
          }
        }

        const gridDots =
          this.visDiscriminator.selectAll('.uniform-dot').data(dData);
        if (this.iterationCount === 1) {
          gridDots.enter()
            .append('rect')
            .attr('class', 'uniform-dot gan-lab')
            .attr('width', this.plotSizePx / NUM_GRID_CELLS)
            .attr('height', this.plotSizePx / NUM_GRID_CELLS)
            .attr(
            'x',
            (d: number, i: number) =>
              (i % NUM_GRID_CELLS) * (this.plotSizePx / NUM_GRID_CELLS))
            .attr(
            'y',
            (d: number, i: number) => this.plotSizePx -
              (Math.floor(i / NUM_GRID_CELLS) + 1) *
              (this.plotSizePx / NUM_GRID_CELLS))
            .style('fill', (d: number) => this.colorScale(d))
            .append('title')
            .text((d: number) => Number(d).toFixed(3));
        }
        gridDots.style('fill', (d: number) => this.colorScale(d));
        gridDots.select('title').text((d: number) => Number(d).toFixed(3));

        // Visualize generated samples.
        const gData: Array<[number, number]> = [];
        const gResult = this.session.eval(
          this.generatedTensor,
          [{ tensor: this.noiseTensor, data: this.noiseProviderFixed }]);
        const gResultData = await gResult.data();
        for (let j = 0; j < gResultData.length / 2; ++j) {
          gData.push([gResultData[j * 2], gResultData[j * 2 + 1]]);
        }

        const gDots =
          this.visGeneratedSamples.selectAll('.generated-dot').data(gData);
        if (this.iterationCount === 1) {
          gDots.enter()
            .append('circle')
            .attr('class', 'generated-dot gan-lab')
            .attr('r', 2)
            .attr('cx', (d: number[]) => d[0] * this.plotSizePx)
            .attr('cy', (d: number[]) => (1.0 - d[1]) * this.plotSizePx);
        }
        gDots.attr('cx', (d: number[]) => d[0] * this.plotSizePx)
          .attr('cy', (d: number[]) => (1.0 - d[1]) * this.plotSizePx);

        // Visualize manifold for 1-D or 2-D noise.
        interface ManifoldCell {
          points: TypedArray[];
          area?: number;
        }

        if (this.noiseSize <= 2) {
          const manifoldData: TypedArray[] = [];
          const numBatches = Math.ceil(Math.pow(
            NUM_MANIFOLD_CELLS + 1, this.noiseSize) / BATCH_SIZE);
          const remainingDummy = BATCH_SIZE * numBatches - Math.pow(
            NUM_MANIFOLD_CELLS + 1, this.noiseSize) * 2;
          for (let k = 0; k < numBatches; ++k) {
            const result = this.session.eval(
              this.generatedTensor,
              [{ tensor: this.noiseTensor, data: this.uniformNoiseProvider }]);

            const maniResult: TypedArray = await result.data() as TypedArray;

            for (let i = 0; i < (k + 1 < numBatches ?
              BATCH_SIZE : BATCH_SIZE - remainingDummy); ++i) {
              manifoldData.push(maniResult.slice(i * 2, i * 2 + 2));
            }
          }

          // Create grid cells.
          const gridData: ManifoldCell[] = [];
          let areaSum = 0.0;
          if (this.noiseSize === 1) {
            gridData.push({ points: manifoldData });
          } else if (this.noiseSize === 2) {
            for (let i = 0; i < NUM_MANIFOLD_CELLS * NUM_MANIFOLD_CELLS; ++i) {
              const x = i % NUM_MANIFOLD_CELLS;
              const y = Math.floor(i / NUM_MANIFOLD_CELLS);
              const index = x + y * (NUM_MANIFOLD_CELLS + 1);

              const gridCell = [];
              gridCell.push(manifoldData[index]);
              gridCell.push(manifoldData[index + 1]);
              gridCell.push(manifoldData[index + 1 + (NUM_MANIFOLD_CELLS + 1)]);
              gridCell.push(manifoldData[index + (NUM_MANIFOLD_CELLS + 1)]);
              gridCell.push(manifoldData[index]);

              // Calculate area by using four points.
              let area = 0.0;
              for (let j = 0; j < 4; ++j) {
                area += gridCell[j % 4][0] * gridCell[(j + 1) % 4][1] -
                  gridCell[j % 4][1] * gridCell[(j + 1) % 4][0];
              }
              area = 0.5 * Math.abs(area);
              areaSum += area;

              gridData.push({ points: gridCell, area });
            }
            // Normalize area.
            gridData.forEach(grid => {
              if (grid.area) {
                grid.area = grid.area / areaSum;
              }
            });
          }

          const manifoldCell =
            line()
              .x((d: number[]) => d[0] * this.plotSizePx)
              .y((d: number[]) => (1.0 - d[1]) * this.plotSizePx);

          const grids = this.visManifold.selectAll('.grids').data(gridData);

          if (this.iterationCount === 1) {
            grids.enter()
              .append('g')
              .attr('class', 'grids gan-lab')
              .append('path')
              .attr('class', 'manifold-cell gan-lab');
          }
          grids.select('.manifold-cell')
            .attr('d', (d: ManifoldCell) => manifoldCell(
              d.points.map(point => {
                const p: [number, number] = [point[0], point[1]];
                return p;
              })
            ))
            .style('fill', () => {
              return this.noiseSize === 2 ? '#7b3294' : 'none';
            })
            .style('fill-opacity', (d: ManifoldCell) => {
              return this.noiseSize === 2 ? Math.max(
                0.9 - d.area! * 0.4 * Math.pow(NUM_MANIFOLD_CELLS, 2), 0.1) :
                'none';
            });

          if (this.noiseSize === 1) {
            const manifoldDots =
              this.visManifold.selectAll('.uniform-generated-dot')
                .data(manifoldData);
            if (this.iterationCount === 1) {
              manifoldDots.enter()
                .append('circle')
                .attr('class', 'uniform-generated-dot gan-lab')
                .attr('r', 1);
            }
            manifoldDots.attr('cx', (d: number[]) => d[0] * this.plotSizePx)
              .attr('cy', (d: number[]) => (1.0 - d[1]) * this.plotSizePx);
          }
        }

        // Obtain simple evaluation scores.
        const eResult = this.session.evalAll(
          [this.predictionTensor1, this.predictionTensor2],
          [
            { tensor: this.inputTensor, data: this.trueSampleProviderFixed },
            { tensor: this.noiseTensor, data: this.noiseProviderFixed }
          ]);
        const eResultData1: Float32Array =
          await eResult[0].data() as Float32Array;
        const eResultData2: Float32Array =
          await eResult[1].data() as Float32Array;
        const acc1 = eResultData1.filter((v: number) => v >= 0.499).length /
          eResultData1.length;
        const acc2 = eResultData2.filter((v: number) => v <= 0.501).length /
          eResultData2.length;
        const acc3 = eResultData2.filter((v: number) => v >= 0.499).length /
          eResultData2.length;

        console.log([acc1.toFixed(3), acc2.toFixed(3),
        ((acc1 + acc2) * 0.5).toFixed(3), acc3.toFixed(3)]);
      }
    });

    if (this.iterationCount > 10000) {
      this.isPlaying = false;
    }

    requestAnimationFrame(() => this.iterateTraining(true));
  }

  private buildNetwork() {
    this.graph = new Graph();
    const g = this.graph;

    // Noise.
    const noise = g.placeholder('noise', [BATCH_SIZE, this.noiseSize]);
    this.noiseTensor = noise;

    // Generator.
    const gfc0W = g.variable(
      'gfc0W',
      NDArray.randNormal(
        [this.noiseSize, this.numGeneratorNeurons], 0, 1.0 / Math.sqrt(2)));
    const gfc0B =
      g.variable('gfc0B', Array1D.zeros([this.numGeneratorNeurons]));

    let network = g.matmul(this.noiseTensor, gfc0W);
    network = g.add(network, gfc0B);
    network = g.relu(network);

    for (let i = 0; i < this.numGeneratorLayers; ++i) {
      const gfcW = g.variable(
        `gfc${i + 1}W`,
        NDArray.randNormal(
          [this.numGeneratorNeurons, this.numGeneratorNeurons], 0,
          1.0 / Math.sqrt(this.numGeneratorNeurons)));
      const gfcB = g.variable(
        `gfc${i + 1}B`, Array1D.zeros([this.numGeneratorNeurons]));

      network = g.matmul(network, gfcW);
      network = g.add(network, gfcB);
      network = g.relu(network);
    }

    const gfcLastW = g.variable(
      'gfcLastW',
      NDArray.randNormal(
        [this.numGeneratorNeurons, 2], 0,
        1.0 / Math.sqrt(this.numGeneratorNeurons)));
    const gfcLastB = g.variable('gfcLastB', Array1D.zeros([2]));

    network = g.matmul(network, gfcLastW);
    network = g.add(network, gfcLastB);
    this.generatedTensor = g.sigmoid(network);

    // Real samples.
    this.inputTensor = g.placeholder('input', [BATCH_SIZE, 2]);

    // Discriminator.
    const dfc0W = g.variable(
      'dfc0W',
      NDArray.randNormal(
        [2, this.numDiscriminatorNeurons], 0, 1.0 / Math.sqrt(2)));
    const dfc0B =
      g.variable('dfc0B',
        NDArray.randNormal(
          [this.numDiscriminatorNeurons], 0,
          1.0 / Math.sqrt(this.numDiscriminatorNeurons)));

    let network1 = g.matmul(this.inputTensor, dfc0W);
    network1 = g.add(network1, dfc0B);
    network1 = g.relu(network1);

    let network2 = g.matmul(this.generatedTensor, dfc0W);
    network2 = g.add(network2, dfc0B);
    network2 = g.relu(network2);

    for (let i = 0; i < this.numDiscriminatorLayers; ++i) {
      const dfcW = g.variable(
        `dfc${i + 1}W`,
        NDArray.randNormal(
          [this.numDiscriminatorNeurons, this.numDiscriminatorNeurons], 0,
          1.0 / Math.sqrt(this.numDiscriminatorNeurons)));
      const dfcB = g.variable(
        `dfc${i + 1}B`, Array1D.zeros([this.numDiscriminatorNeurons]));

      network1 = g.matmul(network1, dfcW);
      network1 = g.add(network1, dfcB);
      network1 = g.relu(network1);

      network2 = g.matmul(network2, dfcW);
      network2 = g.add(network2, dfcB);
      network2 = g.relu(network2);
    }

    const dfcLastW = g.variable(
      'dfcLastW',
      NDArray.randNormal(
        [this.numDiscriminatorNeurons, 1], 0,
        1.0 / Math.sqrt(this.numDiscriminatorNeurons)));
    const dfcLastB = g.variable('dfcLastB', NDArray.zeros([1]));

    network1 = g.matmul(network1, dfcLastW);
    network1 = g.add(network1, dfcLastB);
    network1 = g.sigmoid(network1);
    network1 = g.reshape(network1, [BATCH_SIZE]);
    this.predictionTensor1 = network1;

    network2 = g.matmul(network2, dfcLastW);
    network2 = g.add(network2, dfcLastB);
    network2 = g.sigmoid(network2);
    network2 = g.reshape(network2, [BATCH_SIZE]);
    this.predictionTensor2 = network2;

    // Define losses.
    const dRealCostTensor = g.log(this.predictionTensor1);
    const dFakeCostTensor = g.log(
      g.subtract(g.constant(Scalar.new(1)), this.predictionTensor2));
    this.dCostTensor = g.multiply(
      g.add(dRealCostTensor, dFakeCostTensor), g.constant(Scalar.new(-1)));
    this.gCostTensor = g.multiply(
      g.log(this.predictionTensor2), g.constant(Scalar.new(-1)));

    this.dCostTensor = g.divide(g.reduceSum(this.dCostTensor),
      g.constant(Scalar.new(BATCH_SIZE)));
    this.gCostTensor = g.divide(g.reduceSum(this.gCostTensor),
      g.constant(Scalar.new(BATCH_SIZE)));

    // Filter variable nodes for optimizers.
    const gNodes = g.getNodes().filter(v => {
      return v.name.slice(0, 3) === 'gfc';
    });
    const dNodes = g.getNodes().filter(v => {
      return v.name.slice(0, 3) === 'dfc';
    });

    this.gOptimizer = new SGDOptimizer(this.learningRate, gNodes);
    this.dOptimizer = new SGDOptimizer(this.learningRate, dNodes);
  }

  private recreateCharts() {
    const chartContainer =
      document.getElementById('chart-container') as HTMLElement;
    chartContainer.style.visibility = 'hidden';

    this.dCostChartData = [];
    this.gCostChartData = [];
    if (this.costChart != null) {
      this.costChart.destroy();
    }
    this.costChart = this.createChart(
      'cost-chart', 'Cost', this.dCostChartData, this.gCostChartData, 0, 2);

    const evalChartContainer =
      document.getElementById('eval-chart-container') as HTMLElement;
    evalChartContainer.style.visibility = 'hidden';

    this.evalChartData1 = [];
    this.evalChartData2 = [];
    this.evalChartData3 = [];
    if (this.evalChart != null) {
      this.evalChart.destroy();
    }
    this.evalChart = this.createEvalChart(
      'eval-chart', 'Cost',
      this.evalChartData1, this.evalChartData2, this.evalChartData3, 0);
  }

  private createChart(
    canvasId: string, label: string, data1: ChartData[], data2: ChartData[],
    min?: number, max?: number): Chart {
    const context = (document.getElementById(canvasId) as HTMLCanvasElement)
      .getContext('2d') as CanvasRenderingContext2D;
    return new Chart(context, {
      type: 'line',
      data: {
        datasets: [
          {
            data: data1,
            fill: false,
            label: 'Discriminator\'s Loss',
            pointRadius: 0,
            borderColor: 'rgba(5, 117, 176, 0.5)',
            borderWidth: 1,
            lineTension: 0,
            pointHitRadius: 8
          },
          {
            data: data2,
            fill: false,
            label: 'Generator\'s Loss',
            pointRadius: 0,
            borderColor: 'rgba(123, 50, 148, 0.5)',
            borderWidth: 1,
            lineTension: 0,
            pointHitRadius: 8
          }
        ]
      },
      options: {
        animation: { duration: 0 },
        responsive: false,
        scales: {
          xAxes: [{ type: 'linear', position: 'bottom' }],
          yAxes: [{
            ticks: {
              max,
              min,
            }
          }]
        }
      }
    });
  }

  private createEvalChart(
    canvasId: string, label: string,
    data1: ChartData[], data2: ChartData[], data3: ChartData[],
    min?: number, max?: number): Chart {
    const context = (document.getElementById(canvasId) as HTMLCanvasElement)
      .getContext('2d') as CanvasRenderingContext2D;
    return new Chart(context, {
      type: 'line',
      data: {
        datasets: [
          {
            data: data1,
            fill: false,
            label: 'True Likelihood for G Samples',
            pointRadius: 0,
            borderColor: 'rgba(5, 117, 176, 0.5)',
            borderWidth: 1,
            lineTension: 0,
            pointHitRadius: 8
          },
          {
            data: data2,
            fill: false,
            label: 'G Likelihood for True Samples',
            pointRadius: 0,
            borderColor: 'rgba(123, 50, 148, 0.5)',
            borderWidth: 1,
            lineTension: 0,
            pointHitRadius: 8
          },
          {
            data: data3,
            fill: false,
            label: 'JS Divergence (grid)',
            pointRadius: 0,
            borderColor: 'rgba(220, 120, 64, 0.5)',
            borderWidth: 2,
            lineTension: 0,
            pointHitRadius: 8
          }
        ]
      },
      options: {
        animation: { duration: 0 },
        responsive: false,
        scales: {
          xAxes: [{ type: 'linear', position: 'bottom' }],
          yAxes: [{
            ticks: {
              max,
              min,
            }
          }]
        }
      }
    });
  }
}

document.registerElement(GANLab.prototype.is, GANLab);
