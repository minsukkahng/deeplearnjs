import * as d3 from 'd3-selection';
import { contourDensity } from 'd3-contour';
import { geoPath } from 'd3-geo';
import { scaleLinear, scaleSequential } from 'd3-scale';
import { interpolateYlGnBu } from 'd3-scale-chromatic';
//import { line } from 'd3-shape';
import * as d3Transition from 'd3-transition';

import { PolymerElement, PolymerHTMLElement } from '../polymer-spec';
import * as dl from 'deeplearn';
//import { TypedArray } from '../../src/types';

import * as gan_lab_input_providers from './gan_lab_input_providers';
import * as gan_lab_drawing from './gan_lab_drawing';
import * as gan_lab_evaluators from './gan_lab_evaluators';

const BATCH_SIZE = 150;
const ATLAS_SIZE = 12000;
const NUM_GRID_CELLS = 30;
//const NUM_MANIFOLD_CELLS = 20;
const GENERATED_SAMPLES_VISUALIZATION_INTERVAL = 10;
const NUM_SAMPLES_VISUALIZED = 300;
const NUM_TRUE_SAMPLES_VISUALIZED = 300;
const SLOW_INTERVAL_MS = 600;

// tslint:disable-next-line:variable-name
const GANLabPolymer: new () => PolymerHTMLElement = PolymerElement({
  is: 'gan-lab',
  properties: {
    dLearningRate: Number,
    gLearningRate: Number,
    learningRateOptions: Array,
    dOptimizerType: String,
    gOptimizerType: String,
    optimizerTypeOptions: Array,
    lossType: String,
    lossTypeOptions: Array,
    selectedShapeName: String,
    shapeNames: Array,
    selectedNoiseType: String,
    noiseTypes: Array
  }
});

class GANLab extends GANLabPolymer {
  private iterationCount: number;

  private gOptimizer: dl.Optimizer;
  private dOptimizer: dl.Optimizer;

  private noiseProvider: dl.InputProvider;
  private trueSampleProvider: dl.InputProvider;
  //private uniformNoiseProvider: dl.InputProvider;
  private uniformInputProvider: dl.InputProvider;

  private noiseSize: number;
  private numGeneratorLayers: number;
  private numDiscriminatorLayers: number;
  private numGeneratorNeurons: number;
  private numDiscriminatorNeurons: number;

  private kDSteps: number;
  private kGSteps: number;

  private plotSizePx: number;

  private evaluator: gan_lab_evaluators.GANLabEvaluatorGridDensities;

  private canvas: HTMLCanvasElement;
  private drawing: gan_lab_drawing.GANLabDrawing;

  ready() {
    // HTML elements.
    const noiseSizeElement =
      document.getElementById('noise-size') as HTMLElement;
    this.noiseSize = +noiseSizeElement.innerText;
    document.getElementById('noise-size-add-button')!.addEventListener(
      'click', () => {
        if (this.noiseSize < 5) {
          this.noiseSize += 1;
          noiseSizeElement.innerText = this.noiseSize.toString();
          this.createExperiment();
        }
      });
    document.getElementById('noise-size-remove-button')!.addEventListener(
      'click', () => {
        if (this.noiseSize > 1) {
          this.noiseSize -= 1;
          noiseSizeElement.innerText = this.noiseSize.toString();
          this.createExperiment();
        }
      });

    const numGeneratorLayersElement =
      document.getElementById('num-g-layers') as HTMLElement;
    this.numGeneratorLayers = +numGeneratorLayersElement.innerText;
    document.getElementById('g-layers-add-button')!.addEventListener(
      'click', () => {
        if (this.numGeneratorLayers < 5) {
          this.numGeneratorLayers += 1;
          numGeneratorLayersElement.innerText =
            this.numGeneratorLayers.toString();
          this.createExperiment();
        }
      });
    document.getElementById('g-layers-remove-button')!.addEventListener(
      'click', () => {
        if (this.numGeneratorLayers > 0) {
          this.numGeneratorLayers -= 1;
          numGeneratorLayersElement.innerText =
            this.numGeneratorLayers.toString();
          this.createExperiment();
        }
      });

    const numDiscriminatorLayersElement =
      document.getElementById('num-d-layers') as HTMLElement;
    this.numDiscriminatorLayers = +numDiscriminatorLayersElement.innerText;
    document.getElementById('d-layers-add-button')!.addEventListener(
      'click', () => {
        if (this.numDiscriminatorLayers < 5) {
          this.numDiscriminatorLayers += 1;
          numDiscriminatorLayersElement.innerText =
            this.numDiscriminatorLayers.toString();
          this.createExperiment();
        }
      });
    document.getElementById('d-layers-remove-button')!.addEventListener(
      'click', () => {
        if (this.numDiscriminatorLayers > 0) {
          this.numDiscriminatorLayers -= 1;
          numDiscriminatorLayersElement.innerText =
            this.numDiscriminatorLayers.toString();
          this.createExperiment();
        }
      });

    const numGeneratorNeuronsElement =
      document.getElementById('num-g-neurons') as HTMLElement;
    this.numGeneratorNeurons = +numGeneratorNeuronsElement.innerText;
    document.getElementById('g-neurons-add-button').addEventListener(
      'click', () => {
        if (this.numGeneratorNeurons < 16) {
          this.numGeneratorNeurons += 1;
          numGeneratorNeuronsElement.innerText =
            this.numGeneratorNeurons.toString();
          this.createExperiment();
        }
      });
    document.getElementById('g-neurons-remove-button').addEventListener(
      'click', () => {
        if (this.numGeneratorNeurons > 0) {
          this.numGeneratorNeurons -= 1;
          numGeneratorNeuronsElement.innerText =
            this.numGeneratorNeurons.toString();
          this.createExperiment();
        }
      });

    const numDiscriminatorNeuronsElement =
      document.getElementById('num-d-neurons') as HTMLElement;
    this.numDiscriminatorNeurons = +numDiscriminatorNeuronsElement.innerText;
    document.getElementById('d-neurons-add-button').addEventListener(
      'click', () => {
        if (this.numDiscriminatorNeurons < 16) {
          this.numDiscriminatorNeurons += 1;
          numDiscriminatorNeuronsElement.innerText =
            this.numDiscriminatorNeurons.toString();
          this.createExperiment();
        }
      });
    document.getElementById('d-neurons-remove-button').addEventListener(
      'click', () => {
        if (this.numDiscriminatorNeurons > 0) {
          this.numDiscriminatorNeurons -= 1;
          numDiscriminatorNeuronsElement.innerText =
            this.numDiscriminatorNeurons.toString();
          this.createExperiment();
        }
      });

    const numKDStepsElement =
      document.getElementById('k-d-steps') as HTMLElement;
    this.kDSteps = +numKDStepsElement.innerText;
    document.getElementById('k-d-steps-add-button')!.addEventListener(
      'click', () => {
        if (this.kDSteps < 10) {
          this.kDSteps += 1;
          numKDStepsElement.innerText = this.kDSteps.toString();
        }
      });
    document.getElementById('k-d-steps-remove-button')!.addEventListener(
      'click', () => {
        if (this.kDSteps > 0) {
          this.kDSteps -= 1;
          numKDStepsElement.innerText = this.kDSteps.toString();
        }
      });

    const numKGStepsElement =
      document.getElementById('k-g-steps') as HTMLElement;
    this.kGSteps = +numKGStepsElement.innerText;
    document.getElementById('k-g-steps-add-button')!.addEventListener(
      'click', () => {
        if (this.kGSteps < 10) {
          this.kGSteps += 1;
          numKGStepsElement.innerText = this.kGSteps.toString();
        }
      });
    document.getElementById('k-g-steps-remove-button')!.addEventListener(
      'click', () => {
        if (this.kGSteps > 0) {
          this.kGSteps -= 1;
          numKGStepsElement.innerText = this.kGSteps.toString();
        }
      });

    this.lossTypeOptions = ['Log loss', 'LeastSq loss'];
    this.lossType = 'Log loss';
    this.querySelector('#loss-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.lossType = event.detail.selected;
        this.createExperiment();
      });

    this.learningRateOptions = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0];
    this.dLearningRate = 0.1;
    this.querySelector('#d-learning-rate-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.dLearningRate = +event.detail.selected;
        this.updateOptimizers('D');
      });
    this.gLearningRate = 0.1;
    this.querySelector('#g-learning-rate-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.gLearningRate = +event.detail.selected;
        this.updateOptimizers('G');
      });

    this.optimizerTypeOptions = ['SGD', 'Adam'];
    this.dOptimizerType = 'SGD';
    this.querySelector('#d-optimizer-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.dOptimizerType = event.detail.selected;
        this.updateOptimizers('D');
      });
    this.gOptimizerType = 'SGD';
    this.querySelector('#g-optimizer-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.gOptimizerType = event.detail.selected;
        this.updateOptimizers('G');
      });

    this.shapeNames = [
      'Line', 'Gaussian', 'Two Gaussian Hills', 'Three Dots', 'Drawing'];
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

    this.noiseTypes = ['Random', 'Gaussian'];
    this.selectedNoiseType = 'Random';
    this.querySelector('#noise-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.selectedNoiseType = event.detail.selected;
        this.createExperiment();
      });

    // Checkbox toggles.
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
    this.querySelector('#show-g-gradients')!.addEventListener(
      'change', (event: Event) => {
        const container =
          this.querySelector('#vis-generator-gradients') as SVGGElement;
        // tslint:disable-next-line:no-any
        container.style.visibility =
          (event.target as any).active ? 'visible' : 'hidden';
      });

    // Timeline controls.
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

    this.slowMode = false;
    this.querySelector('#slow-step')!.addEventListener(
      'click', (event: Event) => {
        // tslint:disable-next-line:no-any
        this.slowMode = !this.slowMode;

        if (this.slowMode === true) {
          document.getElementById('overlay-background')!.classList.add('shown');
          document.getElementById('tooltips')!.classList.add('shown');
        } else {
          this.dehighlightStep();
          document.getElementById(
            'group-discriminator').classList.remove('activated');
          document.getElementById(
            'group-generator').classList.remove('activated');
          document.getElementById(
            'overlay-background')!.classList.remove('shown');
          document.getElementById('tooltips')!.classList.remove('shown');
        }
      });

    this.editMode = true;
    document.getElementById('edit-model-button')!.addEventListener(
      'click', () => {
        const elements: NodeListOf<HTMLDivElement> =
          this.querySelectorAll('.config-item');
        for (let i = 0; i < elements.length; ++i) {
          elements[i].style.visibility =
            this.editMode ? 'hidden' : 'visible';
        }
        this.editMode = !this.editMode;
      });

    this.iterCountElement =
      document.getElementById('iteration-count') as HTMLElement;

    // Visualization.
    this.plotSizePx = 400;
    this.mediumPlotSizePx = 140;
    this.smallPlotSizePx = 80;

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
    this.math = dl.ENV.math;

    this.createExperiment();
  }

  private createExperiment() {
    // Reset.
    this.pause();
    this.iterationCount = 0;
    this.iterCountElement.innerText = this.iterationCount;

    this.isPausedOngoingIteration = false;

    this.recreateCharts();

    const dataElements = [
      d3.select('#vis-true-samples').selectAll('.true-dot'),
      d3.select('#svg-real-samples').selectAll('.true-dot'),
      d3.select('#svg-true-prediction').selectAll('.true-dot'),
      d3.select('#vis-true-samples-contour').selectAll('path'),
      d3.select('#svg-noise').selectAll('.noise-dot'),
      d3.select('#vis-generated-samples').selectAll('.generated-dot'),
      d3.select('#svg-generated-samples').selectAll('.generated-dot'),
      d3.select('#svg-generated-prediction').selectAll('.generated-dot'),
      d3.select('#vis-discriminator-output').selectAll('.uniform-dot'),
      d3.select('#svg-discriminator-output').selectAll('.uniform-dot'),
      d3.select('#svg-prediction').selectAll('.uniform-dot'),
      d3.select('#vis-manifold').selectAll('.uniform-generated-dot'),
      d3.select('#vis-manifold').selectAll('.manifold-cells'),
      d3.select('#vis-manifold').selectAll('.grids'),
      d3.select('#svg-generator-manifold').selectAll('.uniform-generated-dot'),
      d3.select('#svg-generator-manifold').selectAll('.manifold-cells'),
      d3.select('#svg-generator-manifold').selectAll('.grids'),
      d3.select('#vis-generator-gradients').selectAll('.gradient-generated'),
      d3.select('#svg-generator-gradients').selectAll('.gradient-generated')
    ];
    dataElements.forEach((element) => {
      element.data([]).exit().remove();
    });

    // Create a new graph.
    /*this.buildNetwork();

    if (this.session != null) {
      this.session.dispose();
    }
    this.session = new dl.Session(this.graph, this.math);*/

    // Input providers.
    const noiseProviderBuilder =
      new gan_lab_input_providers.GANLabNoiseProviderBuilder(
        this.noiseSize, this.selectedNoiseType,
        NUM_SAMPLES_VISUALIZED, BATCH_SIZE);
    noiseProviderBuilder.generateAtlas();
    this.noiseProvider = noiseProviderBuilder.getInputProvider();
    this.noiseProviderFixed = noiseProviderBuilder.getInputProvider(true);

    const drawingPositions = this.drawing.drawingPositions;
    const trueSampleProviderBuilder =
      new gan_lab_input_providers.GANLabTrueSampleProviderBuilder(
        ATLAS_SIZE, this.selectedShapeName,
        drawingPositions, this.sampleFromTrueDistribution, BATCH_SIZE);
    trueSampleProviderBuilder.generateAtlas();
    this.trueSampleProvider = trueSampleProviderBuilder.getInputProvider();
    this.trueSampleProviderFixed =
      trueSampleProviderBuilder.getInputProvider(true);

    /*if (this.noiseSize <= 2) {
      const uniformNoiseProviderBuilder =
        new gan_lab_input_providers.GANLabUniformNoiseProviderBuilder(
          this.math, this.noiseSize, NUM_MANIFOLD_CELLS, BATCH_SIZE);
      uniformNoiseProviderBuilder.generateAtlas();
      this.uniformNoiseProvider =
        uniformNoiseProviderBuilder.getInputProvider();
    }*/

    const uniformSampleProviderBuilder =
      new gan_lab_input_providers.GANLabUniformSampleProviderBuilder(
        NUM_GRID_CELLS, BATCH_SIZE);
    uniformSampleProviderBuilder.generateAtlas();
    this.uniformInputProvider = uniformSampleProviderBuilder.getInputProvider();

    // Visualize true samples.
    this.visualizeTrueDistribution(trueSampleProviderBuilder.getInputAtlas());

    // Visualize noise samples.
    this.visualizeNoiseDistribution(noiseProviderBuilder.getNoiseSample());

    // Initialize evaluator.
    this.evaluator =
      new gan_lab_evaluators.GANLabEvaluatorGridDensities(NUM_GRID_CELLS);
    this.evaluator.createGridsForTrue(
      trueSampleProviderBuilder.getInputAtlas(), NUM_TRUE_SAMPLES_VISUALIZED);

    // Prepare for model.
    this.initializeVariables();
    this.updateOptimizers();
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
      case 'Gaussian': {
        return [
          0.6 + 0.125 * gan_lab_input_providers.randNormal(),
          0.3 + 0.05 * gan_lab_input_providers.randNormal()
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
      case 'Three Dots': {
        const stdev = 0.03;
        if (rand < 0.333) {
          return [
            0.35 + stdev * gan_lab_input_providers.randNormal(),
            0.75 + stdev * gan_lab_input_providers.randNormal()
          ];
        } else if (rand < 0.666) {
          return [
            0.75 + stdev * gan_lab_input_providers.randNormal(),
            0.6 + stdev * gan_lab_input_providers.randNormal()
          ];
        } else {
          return [
            0.45 + stdev * gan_lab_input_providers.randNormal(),
            0.35 + stdev * gan_lab_input_providers.randNormal()
          ];
        }
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

    const trueContourList = [
      d3.select('#vis-true-samples-contour')
        .selectAll('path')
        .data(contour(trueDistribution))
    ];
    trueContourList.forEach((contours) => {
      contours.enter()
        .append('path')
        .attr('fill', (d: any) => color(d.value))
        .attr('data-value', (d: any) => d.value)
        .attr('d', geoPath());
    });

    const trueDotsList = [
      d3.select('#vis-true-samples')
        .selectAll('.true-dot').data(trueDistribution),
      d3.select('#svg-real-samples')
        .selectAll('.true-dot').data(trueDistribution),
      d3.select('#svg-true-prediction').select('.true-dots')
        .selectAll('.true-dot').data(trueDistribution)
    ];
    trueDotsList.forEach((dots, k) => {
      const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
      const radius = k === 0 ? 2 : 1;
      dots.enter()
        .append('circle')
        .attr('class', 'true-dot gan-lab')
        .attr('r', radius)
        .attr('cx', (d: number[]) => d[0] * plotSizePx)
        .attr('cy', (d: number[]) => (1.0 - d[1]) * plotSizePx)
        .append('title')
        .text((d: number[]) => `${d[0].toFixed(2)}, ${d[1].toFixed(2)}`);
    });
  }

  private visualizeNoiseDistribution(inputList: Float32Array) {
    const noiseSamples: number[][] = [];
    for (let i = 0; i < inputList.length / this.noiseSize; ++i) {
      const values = [];
      for (let j = 0; j < this.noiseSize; ++j) {
        values.push(inputList[i * this.noiseSize + j]);
      }
      noiseSamples.push(values);
    }

    if (this.noiseSize === 1) {
      d3.select('#svg-noise')
        .selectAll('.noise-dot').data(noiseSamples)
        .enter()
        .append('circle')
        .attr('class', 'noise-dot gan-lab')
        .attr('r', 1)
        .attr('cx', (d: number[]) => d[0] * this.smallPlotSizePx)
        .attr('cy', this.smallPlotSizePx / 2);
    } else if (this.noiseSize >= 2) {
      d3.select('#svg-noise')
        .selectAll('.noise-dot').data(noiseSamples)
        .enter()
        .append('circle')
        .attr('class', 'noise-dot gan-lab')
        .attr('r', 1)
        .attr('cx', (d: number[]) => d[0] * this.smallPlotSizePx)
        .attr('cy', (d: number[]) => (1.0 - d[1]) * this.smallPlotSizePx);
    }
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
    if (!this.isPausedOngoingIteration) {
      this.iterateTraining(true);
    }
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
        const dCost = this.dOptimizer.minimize(() => {
          const noiseBatch = this.noiseProvider.getNextCopy() as dl.Tensor2D;
          const trueSampleBatch =
            this.trueSampleProvider.getNextCopy() as dl.Tensor2D;
          const pred = this.modelDiscriminator(
            trueSampleBatch, this.modelGenerator(noiseBatch));
          return this.dLoss(pred.truePred, pred.generatedPred);
        }, true, this.dVariables);
        if (j + 1 === this.kDSteps) {
          dCostVal = dCost.get();
        }
      }

      let gCostVal = null;
      for (let j = 0; j < kGSteps; j++) {
        const gCost = this.gOptimizer.minimize(() => {
          const noiseBatch = this.noiseProvider.getNextCopy() as dl.Tensor2D;
          const pred = this.modelDiscriminator(
            null, this.modelGenerator(noiseBatch));
          return this.gLoss(pred.generatedPred);
        }, true, this.gVariables);
        if (j + 1 === this.kGSteps) {
          gCostVal = gCost.get();
        }
      }

      this.iterCountElement.innerText = this.iterationCount;

      if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
        this.iterationCount % GENERATED_SAMPLES_VISUALIZATION_INTERVAL === 0) {

        if (this.slowMode) {
          await this.sleep(SLOW_INTERVAL_MS);
          document.getElementById(
            'group-discriminator').classList.add('activated');
          this.highlightStep(true, 'component-d-loss', 'tooltip-d-loss',
            ['arrow-t-samples-d', 'arrow-d-t-prediction',
              'arrow-g-samples-d', 'arrow-d-g-prediction',
              'arrow-t-prediction-d-loss', 'arrow-g-prediction-d-loss']);
          await this.sleep(SLOW_INTERVAL_MS);
        }

        // Update discriminator loss.
        if (dCostVal) {
          document.getElementById('d-loss-value')!.innerText =
            dCostVal.toFixed(3);
          document.getElementById('d-loss-value-simple')!.innerText =
            this.lossType === 'LeastSq loss'
              ? dCostVal.toFixed(2)
              : (Math.pow(dCostVal * 0.5, 2)).toFixed(2);
        }

        if (this.slowMode) {
          await this.sleep(SLOW_INTERVAL_MS);
          this.dehighlightStep();
          this.highlightStep(true,
            'component-discriminator-gradients', 'tooltip-d-gradients',
            ['arrow-d-loss-d-1', 'arrow-d-loss-d-2']);
          await this.sleep(SLOW_INTERVAL_MS);
        }

        if (this.slowMode) {
          await this.sleep(SLOW_INTERVAL_MS);
          this.dehighlightStep();
          this.highlightStep(true,
            'group-discriminator', 'tooltip-update-discriminator',
            ['arrow-d-loss-d-3', 'arrow-d-loss-d-4']);
          await this.sleep(SLOW_INTERVAL_MS);
        }

        // Visualize discriminator's output.
        const dData = [];
        for (let i = 0; i < NUM_GRID_CELLS * NUM_GRID_CELLS / BATCH_SIZE; ++i) {
          const inputBatch =
            this.uniformInputProvider.getNextCopy() as dl.Tensor2D;
          const result = this.modelDiscriminator(inputBatch, null);
          const resultData = result.truePred.dataSync();
          //const resultData = await result.data();
          for (let j = 0; j < resultData.length; ++j) {
            dData.push(resultData[j]);
          }
        }

        const gridDotsList = [
          d3.select('#vis-discriminator-output')
            .selectAll('.uniform-dot').data(dData),
          d3.select('#svg-discriminator-output')
            .selectAll('.uniform-dot').data(dData),
          d3.select('#svg-true-prediction').select('.uniform-dots')
            .selectAll('.uniform-dot').data(dData),
          d3.select('#svg-generated-prediction')
            .selectAll('.uniform-dot').data(dData)
        ];
        if (this.iterationCount === 1) {
          gridDotsList.forEach((dots, k) => {
            const plotSizePx = k === 0 ? this.plotSizePx :
              (k === 1 ? this.mediumPlotSizePx : this.smallPlotSizePx);
            dots.enter()
              .append('rect')
              .attr('class', 'uniform-dot gan-lab')
              .attr('width', plotSizePx / NUM_GRID_CELLS)
              .attr('height', plotSizePx / NUM_GRID_CELLS)
              .attr(
                'x',
                (d: number, i: number) =>
                  (i % NUM_GRID_CELLS) * (plotSizePx / NUM_GRID_CELLS))
              .attr(
                'y',
                (d: number, i: number) => plotSizePx -
                  (Math.floor(i / NUM_GRID_CELLS) + 1) *
                  (plotSizePx / NUM_GRID_CELLS))
              .style('fill', (d: number) => this.colorScale(d))
              .append('title')
              .text((d: number) => Number(d).toFixed(3));
          });
        }
        gridDotsList.forEach((dots) => {
          dots.style('fill', (d: number) => this.colorScale(d));
          dots.select('title').text((d: number) => Number(d).toFixed(3));
        });

        if (this.slowMode) {
          await this.sleep(SLOW_INTERVAL_MS);
          this.dehighlightStep();
          document.getElementById(
            'group-discriminator').classList.remove('activated');
          await this.sleep(SLOW_INTERVAL_MS);
          await this.sleep(SLOW_INTERVAL_MS);
          document.getElementById(
            'group-generator').classList.add('activated');
          this.highlightStep(false, 'component-g-loss', 'tooltip-g-loss',
            ['arrow-noise-g', 'arrow-g-g-samples',
              'arrow-g-samples-d', 'arrow-d-g-prediction',
              'arrow-g-prediction-g-loss']);
          await this.sleep(SLOW_INTERVAL_MS);
        }

        // Update generator loss.
        if (gCostVal) {
          document.getElementById('g-loss-value')!.innerText =
            gCostVal.toFixed(3);
          document.getElementById('g-loss-value-simple')!.innerText =
            this.lossType === 'LeastSq loss'
              ? (gCostVal * 2.0).toFixed(2)
              : Math.pow(gCostVal, 2).toFixed(2);
        }

        // Update charts.
        if (this.iterationCount === 1) {
          const chartContainer =
            document.getElementById('chart-container') as HTMLElement;
          chartContainer.style.visibility = 'visible';
        }

        this.updateChartData(
          this.costChartData, this.iterationCount, [dCostVal, gCostVal]);
        this.costChart.update();

        if (this.slowMode) {
          await this.sleep(SLOW_INTERVAL_MS);
          this.dehighlightStep();
          this.highlightStep(false,
            'component-generator-gradients', 'tooltip-g-gradients',
            ['arrow-g-loss-g-1', 'arrow-g-loss-g-2']);
          await this.sleep(SLOW_INTERVAL_MS);
        }
        /*
                        // Visualize gradients for generator
                        const gradData: Array<[number, number, number, number]> = [];
                        const gActivation = await this.session.activationArrayMap.get(
                          this.generatedTensor).data();
                        const gGradient = await this.session.gradientArrayMap.get(
                          this.generatedTensor).data();
                        for (let i = 0; i < gActivation.length / 2; ++i) {
                          gradData.push([
                            gActivation[i * 2], gActivation[i * 2 + 1],
                            gGradient[i * 2], gGradient[i * 2 + 1]
                          ]);
                        }

                        // Todo: If not in step mode, need to update the positions of
                        // generated samples before showing gradients.

                        const gradDotsList = [
                          d3.select('#vis-generator-gradients')
                            .selectAll('.gradient-generated').data(gradData),
                          d3.select('#svg-generator-gradients')
                            .selectAll('.gradient-generated').data(gradData)
                        ];
                        //const gradDotsElementList = [
                        //  '#vis-generator-gradients',
                        //  '#svg-generator-gradients'
                        //];
                        if (this.iterationCount === 1) {
                          gradDotsList.forEach((dots, k) => {
                            const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
                            const arrowSize = k === 0 ? 5.0 : 1.0;
                            const arrowWidth = k === 0 ? 0.002 : 0.001;
                            dots.enter()
                              .append('polygon')
                              .attr('class', 'gradient-generated gan-lab')
                              .attr('points', (d: number[]) => {
                                const gradSize = Math.sqrt(
                                  d[2] * d[2] + d[3] * d[3] + 0.00000001);
                                const xNorm = d[2] / gradSize;
                                const yNorm = d[3] / gradSize;
                                return `${d[0] * plotSizePx},
                                  ${(1.0 - d[1]) * plotSizePx}
                                  ${(d[0] - yNorm * (-1) * arrowWidth) * plotSizePx},
                                  ${(1.0 - (d[1] - xNorm * arrowWidth)) * plotSizePx}
                                  ${(d[0] - d[2] * arrowSize) * plotSizePx},
                                  ${(1.0 - (d[1] - d[3] * arrowSize)) * plotSizePx}
                                  ${(d[0] - yNorm * arrowWidth) * plotSizePx},
                                  ${(1.0 - (d[1] - xNorm * (-1) * arrowWidth)) * plotSizePx}`;
                              });
                          });
                        }

                        gradDotsList.forEach((dots, k) => {
                          const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
                          const arrowSize = k === 0 ? 5.0 : 1.0;
                          const arrowWidth = k === 0 ? 0.002 : 0.001;
                          //d3Transition.transition()//.duration(1000)
                          //  .select(gradDotsElementList[k])
                          //  .selectAll('.gradient-generated').selection().data(gradData)
                          //  .transition().duration(SLOW_INTERVAL_MS)
                dots
                  .attr('points', (d: number[]) => {
                    const gradSize = Math.sqrt(
                      d[2] * d[2] + d[3] * d[3] + 0.00000001);
                    const xNorm = d[2] / gradSize;
                    const yNorm = d[3] / gradSize;
                    return `${d[0] * plotSizePx},
                        ${(1.0 - d[1]) * plotSizePx}
                        ${(d[0] - yNorm * (-1) * arrowWidth) * plotSizePx},
                        ${(1.0 - (d[1] - xNorm * arrowWidth)) * plotSizePx}
                        ${(d[0] - d[2] * arrowSize) * plotSizePx},
                        ${(1.0 - (d[1] - d[3] * arrowSize)) * plotSizePx}
                        ${(d[0] - yNorm * arrowWidth) * plotSizePx},
                        ${(1.0 - (d[1] - xNorm * (-1) * arrowWidth)) * plotSizePx}`;
                  });
              });

            if (this.slowMode) {
              await this.sleep(SLOW_INTERVAL_MS);
              this.dehighlightStep();
              this.highlightStep(false,
                'group-generator', 'tooltip-update-generator',
                ['arrow-g-loss-g-3', 'arrow-g-loss-g-4']);
              await this.sleep(SLOW_INTERVAL_MS);
            }

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

              const gManifoldList = [
                d3.select('#vis-manifold').selectAll('.grids').data(gridData),
                d3.select('#svg-generator-manifold')
                  .selectAll('.grids').data(gridData)
              ];
              gManifoldList.forEach((grids, k) => {
                const plotSizePx =
                  k === 0 ? this.plotSizePx : this.mediumPlotSizePx;
                const manifoldCell =
                  line()
                    .x((d: number[]) => d[0] * plotSizePx)
                    .y((d: number[]) => (1.0 - d[1]) * plotSizePx);

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
              });

              if (this.noiseSize === 1) {
                const manifoldDots =
                  d3.select('#vis-manifold').selectAll('.uniform-generated-dot')
                    .data(manifoldData);
                if (this.iterationCount === 1) {
                  manifoldDots.enter()
                    .append('circle')
                    .attr('class', 'uniform-generated-dot gan-lab')
                    .attr('r', 1);
                }
                manifoldDots.attr('cx', (d: TypedArray) => d[0] * this.plotSizePx)
                  .attr('cy', (d: TypedArray) => (1.0 - d[1]) * this.plotSizePx);
              }
            }

            if (this.slowMode) {
              await this.sleep(SLOW_INTERVAL_MS);
              this.dehighlightStep();
              this.highlightStep(false,
                'component-generated-samples', 'tooltip-generated-samples',
                ['arrow-noise-g', 'arrow-g-g-samples']);
              await this.sleep(SLOW_INTERVAL_MS);
            }
*/
        // Visualize generated samples.
        const gData: Array<[number, number]> = [];
        const noiseBatch = this.noiseProviderFixed.getNextCopy() as dl.Tensor2D;
        const gResult = this.modelGenerator(noiseBatch);
        //const gResultData = await gResult.data();
        const gResultData = gResult.dataSync();
        for (let j = 0; j < gResultData.length / 2; ++j) {
          gData.push([gResultData[j * 2], gResultData[j * 2 + 1]]);
        }
        /*
                    this.evaluator.updateGridsForGenerated(gData);
                    this.updateChartData(this.evalChartData, this.iterationCount, [
                      this.evaluator.getKLDivergenceScore(),
                      this.evaluator.getJSDivergenceScore()
                    ]);
                    this.evalChart.update();
        */
        const gDotsList = [
          d3.select('#vis-generated-samples')
            .selectAll('.generated-dot').data(gData),
          d3.select('#svg-generated-samples')
            .selectAll('.generated-dot').data(gData),
          d3.select('#svg-generated-prediction')
            .selectAll('.generated-dot').data(gData)
        ];
        const gDotsElementList = [
          '#vis-generated-samples',
          '#svg-generated-samples',
          '#svg-generated-prediction'
        ];
        if (this.iterationCount === 1) {
          gDotsList.forEach((dots, k) => {
            const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
            const radius = k === 0 ? 2 : 1;
            dots.enter()
              .append('circle')
              .attr('class', 'generated-dot gan-lab')
              .attr('r', radius)
              .attr('cx', (d: number[]) => d[0] * plotSizePx)
              .attr('cy', (d: number[]) => (1.0 - d[1]) * plotSizePx);
          });
        }
        gDotsList.forEach((dots, k) => {
          const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
          d3Transition.transition()
            .select(gDotsElementList[k])
            .selectAll('.generated-dot')
            .selection().data(gData)
            .transition().duration(SLOW_INTERVAL_MS)
            .attr('cx', (d: number[]) => d[0] * plotSizePx)
            .attr('cy', (d: number[]) => (1.0 - d[1]) * plotSizePx);
        });

        if (this.slowMode) {
          await this.sleep(SLOW_INTERVAL_MS);
          this.dehighlightStep();
          document.getElementById(
            'group-generator').classList.remove('activated');
          await this.sleep(SLOW_INTERVAL_MS);
        }

        if (!this.slowMode) {
          const componentElements: NodeListOf<HTMLDivElement> =
            this.querySelectorAll('.model-component');
          for (let i = 0; i < componentElements.length; ++i) {
            componentElements[i].classList.remove('d-highlighted');
            componentElements[i].classList.remove('g-highlighted');
          }
          const componentGroupElements: NodeListOf<HTMLDivElement> =
            this.querySelectorAll('.model-component-group');
          for (let i = 0; i < componentGroupElements.length; ++i) {
            componentGroupElements[i].classList.remove('activated');
            componentGroupElements[i].classList.remove('d-highlighted');
            componentGroupElements[i].classList.remove('g-highlighted');
          }
          const arrowElements: NodeListOf<HTMLDivElement> =
            this.querySelectorAll('#model-vis-svg path');
          for (let i = 0; i < arrowElements.length; ++i) {
            arrowElements[i].classList.remove('d-highlighted');
            arrowElements[i].classList.remove('g-highlighted');
            if (arrowElements[i].hasAttribute('marker-end')) {
              arrowElements[i].setAttribute('marker-end', 'url(#arrow-head)');
            }
          }
        }
      }
    });

    if (this.iterationCount > 10000) {
      this.isPlaying = false;
    }

    requestAnimationFrame(() => this.iterateTraining(true));
  }

  private highlightStep(isForD: boolean,
    componentElementName: string, tooltipElementName: string,
    arrowElementList: string[]) {
    this.highlightedComponent =
      document.getElementById(componentElementName);
    this.highlightedTooltip =
      document.getElementById(tooltipElementName);
    this.highlightedArrowList = arrowElementList.map(
      elementName => document.getElementById(elementName));

    this.highlightedComponent.classList.add(
      isForD ? 'd-highlighted' : 'g-highlighted');
    this.highlightedTooltip.classList.add('shown');
    this.highlightedTooltip.classList.add('highlighted');
    this.highlightedArrowList.forEach((arrowElement: SVGElement) => {
      arrowElement.classList.add(
        isForD ? 'd-highlighted' : 'g-highlighted');
      if (arrowElement.hasAttribute('marker-end')) {
        arrowElement.setAttribute('marker-end',
          isForD ? 'url(#arrow-head-d-highlighted)'
            : 'url(#arrow-head-g-highlighted)');
      }
    });
  }

  private dehighlightStep() {
    if (this.highlightedComponent) {
      this.highlightedComponent.classList.remove('d-highlighted');
      this.highlightedComponent.classList.remove('g-highlighted');
    }
    if (this.highlightedTooltip) {
      this.highlightedTooltip.classList.remove('shown');
      this.highlightedTooltip.classList.remove('highlighted');
    }
    if (this.highlightedArrowList) {
      this.highlightedArrowList.forEach((arrowElement: SVGElement) => {
        arrowElement.classList.remove('d-highlighted');
        arrowElement.classList.remove('g-highlighted');
        if (arrowElement.hasAttribute('marker-end')) {
          arrowElement.setAttribute('marker-end', 'url(#arrow-head)');
        }
      });
    }
  }

  private initializeVariables() {
    if (this.dVariables) {
      this.dVariables.forEach((v: dl.Tensor) => v.dispose());
    }
    if (this.gVariables) {
      this.gVariables.forEach((v: dl.Tensor) => v.dispose());
    }
    // Filter variable nodes for optimizers.
    this.dVariables = [];
    this.gVariables = [];

    // Generator.
    const gfc0W: dl.Tensor2D = dl.variable(
      dl.Tensor2D.randNormal(
        [this.noiseSize, this.numGeneratorNeurons], 0, 1.0 / Math.sqrt(2)),
      true);
    const gfc0B: dl.Tensor1D = dl.variable(
      dl.Tensor1D.zeros([this.numGeneratorNeurons]), true);

    this.gVariables.push(gfc0W);
    this.gVariables.push(gfc0B);

    for (let i = 0; i < this.numGeneratorLayers; ++i) {
      const gfcW: dl.Tensor2D = dl.variable(
        dl.Tensor2D.randNormal(
          [this.numGeneratorNeurons, this.numGeneratorNeurons], 0,
          1.0 / Math.sqrt(this.numGeneratorNeurons)),
        true);
      const gfcB: dl.Tensor1D = dl.variable(
        dl.Tensor1D.zeros([this.numGeneratorNeurons]),
        true);

      this.gVariables.push(gfcW);
      this.gVariables.push(gfcB);
    }

    const gfcLastW: dl.Tensor2D = dl.variable(
      dl.Tensor2D.randNormal(
        [this.numGeneratorNeurons, 2], 0,
        1.0 / Math.sqrt(this.numGeneratorNeurons)),
      true);
    const gfcLastB: dl.Tensor1D = dl.variable(
      dl.Tensor1D.zeros([2]), true);

    this.gVariables.push(gfcLastW);
    this.gVariables.push(gfcLastB);

    // Discriminator.
    const dfc0W: dl.Tensor2D = dl.variable(
      dl.Tensor2D.randNormal(
        [2, this.numDiscriminatorNeurons], 0, 1.0 / Math.sqrt(2)),
      true);
    const dfc0B: dl.Tensor1D = dl.variable(
      dl.Tensor1D.randNormal(
        [this.numDiscriminatorNeurons], 0,
        1.0 / Math.sqrt(this.numDiscriminatorNeurons)),
      true);

    this.dVariables.push(dfc0W);
    this.dVariables.push(dfc0B);

    for (let i = 0; i < this.numDiscriminatorLayers; ++i) {
      const dfcW: dl.Tensor2D = dl.variable(
        dl.Tensor2D.randNormal(
          [this.numDiscriminatorNeurons, this.numDiscriminatorNeurons], 0,
          1.0 / Math.sqrt(this.numDiscriminatorNeurons)),
        true);
      const dfcB: dl.Tensor1D = dl.variable(
        dl.Tensor1D.zeros([this.numDiscriminatorNeurons]),
        true);

      this.dVariables.push(dfcW);
      this.dVariables.push(dfcB);
    }

    const dfcLastW: dl.Tensor2D = dl.variable(
      dl.Tensor2D.randNormal(
        [this.numDiscriminatorNeurons, 1], 0,
        1.0 / Math.sqrt(this.numDiscriminatorNeurons)),
      true);
    const dfcLastB: dl.Tensor1D = dl.variable(
      dl.Tensor1D.zeros([1]), true);

    this.dVariables.push(dfcLastW);
    this.dVariables.push(dfcLastB);
  }

  private modelGenerator(noiseTensor: dl.Tensor2D): dl.Tensor2D {
    // Generator.
    const gfc0W = this.gVariables[0];
    const gfc0B = this.gVariables[1];

    let network = noiseTensor.matMul(gfc0W)
      .add(gfc0B)
      .relu();

    for (let i = 0; i < this.numGeneratorLayers; ++i) {
      const gfcW = this.gVariables[2 + i * 2];
      const gfcB = this.gVariables[3 + i * 2];

      network = network.matMul(gfcW)
        .add(gfcB)
        .relu();
    }

    const gfcLastW = this.gVariables[2 + this.numGeneratorLayers * 2];
    const gfcLastB = this.gVariables[3 + this.numGeneratorLayers * 2];

    const generatedTensor: dl.Tensor2D = network.matMul(gfcLastW)
      .add(gfcLastB)
      .sigmoid() as dl.Tensor2D;

    return generatedTensor;
  }

  private modelDiscriminator(
    inputTensor: dl.Tensor2D, generatedTensor: dl.Tensor2D): {
      truePred: dl.Tensor1D, generatedPred: dl.Tensor1D
    } {
    // Discriminator.
    const dfc0W = this.dVariables[0];
    const dfc0B = this.dVariables[1];

    let network1;
    if (inputTensor) {
      network1 = inputTensor.matMul(dfc0W)
        .add(dfc0B)
        .relu();
    }

    let network2;
    if (generatedTensor) {
      network2 = generatedTensor.matMul(dfc0W)
        .add(dfc0B)
        .relu();
    }

    for (let i = 0; i < this.numDiscriminatorLayers; ++i) {
      const dfcW = this.dVariables[2 + i * 2];
      const dfcB = this.dVariables[3 + i * 2];

      this.dVariables.push(dfcW);
      this.dVariables.push(dfcB);

      if (inputTensor) {
        network1 = network1.matMul(dfcW)
          .add(dfcB)
          .relu();
      }

      if (generatedTensor) {
        network2 = network2.matMul(dfcW)
          .add(dfcB)
          .relu();
      }
    }
    const dfcLastW = this.dVariables[2 + this.numDiscriminatorLayers * 2];
    const dfcLastB = this.dVariables[3 + this.numDiscriminatorLayers * 2];

    const predictionTensor1: dl.Tensor1D = inputTensor ?
      network1.matMul(dfcLastW)
        .add(dfcLastB)
        .sigmoid()
        .reshape([BATCH_SIZE])
      : null;

    const predictionTensor2: dl.Tensor1D = generatedTensor ?
      network2.matMul(dfcLastW)
        .add(dfcLastB)
        .sigmoid()
        .reshape([BATCH_SIZE])
      : null;

    return { truePred: predictionTensor1, generatedPred: predictionTensor2 };
  }

  // Define losses.
  private dLoss(truePred: dl.Tensor1D, generatedPred: dl.Tensor1D) {
    if (this.lossType === 'LeastSq loss') {
      return dl.add(
        truePred.sub(dl.scalar(1)).square().mean(),
        generatedPred.square().mean()
      ) as dl.Scalar;
    } else {
      return dl.add(
        truePred.log().mul(dl.scalar(0.9)).mean(),
        dl.sub(dl.scalar(1), generatedPred).log().mean()
      ).mul(dl.scalar(-1)) as dl.Scalar;
    }
  }

  private gLoss(generatedPred: dl.Tensor1D) {
    if (this.lossType === 'LeastSq loss') {
      return generatedPred.sub(dl.scalar(1)).square().mean() as dl.Scalar;
    } else {
      return generatedPred.log().mean().mul(dl.scalar(-1)) as dl.Scalar;
    }
  }

  private updateOptimizers(dOrG?: string) {
    if (this.selectedOptimizerType === 'Adam') {
      const beta1 = 0.9;
      const beta2 = 0.999;
      if (dOrG == null || dOrG === 'D') {
        this.dOptimizer = new dl.AdamOptimizer(
          this.dLearningRate, beta1, beta2, this.dNodes);
      }
      if (dOrG == null || dOrG === 'G') {
        this.gOptimizer = new dl.AdamOptimizer(
          this.gLearningRate, beta1, beta2, this.gNodes);
      }
    } else {
      if (dOrG == null || dOrG === 'D') {
        this.dOptimizer = dl.train.sgd(this.dLearningRate);
      }
      if (dOrG == null || dOrG === 'G') {
        this.gOptimizer = dl.train.sgd(this.gLearningRate);
      }
    }
  }

  private recreateCharts() {
    document.getElementById('chart-container').style.visibility = 'hidden';

    this.costChartData = new Array<ChartData>(2);
    for (let i = 0; i < this.costChartData.length; ++i) {
      this.costChartData[i] = [];
    }
    if (this.costChart != null) {
      this.costChart.destroy();
    }
    const costChartSpecification = [
      { label: 'Discriminator\'s Loss', color: 'rgba(5, 117, 176, 0.5)' },
      { label: 'Generator\'s Loss', color: 'rgba(123, 50, 148, 0.5)' }
    ];
    this.costChart = this.createChart(
      'cost-chart', this.costChartData, costChartSpecification, 0, 2);

    this.evalChartData = new Array<ChartData>(2);
    for (let i = 0; i < this.evalChartData.length; ++i) {
      this.evalChartData[i] = [];
    }
    if (this.evalChart != null) {
      this.evalChart.destroy();
    }
    const evalChartSpecification = [
      { label: 'KL Divergence (grid)', color: 'rgba(120, 220, 64, 0.5)' },
      { label: 'JS Divergence (grid)', color: 'rgba(220, 120, 64, 0.5)' }
    ];
    this.evalChart = this.createChart(
      'eval-chart', this.evalChartData, evalChartSpecification, 0);
  }

  private updateChartData(data: ChartData[][], xVal: number, yList: number[]) {
    for (let i = 0; i < yList.length; ++i) {
      data[i].push({ x: xVal, y: yList[i] });
    }
  }

  private createChart(
    canvasId: string, chartData: ChartData[][],
    specification: Array<{ label: string, color: string }>,
    min?: number, max?: number): Chart {
    const context = (document.getElementById(canvasId) as HTMLCanvasElement)
      .getContext('2d') as CanvasRenderingContext2D;
    const chartDatasets = specification.map((chartSpec, i) => {
      return {
        data: chartData[i],
        fill: false,
        label: chartSpec.label,
        pointRadius: 0,
        borderColor: chartSpec.color,
        borderWidth: 1,
        lineTension: 0,
        pointHitRadius: 8
      };
    });

    return new Chart(context, {
      type: 'line',
      data: { datasets: chartDatasets },
      options: {
        animation: { duration: 0 },
        responsive: false,
        scales: {
          xAxes: [{ type: 'linear', position: 'bottom' }],
          yAxes: [{ ticks: { max, min } }]
        }
      }
    });
  }

  private sleep(ms: number) {
    return new Promise(resolve => {
      const check = () => {
        if (this.isPlaying) {
          this.isPausedOngoingIteration = false;
          resolve();
        } else {
          this.isPausedOngoingIteration = true;
          setTimeout(check, 1000);
        }
      };
      setTimeout(check, ms);
    });
  }
}

document.registerElement(GANLab.prototype.is, GANLab);
