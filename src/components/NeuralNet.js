import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import Plot from "react-plotly.js";
import { generateLinearSeparable } from "../scripts/data";
import { Button } from "@chakra-ui/react";
import { model } from "@tensorflow/tfjs";
import _ from "underscore";
import Plotter from "./Plotter";

const colors = { 0: "blue", 1: "red" };
const n = 15;
const batchSize = 5;
const learningRate = 10;

const colorscaleValue = [
  [0, "#0000ff"],
  [1, "#ff0000"],
];

const NeuralNet = (props) => {
  const [points, setPoints] = useState({
    pointsX: [],
    pointsY: [],
    labels: [],
  });

  const [model, setModel] = useState(null);
  const [hmap, setHmap] = useState(tf.zeros([n, n]).add(0.5).arraySync());
  const [data, setData] = useState({});
  const [loss, setLoss] = useState([]);

  const loadModel = async () => {
    // create model (don't train on data yet though)
    const { inputs, labels } = generateLinearSeparable();
    setData({ inputs, labels });

    const pointsX = inputs
      .slice([0, 0], [inputs.shape[0], 1])
      .reshape([-1])
      .arraySync();
    const pointsY = inputs
      .slice([0, 1], [inputs.shape[0], 1])
      .reshape([-1])
      .arraySync();
    setPoints({ pointsX, pointsY, labels: labels.reshape([-1]).arraySync() });

    console.log("NeuralNet useEffect");
    const model = tf.sequential();
    // model.add(
    //   tf.layers.dense({ units: 4, inputShape: [2], activation: "relu" })
    // );
    model.add(
      tf.layers.dense({
        units: 1,
        inputShape: [2],
        activation: "sigmoid",
        name: "output",
      })
    );

    // console.log(JSON.stringify(model.outputs[0].shape));
    // tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // compile model
    model.compile({
      optimizer: tf.train.sgd(learningRate),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });

    setModel(model);

    // reset hmap
    const hmapNew = tf.zeros([n, n]).add(0.5).arraySync();
    setHmap(hmapNew);
    setLoss([]);
  };

  const fit = async () => {
    const { inputs, labels } = data;

    const fit = await model.fit(inputs, labels, {
      batchSize,
      epochs: 1,
      shuffle: true,
      callbacks: console.log,
    });

    setLoss((l) => [...l, fit.history.loss[0]]);

    // model.predict(inputs).print();
    const hmapNew = tf.zeros([n, n]).arraySync();
    const sweepPoints = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        sweepPoints.push([i / n, j / n]);
      }
    }
    const prediction = model.predict(tf.tensor2d(sweepPoints)).arraySync();
    let counter = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        hmapNew[i][j] = prediction[counter][0];
        counter++;
      }
    }

    setHmap(hmapNew);
  };

  console.log(_.range(1, loss.length + 1));
  console.log(loss);

  useEffect(() => loadModel, []);
  return (
    <div>
      <Button onClick={loadModel} style={{ margin: 10 }}>
        Regenerate Data
      </Button>
      <Button onClick={fit} style={{ margin: 10 }}>
        Fit One Epoch
      </Button>
      <Plotter points={points} hmap={hmap} width={200} height={200} />
      <Plot
        data={[
          {
            x: _.range(0, loss.length),
            y: loss,
            type: "scatter",
          },
        ]}
        style={{ width: "100%", height: "100%" }}
        layout={{
          // xaxis: { range: [0, 1] },
          yaxis: { range: [0, 5] },
          showlegend: false,
          autosize: false,
          width: 300,
          height: 300,
          title: "Loss (Binary Cross Entropy)",
          margin: {
            l: 30,
            r: 30,
            b: 30,
            t: 30,
          },
        }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
};

export default NeuralNet;
