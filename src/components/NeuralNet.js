import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import Plot from "react-plotly.js";
import { generateLinearSeparable } from "../scripts/data";
import { Button } from "@chakra-ui/react";
import { model } from "@tensorflow/tfjs";
import _ from "underscore";

const colors = { 0: "blue", 1: "red" };
const n = 30;

const NeuralNet = (props) => {
  const [points, setPoints] = useState({
    pointsX: [],
    pointsY: [],
    labels: [],
  });

  const [model, setModel] = useState(null);
  const [hmap, setHmap] = useState(tf.zeros([n, n]).add(0.5).arraySync());

  const loadModel = async () => {
    // create model (don't train on data yet though)
    console.log("NeuralNet useEffect");
    const model = tf.sequential();
    model.add(
      tf.layers.dense({ units: 1, inputShape: [2], activation: "sigmoid" })
    );
    // model.add(tf.layers.dense({ units: 1 }));

    // console.log(JSON.stringify(model.outputs[0].shape));
    // tfvis.show.modelSummary({name: 'Model Summary'}, model);

    const { inputs, labels } = generateLinearSeparable();
    const pointsX = inputs
      .slice([0, 0], [inputs.shape[0], 1])
      .reshape([-1])
      .arraySync();
    const pointsY = inputs
      .slice([0, 1], [inputs.shape[0], 1])
      .reshape([-1])
      .arraySync();
    setPoints({ pointsX, pointsY, labels: labels.reshape([-1]).arraySync() });

    // compile model
    model.compile({
      optimizer: tf.train.sgd(0.2),
      loss: tf.losses.meanSquaredError,
      metrics: ["mse"],
    });

    const batchSize = 1;
    const epochs = 100;

    window.model = model;

    // for (let i = 0; i < epochs; i++) {
    //   const fit = await model.trainOnBatch(inputs, labels);
    // }
    const fit = await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ["loss", "mse"],
        { height: 200, callbacks: ["onEpochEnd"] }
      ),
    });

    model.predict(inputs).print();
    const hmapNew = tf.zeros([n, n]).arraySync();
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        hmapNew[i][j] = model
          .predict(tf.tensor2d([[i / n, j / n]]))
          .arraySync()[0][0];
      }
    }
    setHmap(hmapNew);
    setModel(model);

    // y.print();
    // console.log(y.reshape([-1]).arraySync());
    // console.log(points.labels.map((label) => colors[label]));
  };

  useEffect(() => loadModel, []);
  return (
    <div>
      <Button onClick={loadModel}>Reload Data</Button>
      <div style={{ width: 300, height: 300 }}>
        <Plot
          data={[
            {
              x: points.pointsX,
              y: points.pointsY,
              type: "scatter",
              mode: "markers",
              marker: { color: points.labels.map((label) => colors[label]) },
            },
          ]}
          style={{ width: "100%", height: "100%" }}
          layout={{
            xaxis: { range: [0, 1] },
            yaxis: { range: [0, 1] },
            showlegend: false,
            autosize: false,
            width: 500,
            height: 500,
          }}
          config={{ displayModeBar: false }}
        />
      </div>
      <div style={{ width: 300, height: 300 }}>
        <Plot
          data={[
            {
              z: hmap,
              x: _.range(0, 1, 1 / n),
              y: _.range(0, 1, 1 / n),
              type: "heatmap",
            },
          ]}
        ></Plot>
      </div>
    </div>
  );
};

export default NeuralNet;
