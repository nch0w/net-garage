import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import Plot from "react-plotly.js";
import { generateLinearSeparable } from "../scripts/data";
import { Button } from "@chakra-ui/react";

const colors = { 0: "red", 1: "blue" };

const NeuralNet = (props) => {
  const [points, setPoints] = useState({
    pointsX: [],
    pointsY: [],
    labels: [],
  });

  const loadModel = () => {
    console.log("NeuralNet useEffect");
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [2] }));
    model.add(tf.layers.dense({ units: 1 }));

    console.log(JSON.stringify(model.outputs[0].shape));
    // tfvis.show.modelSummary({name: 'Model Summary'}, model);

    const { x, y } = generateLinearSeparable();
    const pointsX = x.slice([0, 0], [x.shape[0], 1]).reshape([-1]).arraySync();
    const pointsY = x.slice([0, 1], [x.shape[0], 1]).reshape([-1]).arraySync();
    setPoints({ pointsX, pointsY, labels: y.reshape([-1]).arraySync() });

    y.print();
    console.log(y.reshape([-1]).arraySync());
    console.log(points.labels.map((label) => colors[label]));
  };

  useEffect(loadModel, []);
  return (
    <div>
      <Button onClick={loadModel}>Reload</Button>
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
    </div>
  );
};

export default NeuralNet;
