import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import Plot from "react-plotly.js";
import { generateCircle, generateLinearSeparable } from "../scripts/data";
import {
  Button,
  ButtonGroup,
  Icon,
  IconButton,
  Select,
  Stack,
} from "@chakra-ui/react";
import { FaPlay, FaPause } from "react-icons/fa";
import { model } from "@tensorflow/tfjs";
import _ from "underscore";
import Plotter from "./Plotter";
import {
  AddIcon,
  ArrowRightIcon,
  ChevronRightIcon,
  RepeatClockIcon,
  RepeatIcon,
} from "@chakra-ui/icons";

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
  const [selectedDataModel, setSelectedDataModel] = useState("linear");

  // automatically fitting the model
  const [isPlaying, setIsPlaying] = useState(false);
  const requestRef = useRef();

  const generateData = () => {
    const dataGenerator = {
      linear: generateLinearSeparable,
      circle: generateCircle,
    }[selectedDataModel];
    const { inputs, labels } = dataGenerator();
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
    loadModel();
    return { inputs, labels };
  };

  const loadModel = async () => {
    const model = tf.sequential();
    model.add(
      tf.layers.dense({ units: 4, inputShape: [2], activation: "relu" })
    );
    model.add(
      tf.layers.dense({
        units: 1,
        // inputShape: [2],
        activation: "sigmoid",
        name: "output",
      })
    );

    // console.log(JSON.stringify(model.outputs[0].shape));
    // tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // compile model
    model.compile({
      optimizer: tf.train.sgd(0.2),
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

    let fitOne;
    try {
      fitOne = await model.fit(inputs, labels, {
        batchSize,
        epochs: 1,
        shuffle: true,
      });
    } catch (err) {
      if (
        err.message ==
        "Cannot start training because another fit() call is ongoing."
      ) {
        return;
      }
    }

    setLoss((l) => [...l, fitOne.history.loss[0]]);

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

  const play = async (time) => {
    if (!requestRef.isPlaying) return;
    // automatically fit the model
    await fit();
    requestRef.current = requestAnimationFrame(play);
  };

  useEffect(() => {
    generateData();
  }, []);
  return (
    <div>
      <Stack direction="row" style={{ margin: 10 }}>
        <Select
          width="8em"
          onChange={(ev) => setSelectedDataModel(ev.target.value)}
        >
          <option value="linear">Linear</option>
          <option value="circle">Circle</option>
        </Select>
        <Button onClick={generateData}>Generate</Button>
        <ButtonGroup isAttached>
          <IconButton
            aria-label="refresh"
            icon={<RepeatIcon />}
            onClick={loadModel}
          />
          <Button style={{ width: "5em" }}>Epoch {loss.length}</Button>
          <IconButton aria-label="add" icon={<AddIcon />} onClick={fit} />
          <IconButton
            aria-label="play"
            icon={<Icon as={isPlaying ? FaPause : FaPlay} />}
            onClick={
              isPlaying
                ? () => {
                    requestRef.isPlaying = false;
                    cancelAnimationFrame(requestRef.current);
                    setIsPlaying(false);
                  }
                : () => {
                    setIsPlaying(true);
                    requestRef.isPlaying = true;
                    requestAnimationFrame(play);
                  }
            }
          />
        </ButtonGroup>
      </Stack>

      <Stack direction="row" style={{ margin: 10 }}>
        <Stack direction="column" maxWidth={300}>
          <Plotter points={points} hmap={hmap} width={300} height={320} />
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
              showlegend: false,
              autosize: false,
              width: 300,
              height: 300,
              title: "Training Loss",
              margin: {
                l: 30,
                r: 30,
                b: 30,
                t: 30,
              },
              yaxis: {
                title: {
                  text: "Binary Cross Entropy",
                },
                range: [0, 2],
              },
              xaxis: {
                title: {
                  text: "Epoch",
                },
              },
            }}
            config={{ displayModeBar: false }}
          />
        </Stack>
      </Stack>
    </div>
  );
};

export default NeuralNet;
