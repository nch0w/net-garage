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
  Table,
  TableCaption,
  TableContainer,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverHeader,
  PopoverBody,
  PopoverFooter,
  PopoverArrow,
  PopoverCloseButton,
  PopoverAnchor,
  Text,
  Editable,
  EditablePreview,
  EditableInput,
  Alert,
  AlertIcon,
  useBoolean,
  useColorMode,
} from "@chakra-ui/react";
import { FaPlay, FaPause } from "react-icons/fa";
import _ from "underscore";
import Plotter from "./Plotter";
import {
  AddIcon,
  InfoOutlineIcon,
  MinusIcon,
  MoonIcon,
  RepeatIcon,
  SunIcon,
} from "@chakra-ui/icons";
import Neuron from "./Neuron";
import { layers } from "@tensorflow/tfjs";
import { templateDark, templateLight } from "../scripts/plotlyTemplate";

const n = 15;
const batchSize = 5;
const defaultLearningRate = "0.1";
const defaultLayerSizes = [1];

// TODO make neural network configurable
// add learning rate controls
// add educational stuff, modal at beginning that explains everything
// if time, mobile optimization

const NeuralNet = (props) => {
  const [points, setPoints] = useState({
    pointsX: [],
    pointsY: [],
    labels: [],
  });

  const [model, setModel] = useState(null);
  const [modelUpdated, setModelUpdated] = useBoolean(false);
  const [hmap, setHmap] = useState(tf.zeros([n, n]).add(0.5).arraySync());
  const [data, setData] = useState({});
  const [loss, setLoss] = useState([]);
  const [selectedDataModel, setSelectedDataModel] = useState("linear");
  const [weights, setWeights] = useState([]);
  const [biases, setBiases] = useState([]);
  const [layerSizes, setLayerSizes] = useState(defaultLayerSizes);
  const [learningRate, setLearningRate] = useState(defaultLearningRate);
  const { colorMode, toggleColorMode } = useColorMode();

  // automatically fitting the model
  const [isPlaying, setIsPlaying] = useState(false);
  const requestRef = useRef();

  const generateData = () => {
    requestRef.isPlaying = false;
    setIsPlaying(false);
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
    requestRef.isPlaying = false;
    setIsPlaying(false);
    setModelUpdated.off();

    // create model code
    const model = tf.sequential({
      layers: [
        ...layerSizes.slice(0, layerSizes.length - 1).map((sz, i) =>
          tf.layers.dense({
            units: sz,
            activation: "relu",
            inputShape: !i ? [2] : null,
          })
        ),
        tf.layers.dense({
          units: 1,
          inputShape: layerSizes.length == 1 ? [2] : null,
          activation: "sigmoid",
          name: "output",
        }),
      ],
    });

    // tfvis.show.modelSummary({ name: "Model Summary" }, model);

    console.log(learningRate);
    // compile model
    model.compile({
      optimizer: tf.train.sgd(parseFloat(learningRate)),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });

    setModel(model);

    // reset hmap
    const hmapNew = tf.zeros([n, n]).add(0.5).arraySync();
    setHmap(hmapNew);
    setLoss([]);
    setWeights(model.layers.map((l) => l.getWeights()[0].arraySync()));
    setBiases(model.layers.map((l) => l.getWeights()[1].arraySync()));
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
    setWeights(model.layers.map((l) => l.getWeights()[0].arraySync()));
    setBiases(model.layers.map((l) => l.getWeights()[1].arraySync()));
    // setModel(model);

    // window.fitOne = fitOne;

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

  const pause = () => {
    requestRef.isPlaying = false;
    cancelAnimationFrame(requestRef.current);
    setIsPlaying(false);
  };

  useEffect(() => {
    generateData();
  }, []);

  return (
    <div>
      <Stack direction="row" style={{ gap: 10 }} flexWrap="wrap">
        <Select
          width="10em"
          onChange={(ev) => {
            setModelUpdated.on();
            pause();
            setSelectedDataModel(ev.target.value);
          }}
        >
          <option value="linear">Linear Data</option>
          <option value="circle">Circular Data</option>
        </Select>

        <Popover trigger="hover">
          <PopoverTrigger>
            <Button onClick={generateData}>
              Generate <InfoOutlineIcon style={{ marginLeft: 8 }} />{" "}
            </Button>
          </PopoverTrigger>
          <PopoverContent>
            <PopoverArrow />
            <PopoverHeader>
              <Text textTransform="uppercase" fontWeight="bold">
                Data Generation
              </Text>
            </PopoverHeader>
            <PopoverBody>
              In any neural network, it's essential to be training on
              high-quality data. Real datasets can have a lot of noise and
              dimensions that make it hard to see how the neural network is
              training. In this example, we'll be randomly generating 2D data
              that is linearly or radially separable.
            </PopoverBody>
          </PopoverContent>
        </Popover>

        <ButtonGroup isAttached>
          <IconButton
            aria-label="refresh"
            icon={<RepeatIcon />}
            onClick={loadModel}
            // colorScheme="{modelUpdated ? "blue" : "gray"}"
          />
          <Popover trigger="hover">
            <PopoverTrigger>
              <Button style={{ width: "7em" }}>
                Epoch {loss.length}{" "}
                <InfoOutlineIcon style={{ marginLeft: 8 }} />
              </Button>
            </PopoverTrigger>
            <PopoverContent>
              <PopoverArrow />
              <PopoverHeader>
                <Text textTransform="uppercase" fontWeight="bold">
                  Epochs
                </Text>
              </PopoverHeader>
              <PopoverBody>
                An epoch is a complete pass of the training dataset done by
                neural network. In this scenario, we train the model for 1 epoch
                at a time, in batches of {batchSize} samples.
              </PopoverBody>
            </PopoverContent>
          </Popover>

          <IconButton
            aria-label="add"
            icon={<AddIcon />}
            onClick={fit}
            disabled={modelUpdated}
          />
          <IconButton
            aria-label="play"
            icon={<Icon as={isPlaying ? FaPause : FaPlay} />}
            onClick={
              isPlaying
                ? pause
                : () => {
                    setIsPlaying(true);
                    requestRef.isPlaying = true;
                    requestAnimationFrame(play);
                  }
            }
            disabled={modelUpdated}
          />
        </ButtonGroup>

        <Button>
          Layers:
          <Select
            style={{ marginLeft: 10, width: "5em" }}
            defaultValue={`${layerSizes.length}`}
            variant="filled"
            onChange={(ev) => {
              // leave the output layer alone, and add/truncate if necessary
              const numLayers = ev.target.value;
              if (numLayers < layerSizes.length) {
                setLayerSizes([...layerSizes.slice(0, numLayers - 1), 1]);
              } else if (numLayers > layerSizes.length) {
                setLayerSizes([
                  ...layerSizes,
                  ...new Array(numLayers - layerSizes.length).fill(1),
                ]);
              }
              setModelUpdated.on();
              pause();
            }}
          >
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
          </Select>
        </Button>

        <Button>
          Learning Rate:
          <Select
            defaultValue={learningRate}
            style={{ marginLeft: 10, width: "5em" }}
            variant="filled"
            onChange={(ev) => {
              setLearningRate(parseFloat(ev.target.value));
              setModelUpdated.on();
              pause();
            }}
          >
            <option value="0.002">0.002</option>
            <option value="0.02">0.02</option>
            <option value="0.1">0.1</option>
            <option value="0.5">0.5</option>
            <option value="5">5</option>
          </Select>
        </Button>

        <IconButton
          onClick={toggleColorMode}
          icon={colorMode === "light" ? <MoonIcon /> : <SunIcon />}
        ></IconButton>
      </Stack>

      <Stack direction="row" style={{ margin: 10 }}>
        <div style={{ width: "100%" }}>
          <TableContainer>
            <Table variant="simple">
              <Thead>
                <Tr>
                  {_.range(layerSizes.length).map((l) => (
                    <Th key={l}>
                      {`Layer ${l}`}{" "}
                      {l < layerSizes.length - 1 && (
                        <ButtonGroup isAttached>
                          <IconButton
                            aria-label="minus"
                            icon={<MinusIcon />}
                            onClick={() => {
                              setLayerSizes((ls) => {
                                const lsNew = [...ls];
                                lsNew[l] -= 1;
                                return lsNew;
                              });
                              setModelUpdated.on();
                              pause();
                            }}
                            disabled={layerSizes[l] === 1}
                            size="xs"
                          />
                          <IconButton
                            aria-label="add"
                            icon={<AddIcon />}
                            onClick={() => {
                              setLayerSizes((ls) => {
                                const lsNew = [...ls];
                                lsNew[l] += 1;
                                return lsNew;
                              });
                              setModelUpdated.on();
                              pause();
                            }}
                            size="xs"
                          />
                        </ButtonGroup>
                      )}
                    </Th>
                  ))}
                </Tr>
              </Thead>
              <Tbody>
                {_.range(Math.max(...layerSizes)).map((row) => (
                  <Tr key={row}>
                    {_.range(layerSizes.length).map((layer) => (
                      <Td key={layer}>
                        {row < layerSizes[layer] && (
                          // <div>hi</div>
                          <Neuron
                            num={row}
                            weights={
                              weights.length && weights[layer]?.length
                                ? weights[layer].map((w) => w[row])
                                : []
                            }
                            bias={
                              biases.length && biases[layer]?.length
                                ? biases[layer][row]
                                : 0
                            }
                            key={layer}
                            desc={
                              layer == layerSizes.length - 1
                                ? "Sigmoid"
                                : "ReLu"
                            }
                          />
                        )}
                      </Td>
                    ))}
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </TableContainer>
        </div>
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
              template: colorMode === "light" ? templateLight : templateDark,
            }}
            config={{ displayModeBar: false }}
          />
        </Stack>
      </Stack>
    </div>
  );
};

export default NeuralNet;
