import { Box, Code, Stack, Text } from "@chakra-ui/react";
import Popup from "./popup";

const Neuron = ({ weights, bias, num, desc }) => {
  return (
    <Box p="5" maxWidth={220} borderWidth="1px">
      <Text textTransform="uppercase" fontSize="md" fontWeight="bold">
        {`Neuron ${num}`}
      </Text>
      <Stack direction="row" flexWrap="wrap">
        <Popup
          title="Weights"
          description={`The weights describe how the data
          or outputs from previous neurons are linearly transformed to
          create the input for the neuron. Note that there is one weight for
          each neuron in the previous layer (except for the first layer,
          in which case there are two weights due to the 2D data.)`}
        >
          <Code colorScheme="transparent">Weights</Code>
        </Popup>

        {weights &&
          weights.map((w, i) => (
            <Code
              colorScheme={w > 0 ? "green" : "red"}
              children={`${w?.toFixed(3)}`}
              key={i}
            />
          ))}
      </Stack>
      <Stack direction="row" flexWrap="wrap">
        <Popup
          title="Bias"
          description={`After multiplying the previous outputs by the weights,
          we add a constant term known as the bias. This allows the neuron's
          output to shift independently of its input.`}
        >
          <Code colorScheme="transparent">Bias</Code>
        </Popup>

        <Code colorScheme="gray" children={`${bias?.toFixed(3)}`} />
        {desc &&
          (desc === "Sigmoid" ? (
            <Popup
              title="Sigmoid Activation"
              description={`The final ouput of our network is a 
          number between 0 to 1 representing
          the network's confidence that the sample is blue (0) or red (1).
          Because the matrix multiplications leading to this neuron may have a
          value outside this range, we scale it down using the sigmoid
          activation function.`}
            >
              <Code colorScheme="transparent" children={desc} />
            </Popup>
          ) : (
            <Popup
              title="ReLu Activation"
              description={`The rectified linear unit (ReLu) is the function max(x, 0). It is a common activation function (applied after combining weights and biases), because it
              is simple and adds non-linearity to a neuron.`}
            >
              <Code colorScheme="transparent" children={desc} />
            </Popup>
          ))}
      </Stack>
    </Box>
  );
};

export default Neuron;
