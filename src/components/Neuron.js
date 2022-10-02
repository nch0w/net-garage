import { Box, Code, Stack, Text } from "@chakra-ui/react";

const Neuron = ({ weights, bias, num, desc }) => {
  return (
    <Box p="5" maxW="320px" borderWidth="1px">
      <Text textTransform="uppercase" fontSize="md" fontWeight="bold">
        {`Neuron ${num}`}
      </Text>
      <Stack direction="row">
        <Code colorScheme="transparent">Weights</Code>
        {weights &&
          weights.map((w, i) => (
            <Code
              colorScheme={w > 0 ? "green" : "red"}
              children={`${w?.toFixed(3)}`}
              key={i}
            />
          ))}
      </Stack>
      <Stack direction="row">
        <Code colorScheme="transparent">Bias</Code>
        <Code colorScheme="gray" children={`${bias?.toFixed(3)}`} />
        {desc && <Code colorScheme="transparent" children={desc} />}
      </Stack>
    </Box>
  );
};

export default Neuron;
