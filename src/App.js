import "./App.css";
import NeuralNet from "./components/NeuralNet";
import { ChakraProvider } from "@chakra-ui/react";

function App() {
  return (
    <ChakraProvider>
      <div className="App">
        <NeuralNet />
      </div>
    </ChakraProvider>
  );
}

export default App;
