import { useEffect } from "react";
import * as tf from '@tensorflow/tfjs';
import Plot from 'react-plotly.js';
import { generateLinearSeparable } from "../scripts/data";

const NeuralNet = (props) => {
    useEffect(() => {
        console.log("NeuralNet useEffect");
        const model = tf.sequential();
        model.add(tf.layers.dense({units: 1, inputShape: [2]}));
        console.log(JSON.stringify(model.outputs[0].shape));

        generateLinearSeparable();
    }, [])
    return <div>
        <Plot
        data={[
          {
            x: [0.1, 0.2, 0.5],
            y: [0.4, 0.3, 0.1],
            type: 'scatter',
            mode: 'markers',
            marker: {color: 'red'},
          }
        ]}
        layout={{width: 300, height: 300, xaxis: { range: [0, 1] }, yaxis: {range: [0, 1]}, showlegend: false}}
        config={{staticPlot: true}}
        
      />
    </div> 
}

export default NeuralNet;