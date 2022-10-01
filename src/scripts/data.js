import * as tf from '@tensorflow/tfjs';

export const generateLinearSeparable = () => {
    tf.randomUniform([2, 2]).print();
}

