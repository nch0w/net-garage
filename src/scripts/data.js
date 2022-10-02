import * as tf from "@tensorflow/tfjs";

export const generateLinearSeparable = () => {
  const inputs = tf
    .round(tf.randomUniform([20, 2], 0.1, 0.9).mul(100))
    .div(100);

  // separator
  const theta = tf.randomNormal([1, 2]);
  const labels = tf
    .sign(inputs.add([-0.5, -0.5]).matMul(theta.transpose()))
    .add(1)
    .div(2);

  return { inputs, labels };
};
