import * as tf from "@tensorflow/tfjs";

export const generateLinearSeparable = () => {
  const x = tf.round(tf.randomUniform([20, 2]).mul(100)).div(100);

  // separator
  const theta = tf.randomNormal([1, 2]);
  const y = tf
    .sign(x.add([-0.5, -0.5]).matMul(theta.transpose()))
    .add(1)
    .div(2);

  return { x, y };
};
