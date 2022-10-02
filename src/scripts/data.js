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

export const generateCircle = () => {
  const n1 = 50;
  const points0 = [];
  const points1 = [];
  const labels0 = [];
  const labels1 = [];
  const r0min = 0;
  const r0max = 0.15;
  const r1min = 0.35;
  const r2max = 0.45;
  for (let i = 0; i < n1; i++) {
    const r = Math.random() * (r0max - r0min) + r0min;
    const theta = Math.random() * 2 * Math.PI;
    points0.push([r * Math.cos(theta), r * Math.sin(theta)]);
    labels0.push(0);
  }
  for (let i = 0; i < n1; i++) {
    const r = Math.random() * (r2max - r1min) + r1min;
    const theta = Math.random() * 2 * Math.PI;
    points1.push([r * Math.cos(theta), r * Math.sin(theta)]);
    labels1.push(1);
  }
  const inputs = tf.tensor2d([...points0, ...points1]).add([0.5, 0.5]);
  const labels = tf.tensor1d([...labels0, ...labels1]);
  // TODO shuffle
  return { inputs, labels };
};
