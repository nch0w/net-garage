import React, { useRef, useEffect } from "react";
import _ from "underscore";
import Color from "colorjs.io";

const colors = { 0: "blue", 1: "red" };
const RED = new Color("p3", [1, 0, 0]);
const BLUE = new Color("p3", [0, 0, 1]);
const BLUERED = BLUE.range(RED);

const Plotter = (props) => {
  const { points, hmap } = props;
  const { pointsX, pointsY, labels } = points;

  const canvasRef = useRef(null);

  const draw = (ctx) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const originX = 20;
    const originY = ctx.canvas.height - 20;

    const scaleX = (pX) => pX * (ctx.canvas.width - 30) + originX;
    const scaleY = (pY) => originY - pY * (ctx.canvas.height - 30);

    // axes
    ctx.fillStyle = "#000000";
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(ctx.canvas.width, originY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(originX, 0);
    ctx.stroke();

    ctx.font = "12px Arial";
    ctx.fillText("0", originX - 2, originY + 16);
    ctx.fillText("0", originX - 12, originY + 2);
    ctx.fillText("1", scaleX(1), originY + 16);
    ctx.fillText("1", originX - 12, scaleY(1));
    // hmap
    ctx.globalAlpha = 0.3;
    const n = hmap.length;
    _.range(n).forEach((i) => {
      _.range(n).forEach((j) => {
        ctx.beginPath();
        ctx.rect(
          scaleX(i / n),
          scaleY((j + 1) / n),
          scaleX((i + 1) / n) - scaleX(i / n),
          scaleY(j / n) - scaleY((j + 1) / n)
        );
        // console.log(BLUERED(hmap[i][j]).to("srgb").toString());
        ctx.fillStyle = BLUERED(hmap[i][j]).to("srgb").toString();
        // ctx.fillStyle = "#ff0000";
        // ctx.fillStyle = `rgba(255*hmap[i][j], 0, 255*(1-hmap[i][j]), 0.2)`;
        ctx.fill();
      });
    });
    // points
    ctx.globalAlpha = 1;
    _.range(pointsX.length).forEach((i) => {
      ctx.beginPath();
      ctx.arc(scaleX(pointsX[i]), scaleY(pointsY[i]), 4, 0, 2 * Math.PI);
      ctx.fillStyle = colors[labels[i]];
      ctx.fill();
    });
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    //Our draw come here
    draw(context);
  }, [draw]);

  return <canvas ref={canvasRef} {...props} />;
};

export default Plotter;
