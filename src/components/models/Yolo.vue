<template>
  <WebcamModel
    modelName="Yolo"
    :hasWebGL="hasWebGL"
    :modelFilepath="modelFilepath"
    :imageSize="224"
    :imageUrls="imageUrls"
    :warmupModel="warmupModel"
    :preprocess="preprocess"
    :postprocess="postprocess"
  ></WebcamModel>
</template>

<script lang="ts">
import ndarray from "ndarray";
import ops from "ndarray-ops";
import WebcamModel from "../common/WebcamModelUI.vue";
import { Vue, Component, Prop } from "vue-property-decorator";
import { runModelUtils, yolo, yoloTransforms } from "../../utils/index";
import { YOLO_IMAGE_URLS } from "../../data/sample-image-urls";
import { Tensor, InferenceSession } from "onnxruntime-web";

const MODEL_FILEPATH_PROD = `/onnxruntime-web-demo/yolo.onnx`;
const MODEL_FILEPATH_DEV = "/Vladimir Putin.onnx";

@Component({
  components: {
    WebcamModel,
  },
})
export default class Yolo extends Vue {
  @Prop(Boolean) hasWebGL!: boolean;
  imageUrls: Array<{ text: string; value: string }>;
  modelFilepath: string;

  constructor() {
    super();
    this.imageUrls = YOLO_IMAGE_URLS;
    this.modelFilepath =
      process.env.NODE_ENV === "production"
        ? MODEL_FILEPATH_PROD
        : MODEL_FILEPATH_DEV;
  }

  warmupModel(session: InferenceSession) {
    return runModelUtils.warmupModel(session, [1, 224, 224, 3]);
  }

  preprocess(ctx: CanvasRenderingContext2D): Tensor {
    const imageData = ctx.getImageData(
      0,
      0,
      224,
      224
    );
    const { data, width, height } = imageData;
    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    console.log(dataTensor)
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
      1,
      width,
      height,
      3,
    ]);

    ops.assign(
      dataProcessedTensor.pick(0, null, null, 0),
      dataTensor.pick(null, null, 0)
    );
    ops.assign(
      dataProcessedTensor.pick(0, null, null, 1),
      dataTensor.pick(null, null, 1)
    );
    ops.assign(
      dataProcessedTensor.pick(0, null, null, 2),
      dataTensor.pick(null, null, 2)
    );

    ops.divseq(dataProcessedTensor, 255);

    const tensor = new Tensor("float32", new Float32Array(width * height * 3), [
      1,
      width,
      height,
      3,
    ]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  }

  async postprocess(tensor: Tensor, inferenceTime: number) {
    try {
      console.log(tensor)
      const originalOutput = new Tensor(
        "float32",
        tensor.data as Float32Array,
        [1, 224, 224, 3]
      );
      console.log(originalOutput.data.map(x => x*255))
      const intData = new Uint8ClampedArray(originalOutput.data.map(x => x*255))
      const canvas = document.getElementById('input-canvas');
      const ctx = canvas.getContext('2d');
      const imageData = ctx.createImageData(224, 224);
      var width = 224,
      height = 224,
      buffer = new Uint8ClampedArray(width * height * 4); // have enough bytes
      for(var y = 0; y < height; y++) {
        for(var x = 0; x < width; x++) {
            var pos = (y * width + x) * 4; // position in buffer based on x and y
            var pos2 = (y * width + x) * 3; // position in buffer based on x and y
            buffer[pos  ] = intData[pos2]           // some R value [0, 255]
            buffer[pos+1] = intData[pos2+1];           // some G value
            buffer[pos+2] = intData[pos2+2];           // some B value
            buffer[pos+3] = 255;           // set alpha channel
        }
      }
      imageData.data.set(buffer);
      ctx.putImageData(imageData, 0, 0);

    } catch (e) {
      alert("Model is not valid!");
      console.log(e)
    }
  }

  drawRect(
    x: number,
    y: number,
    w: number,
    h: number,
    text = "",
    color = "red"
  ) {
    const webcamContainerElement = document.getElementById("webcam-container") as HTMLElement;
    // Depending on the display size, webcamContainerElement might be smaller than 416x416.
    const [ox, oy] = [(webcamContainerElement.offsetWidth - 416) / 2, (webcamContainerElement.offsetHeight - 416) / 2];
    const rect = document.createElement("div");
    rect.style.cssText = `top: ${y+oy}px; left: ${x+ox}px; width: ${w}px; height: ${h}px; border-color: ${color};`;
    const label = document.createElement("div");
    label.innerText = text;
    rect.appendChild(label);

    webcamContainerElement.appendChild(
      rect
    );
  }
}
</script>