
const express = require('express')
const multer  = require('multer')

const tf = require('@tensorflow/tfjs');
const tfjsNode = require('@tensorflow/tfjs-node');
const mobilenet= require('@tensorflow-models/mobilenet');

const fs = require("fs");

const upload = multer({ dest: 'uploads/' })

const app = express();
const port = 3000;

app.post('/recognize', upload.single('face'), async function (req, res, next) {

    const imageDataArrayBuffer = fs.readFileSync(req.file.path);

    const imageData = new Uint8Array(imageDataArrayBuffer);

    const imageTensor = tfjsNode.node.decodeJpeg(imageData);

    const model = await mobilenet.load();

    const tempPrediction = await model.classify(imageTensor);

    return res.send(JSON.stringify(tempPrediction));
})

app.get('/', (req, res) => {
  res.send('Post a JPEG/JPG image at /recognize path and see the magic.')
})

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})