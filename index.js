async function runExample() {
    let x = new Float32Array(4);

    x[0] = parseFloat(document.getElementById('box1').value);
    x[1] = parseFloat(document.getElementById('box2').value);
    x[2] = parseFloat(document.getElementById('box3').value);
    x[3] = parseFloat(document.getElementById('box4').value);

    let tensorX = new onnx.Tensor(x, 'float32', [1, 4]);
    let session = new onnx.InferenceSession();
    await session.loadModel("./DLnet_BanknoteData.onnx");
    
    // Make sure 'input1' matches the expected input name of the model
    let inputName = 'input1'; // This should be the actual input name expected by the model
    let outputMap = await session.run({ [inputName]: tensorX });
    let outputData = outputMap.get('output1');

    let predictions = document.getElementById('predictions');
    predictions.innerHTML = `
        <hr> Got an output tensor with values: <br/>
        <table>
            <tr>
                <td>Real or Fake</td>
                <td id="td0">${outputData.data[0].toFixed(2)}</td>
            </tr>
        </table>`;
}
