async function runExample() {
    // Initialize the Float32Array with the correct length
    var x = new Float32Array(4);
    x[0] = parseFloat(document.getElementById('box1').value);
    x[1] = parseFloat(document.getElementById('box2').value);
    x[2] = parseFloat(document.getElementById('box3').value);
    x[3] = parseFloat(document.getElementById('box4').value);

    // Create the tensor with the correct shape
    let tensorX = new onnx.Tensor(x, 'float32', [1, 4]);

    try {
        let session = new onnx.InferenceSession();
        await session.loadModel("./DLnet_BanknoteData.onnx");
        let outputMap = await session.run({ input1: tensorX }); // Ensure the input name is correct
        let outputData = outputMap.get('output1');

        // Display the prediction result
        let predictions = document.getElementById('predictions');
        predictions.innerHTML = `<hr> Got an output tensor with values: <br/>
        <table>
            <tr>
                <td>Real or Fake</td>
                <td id="td0">${outputData.data[0].toFixed(2)}</td>
            </tr>
        </table>`;
    } catch (error) {
        console.error('Error during model loading or inference:', error);
    }
}
