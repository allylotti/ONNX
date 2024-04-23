document.addEventListener('DOMContentLoaded', (event) => {
    async function runExample() {
        try {
            // Initialize the Float32Array with the correct length
            const x = new Float32Array(4);
            x[0] = parseFloat(document.getElementById('box1').value);
            x[1] = parseFloat(document.getElementById('box2').value);
            x[2] = parseFloat(document.getElementById('box3').value);
            x[3] = parseFloat(document.getElementById('box4').value);
    
            // Create the tensor with the correct shape
            const tensorX = new onnx.Tensor(x, 'float32', [1, 4]);
    
            // Create an inference session with default settings.
            const session = new onnx.InferenceSession();
            // Load the ONNX model file from the path specified.
            await session.loadModel("./DLnet_BanknoteData.onnx");
    
            // Run the model with the input tensor and get the output.
            const outputMap = await session.run({ input1: tensorX });
            const outputData = outputMap.get('output1');
    
            // Display the prediction result
            const predictions = document.getElementById('predictions');
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
    
    // Make the runExample function available to the window object
    window.runExample = runExample;
});
