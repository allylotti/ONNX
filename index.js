document.addEventListener('DOMContentLoaded', (event) => {
    console.log('DOM fully loaded and parsed');
    
    async function runExample() {
        console.log('runExample is called'); // Check if runExample is triggered

        // Initialize the Float32Array with the correct length
        const x = new Float32Array(4);
        x[0] = parseFloat(document.getElementById('box1').value);
        x[1] = parseFloat(document.getElementById('box2').value);
        x[2] = parseFloat(document.getElementById('box3').value);
        x[3] = parseFloat(document.getElementById('box4').value);

        // Log the input values to verify they are being read correctly
        console.log('Input values:', x);

        try {
            console.log('Creating inference session...');
            const session = new onnx.InferenceSession();

            console.log('Loading the model...');
            await session.loadModel("./DLnet_BanknoteData.onnx");
            console.log('Model loaded successfully.');

            // Create the tensor with the correct shape
            const tensorX = new onnx.Tensor(x, 'float32', [1, 4]);

            console.log('Running inference...');
            const outputMap = await session.run({ input1: tensorX });
            const outputData = outputMap.get('output1');

            // Display the prediction result
            const predictions = document.getElementById('predictions');
            predictions.innerHTML = `<hr>Got an output tensor with values:<br/>
                                     <table><tr><td>Real or Fake</td>
                                     <td id="td0">${outputData.data[0].toFixed(2)}</td>
                                     </tr></table>`;
            console.log('Inference completed. Output:', outputData.data);
        } catch (error) {
            console.error('Error during model loading or inference:', error);
        }
    }
    
    // Make the runExample function available to the window object
    window.runExample = runExample;
});
