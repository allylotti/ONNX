async function runExample() {
    try {
        let x = [
            parseFloat(document.getElementById('box1').value),
            parseFloat(document.getElementById('box2').value),
            parseFloat(document.getElementById('box3').value),
            parseFloat(document.getElementById('box4').value)
        ];

        let tensorX = new ort.Tensor('float32', x, [1, 4]);
        let feeds = { float_input: tensorX };

        let session = await ort.InferenceSession.create('xgboost_BankNote_ort.onnx');
        let result = await session.run(feeds);

        // Make sure 'output1' is the correct output name
        let outputData = result.output1.data;

        // Ensure the output is correctly interpreted as a number
        outputData = parseFloat(outputData).toFixed(2);

        let predictions = document.getElementById('predictions');
        predictions.innerHTML = ` <hr> Genuine / Forged: <br/>
            <table>
                <tr>
                    <td>  Real or Fake  </td>
                    <td id="td0">  ${outputData}  </td>
                </tr>
            </table>`;
    } catch (error) {
        console.error('Error running model inference', error);
    }
}

window.runExample = runExample;
