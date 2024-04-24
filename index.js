window.runExample = runExample;
async function runExample() {

    var x = new Float32Array( 1, 4 )

    var x = [];

     x[0] = document.getElementById('box1').value;
     x[1] = document.getElementById('box2').value;
     x[2] = document.getElementById('box3').value;
     x[3] = document.getElementById('box4').value;
 
    let tensorX = new ort.Tensor('float32', x, [1, 4] );
    let feeds = {float_input: tensorX};

    let session = await ort.InferenceSession.create('xgboost_BankNote_ort.onnx');
    
   let result = await session.run(feeds);
   let outputData = result.output1.data;

  outputData = parseFloat(outputData).toFixed(2)

   let predictions = document.getElementById('predictions');

   let isReal = outputData > 0.5 ? "Real" : "Fake";
   predictions.innerHTML = ` <hr> Prediction: <b>${isReal}</b> <br/>`;
 

    
  predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
   <table>
     <tr>
       <td>  Real or Fake  </td>
       <td id="td0">  ${outputData}  </td>
     </tr>
  </table>`;
    
}
