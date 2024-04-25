
async function runExample() {

  var x = [];

  x[0] = document.getElementById('box1').value;
  x[1] = document.getElementById('box2').value;
  x[2] = document.getElementById('box3').value;
  x[3] = document.getElementById('box4').value;
 
  let tensorX = new onnx.Tensor(x, 'float32', [1, 4]);

  let session = new onnx.InferenceSession();

  await session.loadModel("DLnet_BanknoteData.onnx");
  let outputMap = await session.run([tensorX]);
  let outputData = outputMap.get('output1');

  
  let predictions = document.getElementById('predictions');
  predictions.innerHTML = ` <hr> Got an output tensor with value: <br />
  <table>
     <tr>
        <td>  Real or Fake  </td>
        <td id="td0">  ${outputData.data[0].toFixed(2)}  </td>
     </tr>
  </table>
  `;


  
}
