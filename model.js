const Model = async() => {
    const surface1 = document.getElementById("model1");
    const surface2 = document.getElementById("model2");

    const data = await fetch("./carsData.json");
    let cars = await data.json();
    
    cars = cars.map(car => {
        return car = extractData(car);
    }).filter(car => removeErrors(car));
    
    plotData(cars, surface1);

    const inputs = cars.map(car => car.x);
    const labels = cars.map(car => car.y);

    // Convert arrays to 2d tensors
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Normalize data
    const inputMin = inputTensor.min();
    const inputMax = inputTensor.max();
    const labelMin = labelTensor.min();
    const labelMax = labelTensor.max();

    const nmInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const nmLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
    
    // Define a model
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    model.add(tf.layers.dense({units: 1, useBias: true}));

    // Compile the model
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    // Train model
    await trainModel(model, nmInputs, nmLabels, surface2);

    // unNormalize the data
    let unX = tf.linspace(0, 1 , 100);
    let unY = model.predict(unX.reshape([100, 1]));
    
    const unNormalize_unX = unX.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormalize_unY = unY.mul(labelMax.sub(labelMin)).add(labelMin);

    unX = unNormalize_unX.dataSync();
    unY = unNormalize_unY.dataSync();

    // Plot the result
    const predicted = Array.from(unX).map((val, i) => {
        return {x: val, y: unY[i]}
    });

    plotData([cars, predicted], surface1);
}

// Clean data
function extractData(obj) {
    obj =  { x:obj.Horsepower, y: obj.Miles_per_Gallon }
    return obj;
}

// Remove errors if x or y is null
function removeErrors(obj) {
    return obj.x != null && obj.y != null;
}

// plot data
function plotData(values, surface) {
    tfvis.render.scatterplot(surface, 
        {values: values, series: ['Original','Predicted']},
        {xLabel:'Horsepower', yLabel:'Miles_per_Gallon'}
    );
}

// train the model
async function trainModel(model, inputs, labels, surface) {
    const batchSize = 25;
    const epochs = 100;
    const callbacks = tfvis.show.fitCallbacks(surface, ['loss'], {callbacks: ['onEpochEnd']})
    return await model.fit(inputs, labels,
        {batchSize, epochs, shuffle: true, callbacks: callbacks}
    );
}

Model()