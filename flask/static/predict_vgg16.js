$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop("files")[0];
    reader.readAsDataURL(file);
});

let model;
tf.loadModel('http://localhost:81/models/model.json').then(model_ => {
    $(".progress-bar").hide();
    model = model_
    console.log('Model load!')
});

$("#predict-button").click(function () {
    let image = $("#selected-image").get(0);
    let meanImageNetRGB = tf.tensor1d([123.68,116.779,103.939])
    // let processedTensor  = tensor.sub(meanImageNetRGB);
    let tensor = tf.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .sub(meanImageNetRGB)
        .reverse(2)
        .expandDims();

    // let meanImageNetRGB = {
    //     red: 123.68,
    //     green: 116.779,
    //     blue: 103.939
    // };
    // let indices = [
    //     tf.tensor1d([0], "int32"),
    //     tf.tensor1d([1], "int32"),
    //     tf.tensor1d([2], "int32")
    // ];
    // let centeredRGB = {
    //     red: tf.gather(tensor, indices[0], 2)
    //         .sub(tf.scalar(meanImageNetRGB.red))
    //         .reshape([50176]),
    //     green: tf.gather(tensor, indices[1], 2)
    //         .sub(tf.scalar(meanImageNetRGB.green))
    //         .reshape([50176]),
    //     blue: tf.gather(tensor, indices[2], 2)
    //         .sub(tf.scalar(meanImageNetRGB.blue))
    //         .reshape([50176])
    // };
    // let processedTensor = tf.stack([
    //         centeredRGB.red, centeredRGB.green, centeredRGB.blue
    //     ], 1)
    //     .reshape([224, 224, 3])
    //     .reverse(2)
    //     .expandDims();

    model.predict(tensor).data().then(predictions => {
        // console.log(predictions)
        let top10 = Array.from(predictions)
            .map(function (p, i) {
                return {
                    probability: p,
                    className: IMAGENET_CLASSES_VGG16[i]
                };
            })
            .sort(function (a, b) {
                return b.probability - a.probability;
            }).slice(0, 10);

        $("#prediction-list").empty();
        top10.forEach(function (p) {
            $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
        });
    });
});