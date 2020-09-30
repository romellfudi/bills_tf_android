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

$("#predict-button").click(function(){ 
        let image = $("#selected-image").get(0);
        let tensor = tf.fromPixels(image)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .expandDims();

        model.predict(tensor).data().then(predictions => {
            console.log(predictions)
            let top4 = Array.from(predictions)
            .map(function (p, i) {
                return {
                    probability: p,
                    className: IMAGENET_CLASSES[i]
                };
            })
            .sort(function (a, b) {
                return b.probability - a.probability;
            }).slice(0, 4);
                            
            $("#prediction-list").empty();
            top4.forEach(function (p) {
                $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
            });
        });  
});     