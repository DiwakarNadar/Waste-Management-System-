document.addEventListener('DOMContentLoaded', function () {
    const video = document.getElementById('video');
    const startButton = document.getElementById('startButton');
    const predictButton = document.getElementById('predictButton');
    const cameraDiv = document.querySelector('.camera');
    let isStreaming = false;
    let videoStream;

    // Access the camera and display video stream
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            videoStream = stream;
            video.srcObject = stream;
            video.play();
            isStreaming = true;
        })
        .catch(function(err) {
            console.log("An error occurred: " + err);
        });

    // Capture and display photo
    startButton.addEventListener('click', function() {
        if (isStreaming) {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imgUrl = canvas.toDataURL('image/png');
            const imgElement = document.createElement('img');
            imgElement.src = imgUrl;
            cameraDiv.innerHTML = ''; // Remove video element
            cameraDiv.appendChild(imgElement); // Append captured image to cameraDiv

            // Wait for 3 seconds before converting back to video
            setTimeout(function() {
                cameraDiv.innerHTML = ''; // Remove image element
                cameraDiv.appendChild(video); // Append video element back
                video.srcObject = videoStream; // Set video source
                video.play(); // Start playing video
            }, 3000); // Wait for 3 seconds
        } else {
            console.log("Video stream not available.");
        }
    });

    // Send the captured photo to Flask for prediction
    predictButton.addEventListener('click', function() {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imgData = canvas.toDataURL('image/png');
        fetch('/', {
            method: 'POST',
            body: JSON.stringify({ image: imgData }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(function(response) {
            return response.json();
        })
        .then(function(data) {
            const predictionElement = document.querySelector('.prediction');
            predictionElement.textContent = data.prediction;
        })
        .catch(function(error) {
            console.log('Error:', error);
        });
    });
});
