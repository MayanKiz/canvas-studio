let handLandmarker = null;

self.onmessage = async (event) => {
  const { type, bitmap, timestamp } = event.data;

  if (type === 'init') {
    try {
      const { FilesetResolver, HandLandmarker } = await import(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs'
      );
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
      );
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });
      self.postMessage({ type: 'ready' });
    } catch (error) {
      self.postMessage({ type: 'error', message: error?.message || 'Worker initialization failed' });
    }
    return;
  }

  if (type === 'detect' && handLandmarker && bitmap) {
    try {
      const results = handLandmarker.detectForVideo(bitmap, timestamp);
      self.postMessage({ type: 'result', landmarks: results.landmarks || [] });
    } catch (error) {
      self.postMessage({ type: 'error', message: error?.message || 'Worker detection failed' });
    } finally {
      bitmap.close();
    }
  }
};
