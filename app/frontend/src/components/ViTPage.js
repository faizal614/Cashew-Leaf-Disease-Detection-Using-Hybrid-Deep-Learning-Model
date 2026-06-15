import { useState, useRef } from "react";

function ViTPage() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef(null);

  const handleImage = (file) => {
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImage(e.dataTransfer.files[0]);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current.click();
  };

  const handlePredict = async () => {
    if (!image) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    const res = await fetch("http://localhost:8000/predict/vit", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="card">
      <h3>Vision Transformer (ViT)</h3>

      <div
        className={`upload-box ${dragActive ? "active" : ""}`}
        onClick={handleBrowseClick}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        <p>Drag & drop leaf image here</p>
        <span>or click to browse</span>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => handleImage(e.target.files[0])}
          hidden
        />
      </div>

      {preview && (
        <img src={preview} alt="preview" className="preview-image" />
      )}

      <button
        className="predict-btn"
        onClick={handlePredict}
        disabled={loading}
      >
        {loading ? "Analyzing Leaf..." : "Predict Disease"}
      </button>

      {result && (
        <div className="result">
          <p><b>Status:</b> {result.status}</p>
          <p><b>Disease:</b> {result.prediction}</p>

          <div className="confidence-container">
            <span><b>Confidence:</b> {result.confidence}%</span>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${result.confidence}%` }}
              ></div>
            </div>
          </div>

          {result.prescription && (
            <div className="prescription-box">
              <h4>Cause</h4>
              <p>{result.prescription.cause}</p>

              <h4>Treatment</h4>
              <ul>
                {result.prescription.treatment.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>

              <h4>Prevention</h4>
              <ul>
                {result.prescription.prevention.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ViTPage;