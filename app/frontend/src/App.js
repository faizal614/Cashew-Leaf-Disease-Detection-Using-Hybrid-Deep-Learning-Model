import { useState } from "react";
import Navbar from "./components/Navbar";
import ViTPage from "./components/ViTPage";
import EfficientPage from "./components/EfficientPage";
import HybridPage from "./components/HybridPage";
import "./App.css";

function App() {
  const [activePage, setActivePage] = useState("vit");

  const renderPage = () => {
    if (activePage === "vit") return <ViTPage />;
    if (activePage === "efficientnet") return <EfficientPage />;
    if (activePage === "hybrid") return <HybridPage />;
  };

  return (
    <div className="app">
      <Navbar setActivePage={setActivePage} />
      <div className="content">
        {renderPage()}
      </div>
    </div>
  );
}

export default App;
