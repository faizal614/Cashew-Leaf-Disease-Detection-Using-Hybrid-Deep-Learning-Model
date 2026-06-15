function Navbar({ setActivePage }) {
  return (
    <nav className="navbar">
      <h2 className="logo">🌿 Cashew Leaf AI</h2>

      <ul className="nav-links">
        <li onClick={() => setActivePage("vit")}>ViT</li>
        <li onClick={() => setActivePage("efficientnet")}>EfficientNet</li>
        <li onClick={() => setActivePage("hybrid")}>Hybrid ViT + EfficientNet</li>
      </ul>
    </nav>
  );
}

export default Navbar;
