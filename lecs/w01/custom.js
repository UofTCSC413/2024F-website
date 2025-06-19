Reveal.on('ready', () => {
  // Create a new button
  const btn = document.createElement('a');
  btn.href = 'https://example.com'; // Replace with your desired link
  btn.target = '_blank';
  btn.title = 'Open Course Material';
  btn.innerHTML = '📘'; // or use <img src="..." /> for custom icon

  // Style the button
  btn.style.marginLeft = '10px';
  btn.style.fontSize = '20px';
  btn.style.textDecoration = 'none';

  // Insert into the reveal control area (bottom right)
  const controls = document.querySelector('.reveal .controls');
  if (controls) {
    controls.appendChild(btn);
  }
});
