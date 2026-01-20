// Minimal JS for mobile nav toggle and active link handling
document.addEventListener('DOMContentLoaded', ()=>{
  const btn = document.getElementById('hamburger')
  const drawer = document.getElementById('drawer')
  if(btn && drawer){
    btn.addEventListener('click', ()=> drawer.classList.toggle('open'))
    drawer.addEventListener('click', (e)=>{ if(e.target.id === 'drawer') drawer.classList.remove('open') })
  }
  // set active link based on pathname
  const links = document.querySelectorAll('.nav a, .panel a')
  links.forEach(l => {
    if(l.getAttribute('href') === window.location.pathname.split('/').pop() || (l.getAttribute('href') === 'index.html' && window.location.pathname.endsWith('/'))){
      l.classList.add('active')
    }
  })

  // Number counter logic
  const numberCounterElement = document.getElementById('numberCounter');
  let currentNumber = 8000000;
  const targetNumber = 8400000;

  const incrementCounter = () => {
    if (currentNumber < targetNumber) {
      currentNumber += 1000;
      numberCounterElement.textContent = currentNumber;
      setTimeout(incrementCounter, 10); // Adjust speed here
    }
  };

  incrementCounter();
})


// WARNING: Storing your API key in frontend code is INSECURE.
// This is for demonstration purposes only.
const API_KEY = 'YOUR_API_KEY';

async function sendRequest() {
    const userInput = document.getElementById('userInput').value;
    const responseArea = document.getElementById('responseArea');

    if (!userInput) return;

    responseArea.textContent = 'Thinking...';

    try {
        const response = await fetch('api.openai.com', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_KEY}`
            },
            body: JSON.stringify({
                model: 'gpt-3.5-turbo', // The model to use
                messages: [
                    { role: 'system', content: 'You are a helpful assistant.' },
                    { role: 'user', content: userInput }
                ],
                max_tokens: 150, // Limits the response length
            })
        });

        const data = await response.json();

        if (data.choices && data.choices.length > 0) {
            responseArea.textContent = data.choices[0].message.content.trim();
        } else {
            responseArea.textContent = 'No response received.';
        }

    } catch (error) {
        console.error('Error:', error);
        responseArea.textContent = 'Error occurred while fetching data.';
    }
}