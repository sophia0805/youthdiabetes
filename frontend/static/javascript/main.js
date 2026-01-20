// Enhanced JS for mobile nav toggle, animations, and interactive features
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

  // Number counter logic for prediabetes counter
  const numberCounterElement = document.getElementById('numberCounter');
  if (numberCounterElement) {
    let currentNumber = 8000000;
    const targetNumber = 8400000;

    const incrementCounter = () => {
      if (currentNumber < targetNumber) {
        currentNumber += 40000;
        numberCounterElement.textContent = currentNumber.toLocaleString();
        setTimeout(incrementCounter, 50);
      } else {
        numberCounterElement.textContent = targetNumber.toLocaleString();
      }
    };

    // Start counter when element is visible
    const counterObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          incrementCounter();
          counterObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });
    
    if (numberCounterElement) {
      counterObserver.observe(numberCounterElement);
    }
  }

  // Statistics counter animation
  const animateStatCounters = () => {
    const statValues = document.querySelectorAll('.stat-value');
    
    statValues.forEach(stat => {
      const target = parseInt(stat.getAttribute('data-target'));
      if (!target) return;
      
      const duration = 2000; // 2 seconds
      const steps = 60;
      const increment = target / steps;
      let current = 0;
      const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
          stat.textContent = target + (target === 33 || target === 700 ? '%' : '%');
          clearInterval(timer);
        } else {
          stat.textContent = Math.floor(current) + (target === 33 || target === 700 ? '%' : '%');
        }
      }, duration / steps);
    });
  };

  // Intersection Observer for fade-in animations
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        
        // Trigger stat counter animation when stats section is visible
        if (entry.target.querySelector('.stats-grid')) {
          setTimeout(animateStatCounters, 300);
        }
      }
    });
  }, observerOptions);

  // Observe all fade-in elements
  document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

  // FAQ Toggle functionality
  const faqQuestions = document.querySelectorAll('.faq-question');
  faqQuestions.forEach(question => {
    question.addEventListener('click', () => {
      const faqItem = question.closest('.faq-item');
      const isActive = faqItem.classList.contains('active');
      
      // Close all FAQ items
      document.querySelectorAll('.faq-item').forEach(item => {
        item.classList.remove('active');
      });
      
      // Open clicked item if it wasn't active
      if (!isActive) {
        faqItem.classList.add('active');
      }
    });
  });

  // Form progress indicator
  const evaluationForm = document.getElementById('evaluationForm');
  if (evaluationForm) {
    // Create progress container
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress-container';
    progressContainer.innerHTML = `
      <div class="progress-bar">
        <div class="progress-fill" id="progressFill"></div>
      </div>
      <div class="progress-text" id="progressText">Form Completion: 0%</div>
    `;
    evaluationForm.insertBefore(progressContainer, evaluationForm.firstChild);

    const updateProgress = () => {
      const requiredFields = evaluationForm.querySelectorAll('select[required], input[required]');
      const filledFields = Array.from(requiredFields).filter(field => field.value !== '');
      const progress = Math.round((filledFields.length / requiredFields.length) * 100);
      
      const progressFill = document.getElementById('progressFill');
      const progressText = document.getElementById('progressText');
      
      if (progressFill && progressText) {
        progressFill.style.width = progress + '%';
        progressText.textContent = `Form Completion: ${progress}%`;
      }
    };

    // Update progress on any field change
    evaluationForm.addEventListener('change', updateProgress);
    evaluationForm.addEventListener('input', updateProgress);
    
    // Initial progress check
    updateProgress();
  }

  // Add smooth scroll behavior for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
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
