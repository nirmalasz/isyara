function getCsrfToken() {
  return document.cookie
    .split('; ')
    .find(row => row.startsWith('csrftoken='))
    ?.split('=')[1];
}

async function sendChatMessage(message) {
  const response = await fetch('/api/chatbot/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCsrfToken(),
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error('Chatbot request failed');
  }

  return response.json();
}

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('chatbot-form');
  const input = document.getElementById('chatbot-input');
  const messagesContainer = document.getElementById('chatbot-messages');

  if (!form) return;

  function appendMessage(text, isUser, videoUrl) {
    const bubble = document.createElement('div');
    bubble.className = isUser ? 'chatbot-bubble chatbot-bubble-user' : 'chatbot-bubble chatbot-bubble-bot';
    bubble.textContent = text;
    messagesContainer.appendChild(bubble);

    if (videoUrl) {
      const link = document.createElement('a');
      link.href = videoUrl;
      link.target = '_blank';
      link.className = 'chatbot-video-link';
      link.textContent = 'Tonton video referensi →';
      messagesContainer.appendChild(link);
    }

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    appendMessage(message, true);
    input.value = '';

    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'chatbot-bubble chatbot-bubble-bot chatbot-typing';
    typingIndicator.textContent = 'Mengetik...';
    messagesContainer.appendChild(typingIndicator);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    try {
      const result = await sendChatMessage(message);
      typingIndicator.remove();
      appendMessage(result.reply, false, result.video_url);
    } catch (err) {
      typingIndicator.remove();
      appendMessage('Maaf, terjadi kesalahan. Coba lagi.', false);
    }
  });
});

document.addEventListener('DOMContentLoaded', () => {
  const toggleBtn = document.getElementById('chatbot-toggle');
  const closeBtn = document.getElementById('chatbot-close');
  const panel = document.getElementById('chatbot-panel');

  toggleBtn?.addEventListener('click', () => {
    panel.classList.toggle('hidden');
  });

  closeBtn?.addEventListener('click', () => {
    panel.classList.add('hidden');
  });
});