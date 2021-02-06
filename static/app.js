const chat_input = document.querySelector("#chatbot-text-input");
const chat_messages =  document.querySelector("#chat-messages");

const url = window.location.href;

function add_message(author, text) {
  chat_messages.innerHTML += `<div class="${author}">${text}</div>`
}

function clear() {
  chat_input.value = ""
}

function handle_chatbot_response(headers, url, json_data) {
  var xhr = new XMLHttpRequest();

  xhr.open("POST", url, true);
      for (key in headers) {
        xhr.setRequestHeader(key, headers[key]);
      }

      xhr.send(JSON.stringify(json_data));

      xhr.onload = () => {
          var data = JSON.parse(xhr.responseText);
          add_message("bot", JSON.parse(JSON.stringify(data['data']['text'])))
      };
}

const headers = {
  'Content-Type': 'application/json'}

chat_input.addEventListener('keypress', (event) => {
  if (event.key === 'Enter') {
    if (chat_input.value != ""){
      add_message("user", chat_input.value)
      handle_chatbot_response(headers, url, {"text": chat_input.value})

      clear()


      chat_messages.scrollIntoView();
  }
}
});

